"""SimCLR-style contrastive pretraining for EEG encoder (v3).

Research rationale
──────────────────
Supervised EEG classification on DEAP plateaus at ~25 % val accuracy because:
1. Extreme class imbalance (47:1) limits what supervised loss can learn.
2. The label semantics (valence/arousal thresholds) are noisy.

Contrastive pretraining lets the encoder discover *structure* in the EEG
feature space without relying on labels.  Two augmented views of the same
sample are pulled together while different samples are pushed apart
(NT-Xent / InfoNCE).  After pretraining, the encoder is fine-tuned with
fewer epochs of supervised focal loss.

Augmentations (designed for DE features with shape (32, 5)):
- Gaussian noise  – simulates sensor noise
- Temporal masking – zeros random channel rows (simulates electrode dropout)
- Channel dropout  – zeros entire frequency bands (simulates artefact rejection)
- Feature scaling  – random per-channel gain (simulates impedance variation)

Architecture
────────────
encoder → projection head (2-layer MLP with BN + ReLU) → L2-normalised
embeddings → NT-Xent loss with temperature τ = 0.07.

The projection head is discarded after pretraining; only the encoder
weights are kept for downstream tasks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.models.eeg_encoder import EEGEncoder
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


# ======================================================================
# EEG Augmentations (operate on (B, 32, 5) or (B, 160) tensors)
# ======================================================================

class EEGAugmentor:
    """Stochastic augmentation pipeline for EEG differential-entropy features.

    Each call randomly applies a subset of augmentations with independent
    probabilities, producing diverse views for contrastive learning.
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        noise_prob: float = 0.5,
        mask_prob: float = 0.3,
        mask_fraction: float = 0.25,
        channel_drop_prob: float = 0.3,
        channel_drop_frac: float = 0.15,
        scale_prob: float = 0.4,
        scale_range: tuple[float, float] = (0.8, 1.2),
    ) -> None:
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.mask_prob = mask_prob
        self.mask_fraction = mask_fraction
        self.channel_drop_prob = channel_drop_prob
        self.channel_drop_frac = channel_drop_frac
        self.scale_prob = scale_prob
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a batch of EEG features.

        Args:
            x: ``(B, 160)`` or ``(B, 32, 5)``

        Returns:
            Augmented tensor with same shape.
        """
        was_flat = x.dim() == 2
        if was_flat:
            x = x.view(x.size(0), 32, 5)

        x = x.clone()
        B, C, F_ = x.shape  # B, 32, 5

        # 1. Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. Temporal / channel-row masking
        if torch.rand(1).item() < self.mask_prob:
            n_mask = max(1, int(C * self.mask_fraction))
            mask_idx = torch.randperm(C)[:n_mask]
            x[:, mask_idx, :] = 0.0

        # 3. Channel (frequency-band) dropout
        if torch.rand(1).item() < self.channel_drop_prob:
            n_drop = max(1, int(F_ * self.channel_drop_frac))
            drop_idx = torch.randperm(F_)[:n_drop]
            x[:, :, drop_idx] = 0.0

        # 4. Feature scaling (per-channel random gain)
        if torch.rand(1).item() < self.scale_prob:
            lo, hi = self.scale_range
            gains = torch.empty(1, C, 1, device=x.device).uniform_(lo, hi)
            x = x * gains

        if was_flat:
            x = x.view(x.size(0), -1)
        return x


# ======================================================================
# NT-Xent (Normalised Temperature-scaled Cross-Entropy) Loss
# ======================================================================

class NTXentLoss(nn.Module):
    """NT-Xent loss from SimCLR (Chen et al., 2020).

    For a batch of N samples producing 2N views (z_i, z_j), the loss
    for positive pair (i, j) is:

        ℓ(i,j) = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )

    We use cosine similarity and average over all 2N positive pairs.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i, z_j: ``(B, D)`` L2-normalised projection embeddings.

        Returns:
            Scalar NT-Xent loss.
        """
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)          # (2B, D)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2B, 2B)
        sim /= self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_i = torch.arange(B, device=sim.device)
        pos_j = pos_i + B
        labels = torch.cat([pos_j, pos_i])  # (2B,) — ground-truth partner index

        loss = F.cross_entropy(sim, labels)
        return loss


# ======================================================================
# Projection Head
# ======================================================================

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning (discarded after pretraining)."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ======================================================================
# Contrastive EEG Dataset (creates two augmented views per sample)
# ======================================================================

class ContrastiveEEGDataset(Dataset):
    """Yields two augmented views of each EEG sample for contrastive learning."""

    def __init__(self, features: np.ndarray, augmentor: EEGAugmentor) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].unsqueeze(0)  # (1, 160) or (1, 32, 5)
        v1 = self.augmentor(x).squeeze(0)
        v2 = self.augmentor(x).squeeze(0)
        return v1, v2


# ======================================================================
# Contrastive EEG Trainer
# ======================================================================

class ContrastiveEEGTrainer:
    """SimCLR-style contrastive pretraining for EEGEncoder.

    After training, the encoder produces higher-quality embeddings that
    capture EEG structure independently of (noisy) emotion labels.

    Usage::

        trainer = ContrastiveEEGTrainer(cfg)
        trainer.train(features, save_dir)
        # Now load eeg_encoder_contrastive.pt for downstream fine-tuning.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"

        ecfg = cfg.model.eeg_encoder
        self.encoder = EEGEncoder(
            input_dim=ecfg.input_dim,
            hidden_dims=list(ecfg.hidden_dims),
            embedding_dim=ecfg.embedding_dim,
            dropout=ecfg.dropout,
        ).to(self.device)

        self.projector = ProjectionHead(
            input_dim=ecfg.embedding_dim,
            hidden_dim=ecfg.embedding_dim * 2,
            output_dim=64,
        ).to(self.device)

        # Hyperparameters — configurable via v3 section or defaults
        v3 = getattr(cfg, "v3", {})
        cl = v3.get("contrastive", {}) if isinstance(v3, dict) else getattr(v3, "contrastive", {})

        self.epochs = cl.get("eeg_epochs", 100) if isinstance(cl, dict) else getattr(cl, "eeg_epochs", 100)
        self.batch_size = cl.get("batch_size", 512) if isinstance(cl, dict) else getattr(cl, "batch_size", 512)
        self.lr = cl.get("lr", 3e-4) if isinstance(cl, dict) else getattr(cl, "lr", 3e-4)
        self.temperature = cl.get("temperature", 0.07) if isinstance(cl, dict) else getattr(cl, "temperature", 0.07)
        self.weight_decay = cl.get("weight_decay", 1e-4) if isinstance(cl, dict) else getattr(cl, "weight_decay", 1e-4)

    def train(
        self,
        features: np.ndarray,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run contrastive pretraining on raw EEG features.

        Args:
            features: ``(N, 160)`` or ``(N, 32, 5)`` EEG DE features (ALL subjects, no labels needed).
            save_dir: Directory to save encoder checkpoint.

        Returns:
            History dict with ``loss`` values per epoch.
        """
        augmentor = EEGAugmentor()
        dataset = ContrastiveEEGDataset(features, augmentor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )

        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = max(1, self.epochs // 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6,
        )
        scaler = GradScaler('cuda', enabled=self.use_amp)
        criterion = NTXentLoss(temperature=self.temperature)

        history: dict[str, list[float]] = {"loss": []}
        log_every = self.cfg.training.get("log_every", 5) if hasattr(self.cfg.training, "get") else 5
        t0 = time.time()

        logger.info(
            "Contrastive EEG pretraining: %d samples, %d epochs, batch=%d, τ=%.3f",
            len(dataset), self.epochs, self.batch_size, self.temperature,
        )

        for epoch in range(1, self.epochs + 1):
            # Warmup
            if epoch <= warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = self.lr * epoch / warmup_epochs

            self.encoder.train()
            self.projector.train()
            running_loss = 0.0
            n_batches = 0

            for v1, v2 in loader:
                v1 = v1.to(self.device, non_blocking=True)
                v2 = v2.to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp):
                    z1 = self.projector(self.encoder(v1))
                    z2 = self.projector(self.encoder(v2))
                    loss = criterion(z1, z2)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                n_batches += 1

            if epoch > warmup_epochs:
                scheduler.step()

            epoch_loss = running_loss / max(n_batches, 1)
            history["loss"].append(epoch_loss)

            if epoch % log_every == 0 or epoch == 1 or epoch == self.epochs:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "CL-EEG %d/%d  loss=%.4f  lr=%.6f",
                    epoch, self.epochs, epoch_loss, lr_now,
                )
                log_gpu_memory()

        elapsed = time.time() - t0
        logger.info("Contrastive EEG pretraining complete in %.0fs", elapsed)

        # Save encoder weights (no projector — it's discarded)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                self.encoder.state_dict(),
                save_dir / "eeg_encoder_contrastive.pt",
            )
            logger.info("Saved contrastive EEG encoder → %s", save_dir / "eeg_encoder_contrastive.pt")

        return history
