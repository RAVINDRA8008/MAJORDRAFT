"""CPC-style contrastive pretraining for Speech encoder (v3).

Research rationale
──────────────────
The speech encoder (CNN-LSTM) trained with supervised CE on IEMOCAP reaches
~55 % val accuracy.  The temporal structure in MFCC sequences contains rich
information that supervised loss alone underutilises.

Contrastive Predictive Coding (CPC, van den Oord et al., 2018) learns
representations by predicting future latent states from past context.
Combined with data augmentations, this encourages the encoder to capture
emotion-relevant acoustic patterns (pitch contour, energy dynamics, spectral
tilt) rather than memorising speaker-specific features.

Augmentations (designed for MFCC sequences (T, 120)):
- Gaussian noise injection
- Time stretching (interpolate temporal axis)
- Frequency masking (zero random MFCC bands)
- Time masking (zero random temporal segments)

Architecture
────────────
We use the existing SpeechEncoder as the feature extractor.  On top, a
projection head maps embeddings to a contrastive space.  We use NT-Xent
loss (same as EEG) since our encoder produces a single embedding vector
per utterance, making CPC's autoregressive objective unnecessary.

The projection head is discarded after pretraining.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.models.speech_encoder import SpeechEncoder
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


# ======================================================================
# Speech Augmentations (operate on (B, T, F) MFCC tensors)
# ======================================================================

class SpeechAugmentor:
    """Stochastic augmentation pipeline for speech MFCC features.

    Each call randomly applies a subset of augmentations with independent
    probabilities, producing diverse views for contrastive learning.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        noise_prob: float = 0.5,
        time_mask_prob: float = 0.4,
        time_mask_ratio: float = 0.15,
        freq_mask_prob: float = 0.4,
        freq_mask_ratio: float = 0.15,
        time_stretch_prob: float = 0.3,
        time_stretch_range: tuple[float, float] = (0.9, 1.1),
        scale_prob: float = 0.3,
        scale_range: tuple[float, float] = (0.9, 1.1),
    ) -> None:
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.time_mask_prob = time_mask_prob
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_ratio = freq_mask_ratio
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to a batch of speech MFCC features.

        Args:
            x: ``(B, T, F)`` — e.g. ``(B, 800, 120)``

        Returns:
            Augmented tensor with same shape.
        """
        x = x.clone()
        B, T, F_ = x.shape

        # 1. Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. Time masking (SpecAugment-style)
        if torch.rand(1).item() < self.time_mask_prob:
            mask_len = max(1, int(T * self.time_mask_ratio))
            for b in range(B):
                t_start = torch.randint(0, max(1, T - mask_len), (1,)).item()
                x[b, t_start : t_start + mask_len, :] = 0.0

        # 3. Frequency masking
        if torch.rand(1).item() < self.freq_mask_prob:
            mask_bands = max(1, int(F_ * self.freq_mask_ratio))
            for b in range(B):
                f_start = torch.randint(0, max(1, F_ - mask_bands), (1,)).item()
                x[b, :, f_start : f_start + mask_bands] = 0.0

        # 4. Time stretching (via interpolation)
        if torch.rand(1).item() < self.time_stretch_prob:
            lo, hi = self.time_stretch_range
            factor = torch.empty(1).uniform_(lo, hi).item()
            new_T = max(1, int(T * factor))
            # Interpolate: (B, T, F) → (B, F, T) → interpolate → (B, F, new_T) → (B, new_T, F)
            x_t = x.permute(0, 2, 1)  # (B, F, T)
            x_t = F.interpolate(x_t, size=new_T, mode="linear", align_corners=False)
            # Pad or truncate back to original T
            if new_T < T:
                pad = torch.zeros(B, F_, T - new_T, device=x.device)
                x_t = torch.cat([x_t, pad], dim=2)
            else:
                x_t = x_t[:, :, :T]
            x = x_t.permute(0, 2, 1)  # (B, T, F)

        # 5. Global scaling
        if torch.rand(1).item() < self.scale_prob:
            lo, hi = self.scale_range
            scale = torch.empty(1, device=x.device).uniform_(lo, hi)
            x = x * scale

        return x


# ======================================================================
# NT-Xent Loss (reuse same formulation as EEG)
# ======================================================================

class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim /= self.temperature
        mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        pos_i = torch.arange(B, device=sim.device)
        labels = torch.cat([pos_i + B, pos_i])
        return F.cross_entropy(sim, labels)


# ======================================================================
# Projection Head
# ======================================================================

class ProjectionHead(nn.Module):
    """MLP projection head (discarded after pretraining)."""

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
# Contrastive Speech Dataset
# ======================================================================

class ContrastiveSpeechDataset(Dataset):
    """Yields two augmented views of each speech sample."""

    def __init__(self, features: np.ndarray, augmentor: SpeechAugmentor) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].unsqueeze(0)  # (1, T, F)
        v1 = self.augmentor(x).squeeze(0)
        v2 = self.augmentor(x).squeeze(0)
        return v1, v2


# ======================================================================
# Contrastive Speech Trainer
# ======================================================================

class ContrastiveSpeechTrainer:
    """Contrastive pretraining for SpeechEncoder.

    Usage::

        trainer = ContrastiveSpeechTrainer(cfg)
        trainer.train(features, save_dir)
        # Load speech_encoder_contrastive.pt for downstream fine-tuning.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"

        scfg = cfg.model.speech_encoder
        self.encoder = SpeechEncoder(
            n_features=scfg.n_mfcc,
            embedding_dim=scfg.embedding_dim,
        ).to(self.device)

        self.projector = ProjectionHead(
            input_dim=scfg.embedding_dim,
            hidden_dim=scfg.embedding_dim * 2,
            output_dim=64,
        ).to(self.device)

        # Hyperparameters
        v3 = getattr(cfg, "v3", {})
        cl = v3.get("contrastive", {}) if isinstance(v3, dict) else getattr(v3, "contrastive", {})

        self.epochs = cl.get("speech_epochs", 60) if isinstance(cl, dict) else getattr(cl, "speech_epochs", 60)
        self.batch_size = cl.get("speech_batch_size", 64) if isinstance(cl, dict) else getattr(cl, "speech_batch_size", 64)
        self.lr = cl.get("lr", 3e-4) if isinstance(cl, dict) else getattr(cl, "lr", 3e-4)
        self.temperature = cl.get("temperature", 0.07) if isinstance(cl, dict) else getattr(cl, "temperature", 0.07)
        self.weight_decay = cl.get("weight_decay", 1e-4) if isinstance(cl, dict) else getattr(cl, "weight_decay", 1e-4)

    def train(
        self,
        features: np.ndarray,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run contrastive pretraining on speech MFCC features.

        Args:
            features: ``(N, T, F)`` — e.g. ``(N, 800, 120)`` MFCC+Δ+ΔΔ features.
            save_dir: Directory to save encoder checkpoint.

        Returns:
            History dict with ``loss`` values per epoch.
        """
        augmentor = SpeechAugmentor()
        dataset = ContrastiveSpeechDataset(features, augmentor)
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
        scaler = GradScaler(enabled=self.use_amp)
        criterion = NTXentLoss(temperature=self.temperature)

        history: dict[str, list[float]] = {"loss": []}
        log_every = self.cfg.training.get("log_every", 5) if hasattr(self.cfg.training, "get") else 5
        t0 = time.time()

        logger.info(
            "Contrastive speech pretraining: %d samples, %d epochs, batch=%d, τ=%.3f",
            len(dataset), self.epochs, self.batch_size, self.temperature,
        )

        for epoch in range(1, self.epochs + 1):
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

                with autocast(enabled=self.use_amp):
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
                    "CL-Speech %d/%d  loss=%.4f  lr=%.6f",
                    epoch, self.epochs, epoch_loss, lr_now,
                )
                log_gpu_memory()

        elapsed = time.time() - t0
        logger.info("Contrastive speech pretraining complete in %.0fs", elapsed)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                self.encoder.state_dict(),
                save_dir / "speech_encoder_contrastive.pt",
            )
            logger.info("Saved contrastive speech encoder → %s", save_dir / "speech_encoder_contrastive.pt")

        return history
