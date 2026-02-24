"""Domain Adversarial Neural Network (DANN) for cross-dataset alignment (v3).

Research rationale
──────────────────
DEAP (EEG) and IEMOCAP (speech) have very different feature distributions.
When encoders are trained independently, the embedding spaces may not be
aligned, making fusion sub-optimal.

DANN (Ganin et al., JMLR 2016) adds a domain classifier with a Gradient
Reversal Layer (GRL).  The encoder is trained to:
1. Minimize emotion classification loss (task-specific)
2. Maximize domain confusion (domain-adversarial)

This forces the encoder to learn domain-invariant features, improving
cross-modal fusion.

Architecture
────────────
- GradientReversalLayer: Reverses gradients during backprop (scales by -λ)
- DomainClassifier: MLP that predicts which modality an embedding came from
- DomainAdaptationTrainer: Jointly trains both encoders with emotion CE +
  domain confusion loss.  Lambda follows a schedule (0→1 over training).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


# ======================================================================
# Gradient Reversal Layer
# ======================================================================

class GradientReversalFunction(Function):
    """Gradient reversal for domain-adversarial training.

    Forward pass: identity.
    Backward pass: negate gradients scaled by λ.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as a nn.Module."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# ======================================================================
# Domain Classifier
# ======================================================================

class DomainClassifier(nn.Module):
    """MLP that predicts domain (EEG=0, Speech=1) from embeddings.

    Architecture: Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Linear(2)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRL then classify domain.

        Args:
            x: ``(B, embed_dim)`` embeddings.

        Returns:
            Domain logits ``(B, 2)``
        """
        x = self.grl(x)
        return self.classifier(x)


# ======================================================================
# Emotion Classifier Head (shared across domains)
# ======================================================================

class EmotionHead(nn.Module):
    """Lightweight emotion classifier for DANN training."""

    def __init__(self, input_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ======================================================================
# Domain Adaptation Dataset
# ======================================================================

class DomainAlignedDataset(Dataset):
    """Interleaves EEG and speech samples for domain alignment training.

    Each sample returns:
    - features: raw input for the appropriate encoder
    - emotion_label: shared emotion label (0-3)
    - domain_label: 0=EEG, 1=Speech
    - modality_flag: 0 or 1 (to route to correct encoder)
    """

    def __init__(
        self,
        eeg_features: np.ndarray,
        eeg_labels: np.ndarray,
        speech_features: np.ndarray,
        speech_labels: np.ndarray,
    ) -> None:
        self.eeg_features = torch.as_tensor(eeg_features, dtype=torch.float32)
        self.eeg_labels = torch.as_tensor(eeg_labels, dtype=torch.long)
        self.speech_features = torch.as_tensor(speech_features, dtype=torch.float32)
        self.speech_labels = torch.as_tensor(speech_labels, dtype=torch.long)

        self.n_eeg = len(self.eeg_features)
        self.n_speech = len(self.speech_features)
        self.total = self.n_eeg + self.n_speech

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        if idx < self.n_eeg:
            return (
                self.eeg_features[idx],
                self.eeg_labels[idx],
                torch.tensor(0, dtype=torch.long),  # domain = EEG
                torch.tensor(0, dtype=torch.long),  # modality flag
            )
        else:
            sidx = idx - self.n_eeg
            return (
                self.speech_features[sidx],
                self.speech_labels[sidx],
                torch.tensor(1, dtype=torch.long),  # domain = Speech
                torch.tensor(1, dtype=torch.long),  # modality flag
            )


# ======================================================================
# Lambda Schedule
# ======================================================================

def _lambda_schedule(epoch: int, total_epochs: int) -> float:
    """Progressive lambda schedule: 0 → 1 over training.

    Uses the schedule from Ganin et al. (2016):
        λ(p) = 2 / (1 + exp(-10p)) - 1,  where p = epoch / total_epochs
    """
    p = epoch / max(total_epochs, 1)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ======================================================================
# Domain Adaptation Trainer
# ======================================================================

class DomainAdaptationTrainer:
    """DANN-based domain adaptation for cross-dataset alignment.

    Trains both encoders jointly with:
    1. Emotion classification loss (CE with label smoothing)
    2. Domain confusion loss (CE through GRL)

    The domain confusion forces encoders to produce modality-invariant
    embeddings, improving downstream fusion.

    Usage::

        trainer = DomainAdaptationTrainer(cfg)
        trainer.train(eeg_feats, eeg_labels, speech_feats, speech_labels, save_dir)
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"

        # Build encoders
        ecfg = cfg.model.eeg_encoder
        scfg = cfg.model.speech_encoder
        self.eeg_encoder = EEGEncoder(
            input_dim=ecfg.input_dim,
            hidden_dims=list(ecfg.hidden_dims),
            embedding_dim=ecfg.embedding_dim,
            dropout=ecfg.dropout,
        ).to(self.device)

        self.speech_encoder = SpeechEncoder(
            n_features=scfg.n_mfcc,
            embedding_dim=scfg.embedding_dim,
        ).to(self.device)

        embed_dim = ecfg.embedding_dim  # 128

        # Domain classifier + emotion head
        self.domain_classifier = DomainClassifier(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            dropout=0.3,
        ).to(self.device)

        self.emotion_head = EmotionHead(
            input_dim=embed_dim,
            num_classes=cfg.model.num_classes,
        ).to(self.device)

        # Hyperparameters from v3 config
        v3 = getattr(cfg, "v3", {})
        da = v3.get("domain_adaptation", {}) if isinstance(v3, dict) else getattr(v3, "domain_adaptation", {})

        self.epochs = da.get("epochs", 30) if isinstance(da, dict) else getattr(da, "epochs", 30)
        self.batch_size = da.get("batch_size", 128) if isinstance(da, dict) else getattr(da, "batch_size", 128)
        self.lr = da.get("lr", 1e-4) if isinstance(da, dict) else getattr(da, "lr", 1e-4)
        self.domain_weight = da.get("domain_weight", 0.3) if isinstance(da, dict) else getattr(da, "domain_weight", 0.3)
        self.label_smoothing = da.get("label_smoothing", 0.1) if isinstance(da, dict) else getattr(da, "label_smoothing", 0.1)

    def load_pretrained(self, save_dir: str | Path) -> None:
        """Load contrastive-pretrained encoder weights."""
        save_dir = Path(save_dir)

        eeg_ckpt = save_dir / "eeg_encoder_contrastive.pt"
        if eeg_ckpt.exists():
            self.eeg_encoder.load_state_dict(torch.load(eeg_ckpt, map_location=self.device))
            logger.info("Loaded contrastive EEG encoder from %s", eeg_ckpt)

        speech_ckpt = save_dir / "speech_encoder_contrastive.pt"
        if speech_ckpt.exists():
            self.speech_encoder.load_state_dict(torch.load(speech_ckpt, map_location=self.device))
            logger.info("Loaded contrastive speech encoder from %s", speech_ckpt)

    def train(
        self,
        eeg_features: np.ndarray,
        eeg_labels: np.ndarray,
        speech_features: np.ndarray,
        speech_labels: np.ndarray,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run domain adversarial training.

        Args:
            eeg_features: ``(N_eeg, 160)`` DE features.
            eeg_labels: ``(N_eeg,)`` emotion labels.
            speech_features: ``(N_speech, T, 120)`` MFCC features.
            speech_labels: ``(N_speech,)`` emotion labels.
            save_dir: Where to save encoder checkpoints.

        Returns:
            History dict with ``emotion_loss``, ``domain_loss``, ``total_loss``.
        """
        dataset = DomainAlignedDataset(
            eeg_features, eeg_labels, speech_features, speech_labels
        )

        # Custom collate to handle variable-shape features
        def collate_fn(batch):
            eeg_feats, eeg_emos, eeg_doms, speech_feats, speech_emos, speech_doms = \
                [], [], [], [], [], []
            for feat, emo, dom, mod in batch:
                if mod.item() == 0:  # EEG
                    eeg_feats.append(feat)
                    eeg_emos.append(emo)
                    eeg_doms.append(dom)
                else:  # Speech
                    speech_feats.append(feat)
                    speech_emos.append(emo)
                    speech_doms.append(dom)
            return {
                "eeg_feats": torch.stack(eeg_feats) if eeg_feats else None,
                "eeg_emos": torch.stack(eeg_emos) if eeg_emos else None,
                "eeg_doms": torch.stack(eeg_doms) if eeg_doms else None,
                "speech_feats": torch.stack(speech_feats) if speech_feats else None,
                "speech_emos": torch.stack(speech_emos) if speech_emos else None,
                "speech_doms": torch.stack(speech_doms) if speech_doms else None,
            }

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Optimizer covers all trainable components
        params = (
            list(self.eeg_encoder.parameters())
            + list(self.speech_encoder.parameters())
            + list(self.emotion_head.parameters())
            + list(self.domain_classifier.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6,
        )
        scaler = GradScaler('cuda', enabled=self.use_amp)

        emotion_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        domain_criterion = nn.CrossEntropyLoss()

        history = {"emotion_loss": [], "domain_loss": [], "total_loss": []}
        log_every = self.cfg.training.get("log_every", 5) if hasattr(self.cfg.training, "get") else 5
        t0 = time.time()

        logger.info(
            "DANN training: EEG=%d + Speech=%d samples, %d epochs, batch=%d, λ_domain=%.2f",
            len(eeg_features), len(speech_features),
            self.epochs, self.batch_size, self.domain_weight,
        )

        for epoch in range(1, self.epochs + 1):
            lambda_ = _lambda_schedule(epoch, self.epochs)
            self.domain_classifier.grl.set_lambda(lambda_)

            self.eeg_encoder.train()
            self.speech_encoder.train()
            self.emotion_head.train()
            self.domain_classifier.train()

            running_emo_loss = 0.0
            running_dom_loss = 0.0
            n_batches = 0

            for batch in loader:
                all_embeddings = []
                all_emo_labels = []
                all_dom_labels = []

                with autocast('cuda', enabled=self.use_amp):
                    # Process EEG samples
                    if batch["eeg_feats"] is not None:
                        eeg_x = batch["eeg_feats"].to(self.device, non_blocking=True)
                        eeg_emb = self.eeg_encoder(eeg_x)
                        all_embeddings.append(eeg_emb)
                        all_emo_labels.append(batch["eeg_emos"].to(self.device, non_blocking=True))
                        all_dom_labels.append(batch["eeg_doms"].to(self.device, non_blocking=True))

                    # Process speech samples
                    if batch["speech_feats"] is not None:
                        sp_x = batch["speech_feats"].to(self.device, non_blocking=True)
                        sp_emb = self.speech_encoder(sp_x)
                        all_embeddings.append(sp_emb)
                        all_emo_labels.append(batch["speech_emos"].to(self.device, non_blocking=True))
                        all_dom_labels.append(batch["speech_doms"].to(self.device, non_blocking=True))

                    if not all_embeddings:
                        continue

                    embeddings = torch.cat(all_embeddings, dim=0)
                    emo_labels = torch.cat(all_emo_labels, dim=0)
                    dom_labels = torch.cat(all_dom_labels, dim=0)

                    # Emotion classification loss
                    emo_logits = self.emotion_head(embeddings)
                    emo_loss = emotion_criterion(emo_logits, emo_labels)

                    # Domain classification loss (through GRL)
                    dom_logits = self.domain_classifier(embeddings)
                    dom_loss = domain_criterion(dom_logits, dom_labels)

                    # Combined loss
                    total_loss = emo_loss + self.domain_weight * dom_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_emo_loss += emo_loss.item()
                running_dom_loss += dom_loss.item()
                n_batches += 1

            scheduler.step()

            epoch_emo = running_emo_loss / max(n_batches, 1)
            epoch_dom = running_dom_loss / max(n_batches, 1)
            history["emotion_loss"].append(epoch_emo)
            history["domain_loss"].append(epoch_dom)
            history["total_loss"].append(epoch_emo + self.domain_weight * epoch_dom)

            if epoch % log_every == 0 or epoch == 1 or epoch == self.epochs:
                dom_acc = self._domain_accuracy(loader)
                logger.info(
                    "DANN %d/%d  emo=%.4f  dom=%.4f  λ=%.3f  dom_acc=%.1f%%",
                    epoch, self.epochs, epoch_emo, epoch_dom, lambda_, dom_acc * 100,
                )
                log_gpu_memory()

        elapsed = time.time() - t0
        logger.info("DANN training complete in %.0fs", elapsed)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                self.eeg_encoder.state_dict(),
                save_dir / "eeg_encoder_dann.pt",
            )
            torch.save(
                self.speech_encoder.state_dict(),
                save_dir / "speech_encoder_dann.pt",
            )
            logger.info(
                "Saved DANN encoders → %s/{eeg,speech}_encoder_dann.pt", save_dir
            )

        return history

    @torch.no_grad()
    def _domain_accuracy(self, loader: DataLoader) -> float:
        """Compute domain classification accuracy (lower = better alignment)."""
        self.domain_classifier.eval()
        self.eeg_encoder.eval()
        self.speech_encoder.eval()

        correct = 0
        total = 0
        for batch in loader:
            embeddings = []
            dom_labels = []

            if batch["eeg_feats"] is not None:
                eeg_x = batch["eeg_feats"].to(self.device)
                eeg_emb = self.eeg_encoder(eeg_x)
                embeddings.append(eeg_emb)
                dom_labels.append(batch["eeg_doms"].to(self.device))

            if batch["speech_feats"] is not None:
                sp_x = batch["speech_feats"].to(self.device)
                sp_emb = self.speech_encoder(sp_x)
                embeddings.append(sp_emb)
                dom_labels.append(batch["speech_doms"].to(self.device))

            if not embeddings:
                continue

            emb = torch.cat(embeddings, dim=0)
            dom = torch.cat(dom_labels, dim=0)

            # Temporarily disable GRL for accuracy computation
            old_lambda = self.domain_classifier.grl.lambda_
            self.domain_classifier.grl.set_lambda(0.0)
            preds = self.domain_classifier(emb).argmax(dim=1)
            self.domain_classifier.grl.set_lambda(old_lambda)

            correct += (preds == dom).sum().item()
            total += len(dom)

        self.domain_classifier.train()
        self.eeg_encoder.train()
        self.speech_encoder.train()
        return correct / max(total, 1)
