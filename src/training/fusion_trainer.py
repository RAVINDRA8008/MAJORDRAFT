"""Fusion trainer with label-aligned multimodal pairing.

Key improvements:
1. **Label-aligned pairing** — EEG and speech samples are matched BY EMOTION
   LABEL, not by index.  Previously sample i from DEAP was paired with
   sample i from IEMOCAP, which are unrelated subjects/tasks.  Now every
   (eeg, speech) pair shares the same emotion class.
2. **Focal loss** (gamma=2) — handles extreme DEAP class imbalance far
   better than plain weighted CE.
3. **Re-shuffled pairs each epoch** — prevents the model from memorising
   specific (eeg_j, speech_k) pairings.
4. **Mixed-precision training** (AMP) + cosine LR + early stopping.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from src.models.fusion import FusionClassifier
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


# ======================================================================
# Focal Loss
# ======================================================================
class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification.

    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ======================================================================
# Label-aligned dataset
# ======================================================================
class LabelAlignedDataset(Dataset):
    """Pairs EEG and speech embeddings by emotion label.

    For each class c, randomly associates EEG samples of class c with
    speech samples of class c.  This creates semantically meaningful
    cross-modal pairs (both modalities express the same emotion).

    Call ``reshuffle()`` between epochs for diverse pairings.
    """

    def __init__(
        self,
        eeg_emb: torch.Tensor,
        eeg_labels: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_labels: torch.Tensor,
        num_classes: int = 4,
        balance_classes: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.balance_classes = balance_classes

        self.eeg_by_class: dict[int, torch.Tensor] = {}
        self.speech_by_class: dict[int, torch.Tensor] = {}
        for c in range(num_classes):
            eeg_mask = eeg_labels == c
            sp_mask = speech_labels == c
            self.eeg_by_class[c] = eeg_emb[eeg_mask]
            self.speech_by_class[c] = speech_emb[sp_mask]

        # Determine samples per class — v4: use MAX of both modalities
        # with oversampling (replacement) to create more diverse pairs
        if balance_classes:
            # Target: largest modality count across all classes, capped at 5000
            sizes = []
            for c in range(num_classes):
                ne = len(self.eeg_by_class[c])
                ns = len(self.speech_by_class[c])
                if ne > 0 and ns > 0:
                    sizes.append(max(ne, ns))
            target = min(max(sizes) if sizes else 100, 5000)
            self.samples_per_class = {c: target for c in range(num_classes)}
        else:
            self.samples_per_class = {}
            for c in range(num_classes):
                ne = len(self.eeg_by_class[c])
                ns = len(self.speech_by_class[c])
                self.samples_per_class[c] = max(ne, ns)

        self.eeg_data = torch.empty(0)
        self.sp_data = torch.empty(0)
        self.labels = torch.empty(0, dtype=torch.long)
        self.total = 0
        self._build_pairs()

    def _build_pairs(self) -> None:
        eeg_list, sp_list, lbl_list = [], [], []
        for c in range(self.num_classes):
            n = self.samples_per_class[c]
            ne = len(self.eeg_by_class[c])
            ns = len(self.speech_by_class[c])
            if ne == 0 or ns == 0:
                continue

            # Random indices (with replacement for oversampling)
            eeg_idx = torch.randint(0, ne, (n,))
            sp_idx = torch.randint(0, ns, (n,))

            eeg_list.append(self.eeg_by_class[c][eeg_idx])
            sp_list.append(self.speech_by_class[c][sp_idx])
            lbl_list.append(torch.full((n,), c, dtype=torch.long))

        self.eeg_data = torch.cat(eeg_list)
        self.sp_data = torch.cat(sp_list)
        self.labels = torch.cat(lbl_list)

        # Shuffle
        perm = torch.randperm(len(self.labels))
        self.eeg_data = self.eeg_data[perm]
        self.sp_data = self.sp_data[perm]
        self.labels = self.labels[perm]
        self.total = len(self.labels)

    def reshuffle(self) -> None:
        """Re-pair samples for a new epoch (call between epochs)."""
        self._build_pairs()

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eeg_data[idx], self.sp_data[idx], self.labels[idx]


# ======================================================================
# Trainer
# ======================================================================
class FusionTrainer:
    """Train FusionClassifier on label-aligned (eeg, speech, label) triples."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"

        fcfg = cfg.model.fusion
        dropout_val = fcfg.dropout
        if isinstance(dropout_val, (int, float)):
            dropout_list = [float(dropout_val)] * len(list(fcfg.hidden_dims))
        else:
            dropout_list = list(dropout_val)

        self.fusion = FusionClassifier(
            eeg_embed_dim=fcfg.eeg_dim,
            speech_embed_dim=fcfg.speech_dim,
            hidden_dims=list(fcfg.hidden_dims),
            num_classes=cfg.model.num_classes,
            dropout=dropout_list,
            modality_dropout_prob=fcfg.modality_dropout,
        ).to(self.device)

        self.epochs = fcfg.get("epochs", 50)
        self.batch_size = fcfg.get("batch_size", 128)
        self.lr = fcfg.get("lr", 1e-3)
        self.patience = fcfg.get("patience", 10)
        self.focal_gamma = fcfg.get("focal_gamma", 2.0)
        self.num_classes = cfg.model.num_classes

    # ------------------------------------------------------------------

    def fit(
        self,
        eeg_emb: torch.Tensor,
        eeg_labels: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_labels: torch.Tensor,
        val_eeg_emb: torch.Tensor | None = None,
        val_eeg_labels: torch.Tensor | None = None,
        val_speech_emb: torch.Tensor | None = None,
        val_speech_labels: torch.Tensor | None = None,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Train fusion model on label-aligned multimodal pairs.

        Returns history with ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``.
        """
        # ── Label-aligned dataset ──
        train_ds = LabelAlignedDataset(
            eeg_emb, eeg_labels, speech_emb, speech_labels,
            num_classes=self.num_classes, balance_classes=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, drop_last=True,
            num_workers=2, pin_memory=True,
        )

        has_val = val_eeg_emb is not None and val_speech_emb is not None
        val_ds = None
        val_loader = None
        if has_val:
            val_ds = LabelAlignedDataset(
                val_eeg_emb, val_eeg_labels,
                val_speech_emb, val_speech_labels,
                num_classes=self.num_classes, balance_classes=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size,
                num_workers=2, pin_memory=True,
            )

        # ── Class weights for focal loss ──
        all_labels = torch.cat([eeg_labels, speech_labels])
        counts = Counter(all_labels.numpy().tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            weights[cls] = total / (n_cls * cnt)
        weights = weights.to(self.device)

        # Log dataset info
        logger.info(
            "Label-aligned dataset: %d train pairs (balanced per class), "
            "%d val pairs",
            len(train_ds), len(val_ds) if val_ds else 0,
        )
        for c in range(self.num_classes):
            ne = len(train_ds.eeg_by_class[c])
            ns = len(train_ds.speech_by_class[c])
            logger.info(
                "  Class %d: %d EEG, %d speech → %d paired samples",
                c, ne, ns, train_ds.samples_per_class[c],
            )

        # ── Optimiser & loss ──
        optimizer = torch.optim.AdamW(
            self.fusion.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        criterion = FocalLoss(
            gamma=self.focal_gamma,
            weight=weights,
            label_smoothing=0.05,
        )
        warmup_epochs = max(1, self.epochs // 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6,
        )
        scaler = GradScaler('cuda', enabled=self.use_amp)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        }
        best_val_acc = 0.0
        best_state = None
        best_epoch = 0
        patience_ctr = 0
        log_every = self.cfg.training.get("log_every", 5)

        for epoch in range(1, self.epochs + 1):
            # Re-shuffle label-aligned pairs each epoch
            train_ds.reshuffle()
            if val_ds is not None:
                val_ds.reshuffle()

            # Warmup LR
            if epoch <= warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = self.lr * epoch / warmup_epochs

            # ── Train ──
            self.fusion.train()
            running_loss, correct, total = 0.0, 0, 0

            for eeg_b, sp_b, lbl_b in train_loader:
                eeg_b = eeg_b.to(self.device, non_blocking=True)
                sp_b = sp_b.to(self.device, non_blocking=True)
                lbl_b = lbl_b.to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp):
                    logits = self.fusion(eeg_b, sp_b)
                    loss = criterion(logits, lbl_b)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * eeg_b.size(0)
                correct += (logits.argmax(1) == lbl_b).sum().item()
                total += eeg_b.size(0)

            if epoch > warmup_epochs:
                scheduler.step()

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # ── Validate ──
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # ── Best checkpoint ──
            if val_acc > best_val_acc + 0.001:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.fusion.state_dict().items()}
                patience_ctr = 0
                if save_dir:
                    save_checkpoint(
                        {"epoch": epoch, "fusion": self.fusion.state_dict()},
                        Path(save_dir) / "best_fusion_baseline.pt",
                    )
            else:
                patience_ctr += 1

            if epoch % log_every == 0 or epoch == 1 or epoch == self.epochs:
                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Fusion %d/%d  loss=%.4f  train_acc=%.3f  val_acc=%.3f  "
                    "best=%.3f@%d  lr=%.6f  pat=%d/%d",
                    epoch, self.epochs, train_loss, train_acc, val_acc,
                    best_val_acc, best_epoch, cur_lr, patience_ctr, self.patience,
                )
                log_gpu_memory()

            # Early stopping
            if patience_ctr >= self.patience:
                logger.info("Fusion early stop at epoch %d", epoch)
                break

        # Restore best
        if best_state is not None:
            self.fusion.load_state_dict(best_state)
            logger.info(
                "Restored best fusion from epoch %d (val_acc=%.4f)",
                best_epoch, best_val_acc,
            )

        logger.info(
            "Fusion training complete — %d epochs, best_val_acc=%.4f",
            min(epoch, self.epochs), best_val_acc,
        )
        return history

    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        self.fusion.eval()
        correct, total, total_loss = 0, 0, 0.0
        for eeg_b, sp_b, lbl_b in loader:
            eeg_b = eeg_b.to(self.device, non_blocking=True)
            sp_b = sp_b.to(self.device, non_blocking=True)
            lbl_b = lbl_b.to(self.device, non_blocking=True)
            with autocast('cuda', enabled=self.use_amp):
                logits = self.fusion(eeg_b, sp_b)
                total_loss += criterion(logits, lbl_b).item() * eeg_b.size(0)
            correct += (logits.argmax(1) == lbl_b).sum().item()
            total += eeg_b.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)
