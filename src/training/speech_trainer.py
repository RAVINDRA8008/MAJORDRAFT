"""Speech encoder pre-trainer — autonomous, overfitting-aware.

Trains the CNN-LSTM speech encoder on IEMOCAP features with:
- Class-weighted cross-entropy loss (auto-computed from label distribution)
- Balanced batch sampling (WeightedRandomSampler)
- Early stopping with configurable patience
- Automatic regularisation escalation on overfitting detection
  (dropout up, weight decay up, SpecAugment-style masking)
- Best-checkpoint selection by validation accuracy
- Comprehensive training report generation
"""

from __future__ import annotations

import copy
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import DictConfig

from src.models.speech_encoder import SpeechEncoder
from src.models.classifier import Classifier
from src.data.dataset import SpeechDataset
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)

# Human-readable label names
_LABEL_NAMES = {0: "angry", 1: "happy", 2: "sad", 3: "neutral"}


# ======================================================================
# Training report
# ======================================================================
@dataclass
class SpeechTrainingReport:
    """Collects metrics and generates a readable text report."""

    dataset_summary: str = ""
    class_distribution: str = ""
    total_epochs_run: int = 0
    best_epoch: int = 0
    best_val_acc: float = 0.0
    best_train_acc: float = 0.0
    final_train_acc: float = 0.0
    final_val_acc: float = 0.0
    overfitting_detected: bool = False
    overfitting_epoch: int | None = None
    regularisation_actions: list[str] = field(default_factory=list)
    early_stopped: bool = False
    wall_time_s: float = 0.0
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "=" * 60,
            "  SPEECH ENCODER — TRAINING REPORT",
            "=" * 60,
            "",
            "--- Dataset ---",
            self.dataset_summary,
            self.class_distribution,
            "",
            "--- Training ---",
            f"  Total epochs run     : {self.total_epochs_run}",
            f"  Wall-clock time      : {self.wall_time_s:.1f} s",
            f"  Early stopped        : {'YES' if self.early_stopped else 'No'}",
            "",
            "--- Best Checkpoint ---",
            f"  Best epoch           : {self.best_epoch}",
            f"  Best val accuracy    : {self.best_val_acc:.4f}",
            f"  Train acc at best    : {self.best_train_acc:.4f}",
            "",
            "--- Final Epoch ---",
            f"  Final train accuracy : {self.final_train_acc:.4f}",
            f"  Final val accuracy   : {self.final_val_acc:.4f}",
            "",
            "--- Overfitting ---",
            f"  Detected             : {'YES' if self.overfitting_detected else 'No'}",
        ]
        if self.overfitting_detected:
            lines.append(f"  First detected epoch : {self.overfitting_epoch}")
        if self.regularisation_actions:
            lines.append("  Actions taken:")
            for a in self.regularisation_actions:
                lines.append(f"    - {a}")
        lines += [
            "",
            "--- Recommendations ---",
        ]
        gap = self.final_train_acc - self.final_val_acc
        if gap > 0.15:
            lines.append("  - Significant train/val gap — consider more data or stronger regularisation.")
        if self.best_val_acc < 0.45:
            lines.append("  - Val accuracy is low — consider a larger model or more features.")
        if self.early_stopped:
            lines.append("  - Training was early-stopped — model was overfitting.")
        if not self.overfitting_detected and not self.early_stopped:
            lines.append("  - Training completed normally without overfitting.")
        lines += ["", "=" * 60]
        return "\n".join(lines)


# ======================================================================
# SpecAugment-style time/frequency masking (on-the-fly)
# ======================================================================
class SpecAugment(nn.Module):
    """Lightweight spectrogram augmentation: random time and freq masks."""

    def __init__(self, freq_mask_width: int = 10, time_mask_width: int = 10) -> None:
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random masks.  x: (B, T, F)."""
        if not self.training:
            return x
        B, T, F = x.shape
        aug = x.clone()
        # Frequency mask
        fw = min(self.freq_mask_width, F)
        f0 = torch.randint(0, F - fw + 1, (1,)).item()
        aug[:, :, f0 : f0 + fw] = 0.0
        # Time mask
        tw = min(self.time_mask_width, T)
        t0 = torch.randint(0, T - tw + 1, (1,)).item()
        aug[:, t0 : t0 + tw, :] = 0.0
        return aug


# ======================================================================
# Speech Trainer — with autonomous overfitting management
# ======================================================================
class SpeechTrainer:
    """Pre-train speech encoder on IEMOCAP features.

    Anti-overfitting features:
    - Class-weighted CE loss
    - Balanced batch sampling
    - Early stopping (patience-based)
    - Dynamic regularisation escalation
    - Best-val-acc checkpoint selection
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()

        scfg = cfg.model.speech_encoder

        # ---- Model ----
        self.encoder = SpeechEncoder(
            n_features=scfg.n_mfcc,
            cnn_channels=list(scfg.cnn_channels),
            lstm_hidden_size=scfg.lstm_hidden,
            lstm_num_layers=scfg.lstm_layers,
            lstm_dropout=scfg.dropout,
            embedding_dim=scfg.embedding_dim,
        ).to(self.device)

        self.head = Classifier(
            embedding_dim=scfg.embedding_dim,
            num_classes=cfg.model.num_classes,
        ).to(self.device)

        # ---- Hyper-params ----
        self.epochs = scfg.get("pretrain_epochs", 30)
        self.batch_size = scfg.get("batch_size", 64)
        self.lr = scfg.get("lr", 1e-3)
        self.base_dropout = scfg.get("dropout", 0.3)
        self.base_weight_decay = scfg.get("weight_decay", 0.0)

        # Early stopping
        self.patience = scfg.get("patience", 7)
        self.min_delta = scfg.get("min_delta", 0.001)

        # Overfitting detection
        self._overfit_window = 3  # consecutive val-drops to trigger
        self._reg_escalated = False
        self._spec_augment = SpecAugment(
            freq_mask_width=scfg.get("freq_mask", 10),
            time_mask_width=scfg.get("time_mask", 10),
        ).to(self.device)
        self._use_specaugment = False  # enabled dynamically

        # Will be created per-fit (need labels for class weights)
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion: nn.Module | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
        """Inverse-frequency class weights."""
        counts = Counter(labels.numpy().tolist())
        total = sum(counts.values())
        num_classes = max(counts.keys()) + 1
        weights = torch.zeros(num_classes)
        for cls, cnt in counts.items():
            weights[cls] = total / (num_classes * cnt)
        return weights

    @staticmethod
    def _make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler so each class is equally likely."""
        counts = Counter(labels.numpy().tolist())
        sample_weights = torch.tensor([1.0 / counts[int(l)] for l in labels])
        return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)

    def _detect_overfitting(self, history: dict) -> bool:
        """Check if train acc keeps rising while val acc drops for N epochs."""
        accs = history["val_acc"]
        train = history["train_acc"]
        if len(accs) < self._overfit_window + 1:
            return False
        recent_val = accs[-self._overfit_window:]
        recent_train = train[-self._overfit_window:]
        # Val is dropping and train is rising
        val_dropping = all(recent_val[i] <= recent_val[i - 1] for i in range(1, len(recent_val)))
        train_rising = all(recent_train[i] >= recent_train[i - 1] for i in range(1, len(recent_train)))
        return val_dropping and train_rising

    def _escalate_regularisation(self, report: SpeechTrainingReport, epoch: int) -> None:
        """Increase regularisation when overfitting is detected."""
        if self._reg_escalated:
            return
        self._reg_escalated = True
        report.overfitting_detected = True
        report.overfitting_epoch = epoch

        # 1. Increase dropout in LSTM
        new_dropout = min(self.base_dropout + 0.15, 0.6)
        self.encoder.lstm.dropout = new_dropout
        report.regularisation_actions.append(
            f"Epoch {epoch}: LSTM dropout {self.base_dropout:.2f} -> {new_dropout:.2f}"
        )

        # 2. Add weight decay via a new optimizer
        wd = 1e-4
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr * 0.5, weight_decay=wd)
        report.regularisation_actions.append(
            f"Epoch {epoch}: Added weight decay {wd}, halved LR -> {self.lr * 0.5:.6f}"
        )

        # 3. Enable SpecAugment
        self._use_specaugment = True
        report.regularisation_actions.append(
            f"Epoch {epoch}: Enabled SpecAugment (freq_mask, time_mask)"
        )

        logger.warning(
            "Overfitting detected at epoch %d — escalated regularisation "
            "(dropout=%.2f, weight_decay=%s, SpecAugment ON).",
            epoch, new_dropout, wd,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        val_features: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Pre-train speech encoder with autonomous overfitting management.

        Returns:
            ``{"train_loss": [...], "train_acc": [...], "val_acc": [...],
               "val_loss": [...], "_report": SpeechTrainingReport}``
        """
        report = SpeechTrainingReport()
        t0 = time.time()

        # ---- Class weights and balanced sampler ----
        class_weights = self._compute_class_weights(labels).to(self.device)
        dist = Counter(labels.numpy().tolist())
        max_cnt = max(dist.values())
        min_cnt = min(dist.values())
        imbalance_ratio = max_cnt / max(min_cnt, 1)

        dist_str = ", ".join(
            f"{_LABEL_NAMES.get(k, k)}: {v}" for k, v in sorted(dist.items())
        )
        report.dataset_summary = (
            f"  Train: {len(labels)} samples | Val: {len(val_labels) if val_labels is not None else 0} samples"
        )
        report.class_distribution = f"  Classes: {dist_str} | Imbalance ratio: {imbalance_ratio:.1f}x"

        logger.info("Label distribution: %s  (imbalance %.1fx)", dist_str, imbalance_ratio)

        use_balanced = imbalance_ratio > 2.0
        if use_balanced:
            logger.info("Imbalance > 2x — using class-weighted loss + balanced sampling.")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights if use_balanced else None
        )

        params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.base_weight_decay
        )

        train_ds = SpeechDataset(features, labels)
        sampler = self._make_balanced_sampler(labels) if use_balanced else None
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
        )

        val_loader = None
        if val_features is not None and val_labels is not None:
            val_ds = SpeechDataset(val_features, val_labels)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": [],
        }

        # Best-checkpoint tracking
        best_val_acc = -1.0
        best_epoch = 0
        best_state: dict | None = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            self.encoder.train()
            self.head.train()
            running_loss, correct, total = 0.0, 0, 0

            for feat, lbl in train_loader:
                feat, lbl = feat.to(self.device), lbl.to(self.device)

                # SpecAugment (if enabled)
                if self._use_specaugment:
                    feat = self._spec_augment(feat)

                emb = self.encoder(feat)
                logits = self.head(emb)
                loss = self.criterion(logits, lbl)

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.head.parameters()),
                    max_norm=5.0,
                )
                self.optimizer.step()

                running_loss += loss.item() * feat.size(0)
                correct += (logits.argmax(1) == lbl).sum().item()
                total += feat.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # ---- Validate ----
            val_acc, val_loss = 0.0, 0.0
            if val_loader is not None:
                val_acc, val_loss = self._evaluate(val_loader)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            # ---- Best checkpoint ----
            if val_acc > best_val_acc + self.min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {
                    "epoch": epoch,
                    "encoder": copy.deepcopy(self.encoder.state_dict()),
                    "head": copy.deepcopy(self.head.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # ---- Logging ----
            log_every = self.cfg.training.get("log_every", 5)
            if epoch % log_every == 0 or epoch == 1 or epoch == self.epochs:
                logger.info(
                    "Speech %d/%d — loss=%.4f  train_acc=%.3f  val_acc=%.3f  "
                    "best=%.3f@%d  patience=%d/%d",
                    epoch, self.epochs, train_loss, train_acc, val_acc,
                    best_val_acc, best_epoch, patience_counter, self.patience,
                )
                log_gpu_memory()

            # ---- Overfitting detection ----
            if self._detect_overfitting(history):
                self._escalate_regularisation(report, epoch)

            # ---- Early stopping ----
            if patience_counter >= self.patience:
                logger.warning(
                    "Early stopping at epoch %d (val_acc did not improve for %d epochs).",
                    epoch, self.patience,
                )
                report.early_stopped = True
                break

            # ---- Periodic checkpoint ----
            save_every = self.cfg.training.get("save_every", 10)
            if save_dir and epoch % save_every == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder": self.encoder.state_dict(),
                        "head": self.head.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    Path(save_dir) / f"speech_epoch_{epoch}.pt",
                )

        # ---- Restore best checkpoint ----
        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.head.load_state_dict(best_state["head"])
            logger.info(
                "Restored best checkpoint from epoch %d (val_acc=%.4f).",
                best_epoch, best_val_acc,
            )

        # Save best model explicitly
        if save_dir:
            best_path = Path(save_dir) / "best_speech_model.pt"
            save_checkpoint(
                best_state or {"encoder": self.encoder.state_dict(), "head": self.head.state_dict()},
                best_path,
            )
            logger.info("Best speech model saved: %s", best_path)

        # ---- Populate report ----
        report.total_epochs_run = len(history["train_acc"])
        report.best_epoch = best_epoch
        report.best_val_acc = best_val_acc
        report.best_train_acc = history["train_acc"][best_epoch - 1] if best_epoch > 0 else 0.0
        report.final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
        report.final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        report.train_accs = history["train_acc"]
        report.val_accs = history["val_acc"]
        report.train_losses = history["train_loss"]
        report.wall_time_s = time.time() - t0

        # Attach report to history for callers
        history["_report"] = report  # type: ignore[assignment]

        logger.info("Speech pre-training complete — %d epochs.", report.total_epochs_run)
        logger.info("\n%s", report.render())

        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return (accuracy, loss) on the given loader."""
        self.encoder.eval()
        self.head.eval()
        correct, total, running_loss = 0, 0, 0.0
        for feat, lbl in loader:
            feat, lbl = feat.to(self.device), lbl.to(self.device)
            logits = self.head(self.encoder(feat))
            loss = self.criterion(logits, lbl)
            running_loss += loss.item() * feat.size(0)
            correct += (logits.argmax(1) == lbl).sum().item()
            total += feat.size(0)
        return correct / max(total, 1), running_loss / max(total, 1)
