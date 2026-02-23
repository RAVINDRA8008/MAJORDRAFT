"""Fusion trainer (without RL).

Trains the late-fusion classifier on pre-encoded EEG + speech
embeddings.  Used for:
  • The non-RL baseline
  • The initial warm-up before RL kicks in

Optimised with:
- Class-weighted CE loss + label smoothing
- Balanced batch sampling
- Mixed-precision training (AMP)
- Cosine LR scheduling with warmup
- Early stopping + best-checkpoint selection
- Gradient clipping
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from omegaconf import DictConfig

from src.models.fusion import FusionClassifier
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


def _class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = Counter(labels.numpy().tolist())
    total = sum(counts.values())
    n_classes = max(counts.keys()) + 1
    w = torch.zeros(n_classes)
    for cls, cnt in counts.items():
        w[cls] = total / (n_classes * cnt)
    return w


def _balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    counts = Counter(labels.numpy().tolist())
    sw = torch.tensor([1.0 / counts[int(l)] for l in labels])
    return WeightedRandomSampler(sw, num_samples=len(labels), replacement=True)


class FusionTrainer:
    """Train FusionClassifier on (eeg_emb, speech_emb, label) triples."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"

        fcfg = cfg.model.fusion
        self.fusion = FusionClassifier(
            eeg_embed_dim=fcfg.eeg_dim,
            speech_embed_dim=fcfg.speech_dim,
            hidden_dims=list(fcfg.hidden_dims),
            num_classes=cfg.model.num_classes,
            dropout=[fcfg.dropout, fcfg.dropout] if not isinstance(fcfg.dropout, list) else list(fcfg.dropout),
            modality_dropout_prob=fcfg.modality_dropout,
        ).to(self.device)

        self.epochs = fcfg.get("epochs", 50)
        self.batch_size = fcfg.get("batch_size", 64)
        self.lr = fcfg.get("lr", 1e-3)
        self.patience = fcfg.get("patience", 10)

    # ------------------------------------------------------------------

    def fit(
        self,
        eeg_emb: torch.Tensor,
        speech_emb: torch.Tensor,
        labels: torch.Tensor,
        val_eeg_emb: torch.Tensor | None = None,
        val_speech_emb: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Train fusion model.

        Returns history with ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``.
        """
        # ── Class balance ──
        dist = Counter(labels.numpy().tolist())
        imbalance = max(dist.values()) / max(min(dist.values()), 1)
        use_balanced = imbalance > 2.0
        weights = _class_weights(labels).to(self.device)
        logger.info("Fusion label imbalance %.1fx — balanced=%s", imbalance, use_balanced)

        sampler = _balanced_sampler(labels) if use_balanced else None
        train_ds = TensorDataset(eeg_emb, speech_emb, labels)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=(sampler is None), sampler=sampler,
            drop_last=True, num_workers=2, pin_memory=True,
        )

        has_val = val_eeg_emb is not None
        val_loader = None
        if has_val:
            val_ds = TensorDataset(val_eeg_emb, val_speech_emb, val_labels)
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size,
                num_workers=2, pin_memory=True,
            )

        # ── Optimiser & loss ──
        optimizer = torch.optim.AdamW(
            self.fusion.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss(
            weight=weights if use_balanced else None,
            label_smoothing=0.1,
        )
        warmup_epochs = max(1, self.epochs // 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6,
        )
        scaler = GradScaler(enabled=self.use_amp)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        }
        best_val_acc = 0.0
        best_state = None
        best_epoch = 0
        patience_ctr = 0
        log_every = self.cfg.training.get("log_every", 5)

        for epoch in range(1, self.epochs + 1):
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

                with autocast(enabled=self.use_amp):
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
            logger.info("Restored best fusion from epoch %d (val_acc=%.4f)", best_epoch, best_val_acc)

        logger.info(
            "Fusion training complete — %d epochs, best_val_acc=%.4f",
            min(epoch, self.epochs), best_val_acc,
        )
        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.fusion.eval()
        correct, total, total_loss = 0, 0, 0.0
        for eeg_b, sp_b, lbl_b in loader:
            eeg_b = eeg_b.to(self.device, non_blocking=True)
            sp_b = sp_b.to(self.device, non_blocking=True)
            lbl_b = lbl_b.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                logits = self.fusion(eeg_b, sp_b)
                total_loss += criterion(logits, lbl_b).item() * eeg_b.size(0)
            correct += (logits.argmax(1) == lbl_b).sum().item()
            total += eeg_b.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)
