"""Fusion trainer (without RL).

Trains the late-fusion classifier on pre-encoded EEG + speech
embeddings.  Used for:
  • The non-RL baseline
  • The initial warm-up before RL kicks in
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

from src.models.fusion import FusionClassifier
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


class FusionTrainer:
    """Train FusionClassifier on (eeg_emb, speech_emb, label) triples."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()

        fcfg = cfg.model.fusion
        self.fusion = FusionClassifier(
            eeg_dim=fcfg.eeg_dim,
            speech_dim=fcfg.speech_dim,
            hidden_dims=list(fcfg.hidden_dims),
            num_classes=cfg.model.num_classes,
            dropout=fcfg.dropout,
            modality_dropout=fcfg.modality_dropout,
        ).to(self.device)

        self.epochs = fcfg.get("epochs", 50)
        self.batch_size = fcfg.get("batch_size", 64)
        self.lr = fcfg.get("lr", 1e-3)

        self.optimizer = torch.optim.Adam(self.fusion.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5,
        )

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

        Returns history with ``train_loss``, ``train_acc``, ``val_acc``.
        """
        train_ds = TensorDataset(eeg_emb, speech_emb, labels)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        has_val = val_eeg_emb is not None
        val_loader = None
        if has_val:
            val_ds = TensorDataset(val_eeg_emb, val_speech_emb, val_labels)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [],
        }
        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            self.fusion.train()
            running_loss, correct, total = 0.0, 0, 0

            for eeg_b, sp_b, lbl_b in train_loader:
                eeg_b = eeg_b.to(self.device)
                sp_b = sp_b.to(self.device)
                lbl_b = lbl_b.to(self.device)

                logits = self.fusion(eeg_b, sp_b)
                loss = self.criterion(logits, lbl_b)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * eeg_b.size(0)
                correct += (logits.argmax(1) == lbl_b).sum().item()
                total += eeg_b.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                self.scheduler.step(val_acc)
            history["val_acc"].append(val_acc)

            if epoch % self.cfg.training.get("log_every", 5) == 0 or epoch == 1:
                logger.info(
                    "Fusion epoch %d/%d — loss=%.4f  train_acc=%.3f  val_acc=%.3f",
                    epoch, self.epochs, train_loss, train_acc, val_acc,
                )
                log_gpu_memory()

            # Best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_dir:
                    save_checkpoint(
                        {"epoch": epoch, "fusion": self.fusion.state_dict()},
                        Path(save_dir) / "best_fusion_baseline.pt",
                    )

        logger.info(
            "Fusion training complete — %d epochs, best_val_acc=%.4f",
            self.epochs, best_val_acc,
        )
        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.fusion.eval()
        correct, total = 0, 0
        for eeg_b, sp_b, lbl_b in loader:
            eeg_b = eeg_b.to(self.device)
            sp_b = sp_b.to(self.device)
            lbl_b = lbl_b.to(self.device)
            logits = self.fusion(eeg_b, sp_b)
            correct += (logits.argmax(1) == lbl_b).sum().item()
            total += eeg_b.size(0)
        return correct / max(total, 1)
