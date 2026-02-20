"""Speech encoder pre-trainer.

Trains the CNN-LSTM speech encoder on IEMOCAP features with a
cross-entropy classification objective before injecting it into
the fusion model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.models.speech_encoder import SpeechEncoder
from src.models.classifier import Classifier
from src.data.dataset import SpeechDataset
from src.utils.checkpoint import save_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


class SpeechTrainer:
    """Pre-train speech encoder on IEMOCAP features."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()

        scfg = cfg.model.speech_encoder
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

        self.epochs = scfg.get("pretrain_epochs", 30)
        self.batch_size = scfg.get("batch_size", 64)
        self.lr = scfg.get("lr", 1e-3)

        params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        val_features: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Pre-train speech encoder and return loss/accuracy curves.

        Args:
            features: ``(N, T, n_mfcc)`` padded MFCC features.
            labels: ``(N,)`` integer class labels.
            val_features / val_labels: Optional validation set.
            save_dir: Optional checkpoint directory.

        Returns:
            ``{"train_loss": [...], "train_acc": [...], "val_acc": [...]}``
        """
        train_ds = SpeechDataset(features, labels)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        val_loader = None
        if val_features is not None and val_labels is not None:
            val_ds = SpeechDataset(val_features, val_labels)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [],
        }

        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.head.train()
            running_loss, correct, total = 0.0, 0, 0

            for feat, lbl in train_loader:
                feat, lbl = feat.to(self.device), lbl.to(self.device)

                emb = self.encoder(feat)
                logits = self.head(emb)
                loss = self.criterion(logits, lbl)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * feat.size(0)
                correct += (logits.argmax(1) == lbl).sum().item()
                total += feat.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
            history["val_acc"].append(val_acc)

            if epoch % self.cfg.training.get("log_every", 5) == 0 or epoch == 1:
                logger.info(
                    "Speech epoch %d/%d — loss=%.4f  train_acc=%.3f  val_acc=%.3f",
                    epoch, self.epochs, train_loss, train_acc, val_acc,
                )
                log_gpu_memory()

            if save_dir and epoch % self.cfg.training.get("save_every", 10) == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder": self.encoder.state_dict(),
                        "head": self.head.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    Path(save_dir) / f"speech_epoch_{epoch}.pt",
                )

        logger.info("Speech pre-training complete — %d epochs.", self.epochs)
        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.encoder.eval()
        self.head.eval()
        correct, total = 0, 0
        for feat, lbl in loader:
            feat, lbl = feat.to(self.device), lbl.to(self.device)
            logits = self.head(self.encoder(feat))
            correct += (logits.argmax(1) == lbl).sum().item()
            total += feat.size(0)
        return correct / max(total, 1)
