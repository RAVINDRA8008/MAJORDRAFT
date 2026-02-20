"""Conditional GAN trainer.

Trains Generator + Discriminator on EEG feature vectors so that
the generator can later produce class-conditioned synthetic samples.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.models.gan import ConditionalGAN
from src.data.dataset import EEGDataset
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


class GANTrainer:
    """Train cGAN on pre-processed EEG feature vectors."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()

        gcfg = cfg.model.gan
        self.gan = ConditionalGAN(
            feature_dim=gcfg.feature_dim,
            noise_dim=gcfg.noise_dim,
            hidden_dim=gcfg.hidden_dim,
            num_classes=cfg.model.num_classes,
            lr_g=gcfg.lr_g,
            lr_d=gcfg.lr_d,
        ).to(self.device)

        self.epochs = gcfg.epochs
        self.batch_size = gcfg.batch_size
        self.log_every = cfg.training.get("log_every", 10)

    # ------------------------------------------------------------------

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        save_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Train the GAN and return loss history.

        Args:
            features: ``(N, feature_dim)``
            labels: ``(N,)`` integer class labels.
            save_dir: Directory for checkpoints (optional).

        Returns:
            ``{"g_loss": [...], "d_loss": [...]}``
        """
        dataset = EEGDataset(features, labels)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        history: dict[str, list[float]] = {"g_loss": [], "d_loss": []}

        for epoch in range(1, self.epochs + 1):
            epoch_g, epoch_d, n_batches = 0.0, 0.0, 0
            for batch_feat, batch_lbl in loader:
                batch_feat = batch_feat.to(self.device)
                batch_lbl = batch_lbl.to(self.device)

                losses = self.gan.train_step(batch_feat, batch_lbl)
                epoch_g += losses["g_loss"]
                epoch_d += losses["d_loss"]
                n_batches += 1

            avg_g = epoch_g / max(n_batches, 1)
            avg_d = epoch_d / max(n_batches, 1)
            history["g_loss"].append(avg_g)
            history["d_loss"].append(avg_d)

            if epoch % self.log_every == 0 or epoch == 1:
                logger.info(
                    "GAN epoch %d/%d — G_loss=%.4f  D_loss=%.4f",
                    epoch, self.epochs, avg_g, avg_d,
                )
                log_gpu_memory()

            # Checkpoint
            if save_dir and epoch % cfg_save_every(self.cfg) == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "gan_state": self.gan.state_dict(),
                        "history": history,
                    },
                    Path(save_dir) / f"gan_epoch_{epoch}.pt",
                )

        logger.info("GAN training complete — %d epochs.", self.epochs)
        return history

    def generate(self, labels: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Generate synthetic EEG features via the trained generator."""
        return self.gan.generate(labels.to(self.device), n)


def cfg_save_every(cfg: DictConfig) -> int:
    """Read checkpoint frequency from config with default."""
    return int(cfg.training.get("save_every", 10))
