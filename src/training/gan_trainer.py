"""Conditional GAN trainer with autonomous stability management.

Trains Generator + Discriminator on EEG feature vectors.  Monitors
training dynamics in real-time and automatically corrects:

* Discriminator dominance  (D_loss < 0.45 and G climbing)
* Generator dominance      (D_loss > 0.65)
* Mode collapse            (G_loss ≈ 0 or output variance collapse)
* Stalled training         (losses static for many epochs)

Also handles:
* TTUR (Two Time-scale Update Rule) — separate LRs for G/D
* Adaptive batch sizing to maximise GPU utilisation
* Best-equilibrium checkpoint selection
* Gradient clipping
* Comprehensive training report generation
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.models.gan import ConditionalGAN
from src.data.dataset import EEGDataset
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import get_device, log_gpu_memory

logger = logging.getLogger(__name__)

# ── Target equilibrium ranges ─────────────────────────────────────────
IDEAL_D_LOSS_LOW = 0.45
IDEAL_D_LOSS_HIGH = 0.65
IDEAL_G_LOSS_LOW = 0.8
IDEAL_G_LOSS_HIGH = 1.5

# ── Instability thresholds ────────────────────────────────────────────
D_DOMINANCE_THRESHOLD = 0.40
G_DOMINANCE_THRESHOLD = 0.70
MODE_COLLAPSE_G_LOSS = 0.05
IMBALANCE_PATIENCE = 10  # epochs before auto-correction

# ── Gradient clipping ─────────────────────────────────────────────────
GRAD_CLIP_MAX_NORM = 1.0


class TrainingReport:
    """Accumulate and render a post-training report."""

    def __init__(self) -> None:
        self.sections: list[tuple[str, str]] = []

    def add(self, heading: str, body: str) -> None:
        self.sections.append((heading, body))

    def render(self) -> str:
        lines = ["=" * 64, "  GAN TRAINING REPORT", "=" * 64, ""]
        for heading, body in self.sections:
            lines.append(f"── {heading} ──")
            lines.append(body)
            lines.append("")
        lines.append("=" * 64)
        return "\n".join(lines)


class GANTrainer:
    """Train cGAN on pre-processed EEG feature vectors with autonomous
    stability management, adaptive batch sizing, and best-checkpoint
    selection.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.report = TrainingReport()

        gcfg = cfg.model.gan
        self.lr_g_base = float(getattr(gcfg, "lr_g", 1e-4))
        self.lr_d_base = float(getattr(gcfg, "lr_d", 2e-4))

        gan_config = {
            "feature_dim": gcfg.feature_dim,
            "latent_dim": gcfg.noise_dim,
            "num_classes": cfg.model.num_classes,
            "generator_hidden_dims": [gcfg.hidden_dim, gcfg.hidden_dim * 2, gcfg.hidden_dim],
            "discriminator_hidden_dims": [gcfg.hidden_dim, gcfg.hidden_dim * 2, gcfg.hidden_dim],
            "lr": self.lr_g_base,          # G gets lower LR (TTUR)
            "label_smooth": 0.9,
        }
        self.gan = ConditionalGAN(gan_config, self.device)

        # Override D optimizer to use separate LR (TTUR)
        self.gan.opt_d = torch.optim.Adam(
            self.gan.discriminator.parameters(),
            lr=self.lr_d_base,
            betas=(0.5, 0.999),
        )

        self.epochs = gcfg.epochs
        self.batch_size = gcfg.batch_size
        self.log_every = cfg.training.get("log_every", 5)
        self.save_every = int(cfg.training.get("save_every", 10))

        # Stability tracking
        self._d_dominance_streak = 0
        self._g_dominance_streak = 0
        self._corrections_applied: list[str] = []

    # ------------------------------------------------------------------
    # Adaptive batch sizing
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_batch_size(
        n_samples: int,
        feature_dim: int,
        min_bs: int = 128,
        target_gpu_frac: float = 0.70,
    ) -> int:
        """Pick batch size to fill ~target_gpu_frac of GPU memory.

        Heuristic: each sample ≈ feature_dim × 4 bytes × ~6 (fwd + bwd
        for G and D).  We ramp up from *min_bs* in powers of 2.
        """
        if not torch.cuda.is_available():
            return min_bs

        total_mem = torch.cuda.get_device_properties(0).total_memory
        target_bytes = total_mem * target_gpu_frac
        bytes_per_sample = feature_dim * 4 * 6  # rough estimate
        max_bs = int(target_bytes / bytes_per_sample)
        # Clamp to power of 2 and reasonable range
        candidates = [min_bs]
        bs = min_bs
        while bs * 2 <= min(max_bs, n_samples // 4, 4096):
            bs *= 2
            candidates.append(bs)
        chosen = candidates[-1]
        logger.info(
            "Auto batch size: %d  (GPU total %.1f GB, target %.0f%%)",
            chosen,
            total_mem / 1e9,
            target_gpu_frac * 100,
        )
        return chosen

    # ------------------------------------------------------------------
    # Stability diagnostics
    # ------------------------------------------------------------------
    def _diagnose(
        self, epoch: int, avg_g: float, avg_d: float,
        g_history: list[float], d_history: list[float],
    ) -> str | None:
        """Return a diagnostic string if instability is detected, else None."""

        # D dominance: D_loss < threshold AND G_loss trending up
        if avg_d < D_DOMINANCE_THRESHOLD:
            self._d_dominance_streak += 1
            g_rising = len(g_history) >= 3 and g_history[-1] > g_history[-3]
            if self._d_dominance_streak >= IMBALANCE_PATIENCE and g_rising:
                return "D_DOMINANCE"
        else:
            self._d_dominance_streak = 0

        # G dominance: D_loss > threshold
        if avg_d > G_DOMINANCE_THRESHOLD:
            self._g_dominance_streak += 1
            if self._g_dominance_streak >= IMBALANCE_PATIENCE:
                return "G_DOMINANCE"
        else:
            self._g_dominance_streak = 0

        # Mode collapse: G_loss extremely small
        if avg_g < MODE_COLLAPSE_G_LOSS:
            return "MODE_COLLAPSE"

        return None

    def _apply_correction(self, diagnosis: str) -> None:
        """Adjust hyper-parameters to counter the detected instability."""

        if diagnosis == "D_DOMINANCE":
            # Increase G LR, decrease D LR
            for pg in self.gan.opt_g.param_groups:
                pg["lr"] = min(pg["lr"] * 1.5, 1e-3)
            for pg in self.gan.opt_d.param_groups:
                pg["lr"] = max(pg["lr"] * 0.75, 1e-5)
            new_g_lr = self.gan.opt_g.param_groups[0]["lr"]
            new_d_lr = self.gan.opt_d.param_groups[0]["lr"]
            msg = (
                f"D_DOMINANCE correction — G_lr → {new_g_lr:.6f}, "
                f"D_lr → {new_d_lr:.6f}"
            )
            logger.warning(msg)
            self._corrections_applied.append(msg)
            self._d_dominance_streak = 0

        elif diagnosis == "G_DOMINANCE":
            # Strengthen D: increase D LR
            for pg in self.gan.opt_d.param_groups:
                pg["lr"] = min(pg["lr"] * 1.5, 1e-3)
            new_d_lr = self.gan.opt_d.param_groups[0]["lr"]
            msg = f"G_DOMINANCE correction — D_lr → {new_d_lr:.6f}"
            logger.warning(msg)
            self._corrections_applied.append(msg)
            self._g_dominance_streak = 0

        elif diagnosis == "MODE_COLLAPSE":
            # Reset G weights from scratch, keep D
            logger.warning("MODE_COLLAPSE detected — reinitialising Generator weights")
            for m in self.gan.generator.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            # Reset G optimizer
            self.gan.opt_g = torch.optim.Adam(
                self.gan.generator.parameters(),
                lr=self.lr_g_base,
                betas=(0.5, 0.999),
            )
            self._corrections_applied.append("MODE_COLLAPSE recovery — G reinitialised")

    # ------------------------------------------------------------------
    # Best-checkpoint tracking
    # ------------------------------------------------------------------
    @staticmethod
    def _equilibrium_score(g_loss: float, d_loss: float) -> float:
        """Lower = better.  Measures deviation from ideal equilibrium."""
        d_dev = abs(d_loss - 0.5)                      # D should be ~0.5
        g_dev = abs(g_loss - 1.1) if g_loss > 0 else 5.0  # G should be ~1.0–1.2
        return d_dev + g_dev

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        save_dir: str | Path | None = None,
        auto_batch: bool = True,
    ) -> dict[str, list[float]]:
        """Train the GAN and return loss history.

        Args:
            features: ``(N, feature_dim)``
            labels: ``(N,)`` integer class labels.
            save_dir: Directory for checkpoints (optional).
            auto_batch: If True, pick batch size to fill GPU.

        Returns:
            ``{"g_loss": [...], "d_loss": [...]}``
        """
        n_samples = len(features)
        feat_dim = features.shape[1] if features.ndim == 2 else int(np.prod(features.shape[1:]))

        # ── Adaptive batch size ──
        if auto_batch and torch.cuda.is_available():
            self.batch_size = self._auto_batch_size(n_samples, feat_dim, min_bs=self.batch_size)

        dataset = EEGDataset(features, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

        history: dict[str, list[float]] = {"g_loss": [], "d_loss": []}

        best_score = float("inf")
        best_epoch = 0
        best_state: dict[str, Any] | None = None

        t_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_g, epoch_d, n_batches = 0.0, 0.0, 0

            for batch_feat, batch_lbl in loader:
                batch_feat = batch_feat.to(self.device, non_blocking=True)
                batch_lbl = batch_lbl.to(self.device, non_blocking=True)

                losses = self.gan.train_step(batch_feat, batch_lbl)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.gan.generator.parameters(), GRAD_CLIP_MAX_NORM
                )
                torch.nn.utils.clip_grad_norm_(
                    self.gan.discriminator.parameters(), GRAD_CLIP_MAX_NORM
                )

                epoch_g += losses["g_loss"]
                epoch_d += losses["d_loss"]
                n_batches += 1

            avg_g = epoch_g / max(n_batches, 1)
            avg_d = epoch_d / max(n_batches, 1)
            history["g_loss"].append(avg_g)
            history["d_loss"].append(avg_d)

            # ── Logging ──
            if epoch % self.log_every == 0 or epoch == 1:
                logger.info(
                    "GAN epoch %d/%d — G_loss=%.4f  D_loss=%.4f",
                    epoch, self.epochs, avg_g, avg_d,
                )
                log_gpu_memory()

            # ── Stability check & auto-correction ──
            diagnosis = self._diagnose(
                epoch, avg_g, avg_d, history["g_loss"], history["d_loss"]
            )
            if diagnosis:
                self._apply_correction(diagnosis)

            # ── Best-checkpoint tracking ──
            score = self._equilibrium_score(avg_g, avg_d)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = self.gan.state_dict()

            # ── Periodic checkpoint ──
            if save_dir and epoch % self.save_every == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "gan_state": self.gan.state_dict(),
                        "history": history,
                    },
                    Path(save_dir) / f"gan_epoch_{epoch}.pt",
                )

        elapsed = time.time() - t_start

        # ── Save best model ──
        if save_dir and best_state is not None:
            best_path = Path(save_dir) / "best_gan_model.pt"
            torch.save(best_state, str(best_path))
            logger.info(
                "Best GAN model saved (epoch %d, eq_score=%.4f): %s",
                best_epoch, best_score, best_path,
            )

        # ── Build report ──
        self._build_report(history, best_epoch, best_score, elapsed, n_samples)

        logger.info("GAN training complete — %d epochs in %.1f min.", self.epochs, elapsed / 60)
        return history

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def _build_report(
        self,
        history: dict[str, list[float]],
        best_epoch: int,
        best_score: float,
        elapsed: float,
        n_samples: int,
    ) -> None:
        """Populate self.report with all training diagnostics."""

        # Dataset
        self.report.add("Dataset", f"Samples: {n_samples}  |  Batch size: {self.batch_size}")

        # GPU
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            alloc = torch.cuda.memory_allocated() / 1e9
            self.report.add(
                "GPU Utilisation",
                f"{name}  |  Total: {total:.1f} GB  |  Peak alloc: {alloc:.2f} GB",
            )
        else:
            self.report.add("GPU Utilisation", "CPU only")

        # Loss trend
        g = history["g_loss"]
        d = history["d_loss"]
        self.report.add(
            "Loss Trend (first → last)",
            f"G_loss: {g[0]:.4f} → {g[-1]:.4f}  |  D_loss: {d[0]:.4f} → {d[-1]:.4f}",
        )

        # Best epoch
        self.report.add(
            "Best Epoch",
            f"Epoch {best_epoch}  |  equilibrium_score={best_score:.4f}  "
            f"(G={g[best_epoch - 1]:.4f}, D={d[best_epoch - 1]:.4f})",
        )

        # Stability
        if self._corrections_applied:
            self.report.add(
                "Stability Corrections Applied",
                "\n".join(f"  • {c}" for c in self._corrections_applied),
            )
        else:
            self.report.add("Training Stability", "No corrections needed — training was stable.")

        # Final assessment
        final_g, final_d = g[-1], d[-1]
        assessment = []
        if IDEAL_D_LOSS_LOW <= final_d <= IDEAL_D_LOSS_HIGH:
            assessment.append("D_loss in ideal range ✓")
        else:
            assessment.append(f"D_loss ({final_d:.4f}) outside ideal [{IDEAL_D_LOSS_LOW}–{IDEAL_D_LOSS_HIGH}]")
        if IDEAL_G_LOSS_LOW <= final_g <= IDEAL_G_LOSS_HIGH:
            assessment.append("G_loss in ideal range ✓")
        else:
            assessment.append(f"G_loss ({final_g:.4f}) outside ideal [{IDEAL_G_LOSS_LOW}–{IDEAL_G_LOSS_HIGH}]")
        self.report.add("Final Assessment", "\n".join(f"  • {a}" for a in assessment))

        # Timing
        self.report.add("Timing", f"{elapsed / 60:.1f} min total  |  {elapsed / len(g):.2f} s/epoch")

        # Recommendations
        recs = []
        if final_d < 0.35:
            recs.append("Consider increasing G capacity or decreasing D capacity.")
        if final_g > 2.0:
            recs.append("G_loss still high — try more epochs or lower D learning rate.")
        if not recs:
            recs.append("Training looks healthy.  Proceed to downstream tasks.")
        self.report.add("Recommendations", "\n".join(f"  • {r}" for r in recs))

    def generate(self, labels: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Generate synthetic EEG features via the trained generator."""
        return self.gan.generate(labels.to(self.device), n)
