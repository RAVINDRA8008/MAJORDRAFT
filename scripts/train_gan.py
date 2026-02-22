#!/usr/bin/env python3
"""Train the conditional GAN on pre-processed DEAP EEG features.

Autonomous pipeline:
  1. Validate & load DEAP data (handle missing/corrupt subjects)
  2. Auto-size batch to GPU memory
  3. Train with real-time stability monitoring & auto-correction
  4. Select best checkpoint by equilibrium score (not final epoch)
  5. Generate & save training report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging
from src.data.deap_loader import DEAPLoader
from src.training.gan_trainer import GANTrainer
from src.utils.visualization import plot_loss_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cGAN on EEG features")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    # ── 1. Data validation & loading ──────────────────────────────────
    loader = DEAPLoader(processed_dir=paths["deap_processed"])
    found, missing = loader.discover_subjects()

    print(f"\n{'='*52}")
    print(f"  DEAP Dataset Status")
    print(f"{'='*52}")
    print(f"  Subjects found:   {len(found)}/32")
    if missing:
        print(f"  Subjects missing: {', '.join(f's{s:02d}' for s in missing)}")
    else:
        print(f"  Subjects missing: none")
    print(f"{'='*52}\n")

    features, labels, subject_ids = loader.load_all(flatten=True)
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    print(f"  Total samples:    {len(labels)}")
    print(f"  Feature dim:      {features.shape[1]}")
    print(f"  Unique classes:   {labels.unique().tolist()}")

    # ── 2. Train with stability management ────────────────────────────
    save_dir = Path(paths["checkpoints"]) / "gan"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = GANTrainer(cfg)
    history = trainer.fit(features, labels, save_dir=save_dir, auto_batch=True)

    # ── 3. Save final model (for backward compat, also save best) ─────
    torch.save(trainer.gan.state_dict(), save_dir / "gan_final.pt")

    # ── 4. Plot G vs D loss ───────────────────────────────────────────
    plot_loss_curves(
        history["g_loss"],
        val_losses=history["d_loss"],
        title="cGAN Training (G vs D loss)",
        save_path=str(Path(paths["outputs"]) / "gan_loss.png"),
    )

    # ── 5. Print report ───────────────────────────────────────────────
    report_text = trainer.report.render()
    print(f"\n{report_text}")

    # Save report to file
    report_path = Path(paths["outputs"]) / "gan_training_report.txt"
    report_path.write_text(report_text)
    print(f"Report saved: {report_path}")
    print(f"GAN training complete. Checkpoint: {save_dir}")


if __name__ == "__main__":
    main()
