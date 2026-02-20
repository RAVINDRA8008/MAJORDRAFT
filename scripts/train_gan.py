#!/usr/bin/env python3
"""Train the conditional GAN on pre-processed DEAP EEG features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging
from src.data.deap_loader import DEAPLoader
from src.data.label_mapper import LabelMapper
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

    # Load data
    loader = DEAPLoader(processed_dir=paths["deap_processed"], label_mapper=LabelMapper())
    features, labels = loader.load_all(flatten=True)
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    # Train
    save_dir = Path(paths["checkpoints"]) / "gan"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = GANTrainer(cfg)
    history = trainer.fit(features, labels, save_dir=save_dir)

    # Save final model
    torch.save(trainer.gan.state_dict(), save_dir / "gan_final.pt")

    # Plot
    plot_loss_curves(
        {"G_loss": history["g_loss"], "D_loss": history["d_loss"]},
        title="cGAN Training",
        save_path=str(Path(paths["outputs"]) / "gan_loss.png"),
    )
    print("GAN training complete. Checkpoint:", save_dir)


if __name__ == "__main__":
    main()
