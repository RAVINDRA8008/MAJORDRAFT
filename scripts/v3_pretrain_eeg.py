#!/usr/bin/env python3
"""v3 — Contrastive pretraining for EEG encoder (SimCLR with NT-Xent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging
from src.utils.device import get_device

from src.data.deap_loader import DEAPLoader
from src.pretraining.contrastive_eeg import ContrastiveEEGTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="v3: Contrastive EEG pretraining")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    # Load DEAP data
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    features, _, _ = deap.load_all(flatten=True)
    print(f"Contrastive EEG pretraining on {len(features)} samples, shape={features.shape}")

    save_dir = Path(paths["checkpoints"]) / "v3"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = ContrastiveEEGTrainer(cfg)
    history = trainer.train(features, save_dir=save_dir)

    print(f"Contrastive EEG done — final loss: {history['loss'][-1]:.4f}")
    print(f"Checkpoint: {save_dir / 'eeg_encoder_contrastive.pt'}")


if __name__ == "__main__":
    main()
