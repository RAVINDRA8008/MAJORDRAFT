#!/usr/bin/env python3
"""Pre-train the speech (CNN-LSTM) encoder on IEMOCAP features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging
from src.data.iemocap_loader import IEMOCAPLoader
from src.data.label_mapper import LabelMapper
from src.training.speech_trainer import SpeechTrainer
from src.utils.visualization import plot_loss_curves, plot_accuracy_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train speech encoder")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    # Load data
    loader = IEMOCAPLoader(processed_dir=paths["iemocap_processed"], label_mapper=LabelMapper())
    features, labels = loader.load_all()

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=cfg.seed,
    )

    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.long)

    # Train
    save_dir = Path(paths["checkpoints"]) / "speech"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = SpeechTrainer(cfg)
    history = trainer.fit(X_train, y_train, X_val, y_val, save_dir=save_dir)

    # Save
    torch.save(trainer.encoder.state_dict(), save_dir / "speech_encoder_final.pt")

    # Plot
    out = Path(paths["outputs"])
    plot_loss_curves(
        {"train_loss": history["train_loss"]},
        title="Speech Encoder Pre-Training Loss",
        save_path=str(out / "speech_loss.png"),
    )
    plot_accuracy_curves(
        {"train": history["train_acc"], "val": history["val_acc"]},
        title="Speech Encoder Accuracy",
        save_path=str(out / "speech_acc.png"),
    )
    print("Speech pre-training complete. Checkpoint:", save_dir)


if __name__ == "__main__":
    main()
