#!/usr/bin/env python3
"""v3 — Domain adversarial training (DANN) for cross-modal alignment."""

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
from src.data.iemocap_loader import IEMOCAPLoader
from src.models.domain_adapter import DomainAdaptationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="v3: DANN domain adaptation")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    # Load data
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_feat, eeg_lbl, _ = deap.load_all(flatten=True)

    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_feat, sp_lbl, _ = iemocap.load_all()

    print(f"DANN training: EEG={len(eeg_feat)} samples, Speech={len(sp_feat)} samples")

    save_dir = Path(paths["checkpoints"]) / "v3"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = DomainAdaptationTrainer(cfg)

    # Load contrastive-pretrained weights if available
    trainer.load_pretrained(save_dir)

    history = trainer.train(
        eeg_feat, eeg_lbl,
        sp_feat, sp_lbl,
        save_dir=save_dir,
    )

    print(f"DANN training done — final emotion_loss: {history['emotion_loss'][-1]:.4f}")
    print(f"  domain_loss: {history['domain_loss'][-1]:.4f}")
    print(f"Checkpoints: {save_dir / 'eeg_encoder_dann.pt'}, {save_dir / 'speech_encoder_dann.pt'}")


if __name__ == "__main__":
    main()
