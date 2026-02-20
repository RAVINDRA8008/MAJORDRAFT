#!/usr/bin/env python3
"""Train the RL (PPO) agent for augmentation control.

This script runs the full RL loop: at each step the agent selects an
augmentation ratio, synthetic data is generated, the fusion model trains
one epoch, and the reward is computed from validation metrics.
"""

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
from src.utils.checkpoint import load_model

from src.data.deap_loader import DEAPLoader
from src.data.iemocap_loader import IEMOCAPLoader
from src.data.label_mapper import LabelMapper

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.gan import ConditionalGAN
from src.models.fusion import FusionClassifier

from src.training.rl_trainer import RLTrainer
from src.utils.visualization import plot_augmentation_ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)

    # ---- Load pre-trained components ----
    ckpt = Path(paths["checkpoints"])

    gan = ConditionalGAN(
        feature_dim=cfg.model.gan.feature_dim,
        noise_dim=cfg.model.gan.noise_dim,
        hidden_dim=cfg.model.gan.hidden_dim,
        num_classes=cfg.model.num_classes,
    )
    gan.load_state_dict(torch.load(ckpt / "gan" / "gan_final.pt", map_location="cpu"))

    eeg_enc = EEGEncoder(
        input_dim=cfg.model.eeg_encoder.input_dim,
        hidden_dims=list(cfg.model.eeg_encoder.hidden_dims),
        embedding_dim=cfg.model.eeg_encoder.embedding_dim,
    )
    eeg_enc.load_state_dict(torch.load(ckpt / "eeg" / "eeg_encoder_final.pt", map_location="cpu"))

    speech_enc = SpeechEncoder(
        n_mfcc=cfg.model.speech_encoder.n_mfcc,
        embedding_dim=cfg.model.speech_encoder.embedding_dim,
    )
    speech_enc.load_state_dict(torch.load(ckpt / "speech" / "speech_encoder_final.pt", map_location="cpu"))

    fusion = FusionClassifier(
        eeg_dim=cfg.model.fusion.eeg_dim,
        speech_dim=cfg.model.fusion.speech_dim,
        num_classes=cfg.model.num_classes,
    )
    # Optionally load warm-up checkpoint
    warmup_path = ckpt / "fusion" / "best_fusion_baseline.pt"
    if warmup_path.exists():
        sd = torch.load(warmup_path, map_location="cpu")
        fusion.load_state_dict(sd.get("fusion", sd))

    # ---- Load data ----
    deap = DEAPLoader(processed_dir=paths["deap_processed"], label_mapper=LabelMapper())
    eeg_feat, eeg_lbl = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"], label_mapper=LabelMapper())
    sp_feat, sp_lbl = iemocap.load_all()

    eeg_Xt, eeg_Xv, eeg_yt, eeg_yv = train_test_split(
        eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
    )
    sp_Xt, sp_Xv, sp_yt, sp_yv = train_test_split(
        sp_feat, sp_lbl, test_size=0.2, stratify=sp_lbl, random_state=cfg.seed,
    )

    # ---- Train RL ----
    rl_save = ckpt / "rl"
    rl_save.mkdir(parents=True, exist_ok=True)

    trainer = RLTrainer(cfg, gan=gan, eeg_encoder=eeg_enc, speech_encoder=speech_enc, fusion=fusion)
    history = trainer.train(
        torch.as_tensor(eeg_Xt, dtype=torch.float32),
        torch.as_tensor(eeg_yt, dtype=torch.long),
        torch.as_tensor(sp_Xt, dtype=torch.float32),
        torch.as_tensor(sp_yt, dtype=torch.long),
        torch.as_tensor(eeg_Xv, dtype=torch.float32),
        torch.as_tensor(eeg_yv, dtype=torch.long),
        torch.as_tensor(sp_Xv, dtype=torch.float32),
        torch.as_tensor(sp_yv, dtype=torch.long),
        save_dir=rl_save,
    )

    plot_augmentation_ratios(
        history["aug_ratio"],
        save_path=str(Path(paths["outputs"]) / "rl_aug_ratios.png"),
    )
    print("RL training complete. Checkpoint:", rl_save)


if __name__ == "__main__":
    main()
