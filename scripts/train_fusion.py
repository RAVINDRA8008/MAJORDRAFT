#!/usr/bin/env python3
"""Train the late-fusion classifier (baseline, no RL)."""

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

from src.data.deap_loader import DEAPLoader
from src.data.iemocap_loader import IEMOCAPLoader
from src.data.label_mapper import LabelMapper

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier
from src.training.fusion_trainer import FusionTrainer
from src.utils.device import get_device
from src.utils.visualization import plot_loss_curves, plot_accuracy_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fusion classifier (no RL)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    device = get_device()

    # Load data
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

    # Load pre-trained encoders
    ckpt = Path(paths["checkpoints"])

    eeg_enc = EEGEncoder(
        input_dim=cfg.model.eeg_encoder.input_dim,
        hidden_dims=list(cfg.model.eeg_encoder.hidden_dims),
        embedding_dim=cfg.model.eeg_encoder.embedding_dim,
    ).to(device)
    eeg_ckpt = ckpt / "eeg" / "eeg_encoder_final.pt"
    if eeg_ckpt.exists():
        eeg_enc.load_state_dict(torch.load(eeg_ckpt, map_location=device))
    eeg_enc.eval()

    speech_enc = SpeechEncoder(
        n_mfcc=cfg.model.speech_encoder.n_mfcc,
        embedding_dim=cfg.model.speech_encoder.embedding_dim,
    ).to(device)
    sp_ckpt = ckpt / "speech" / "speech_encoder_final.pt"
    if sp_ckpt.exists():
        speech_enc.load_state_dict(torch.load(sp_ckpt, map_location=device))
    speech_enc.eval()

    # Encode
    with torch.no_grad():
        eeg_emb_t = eeg_enc(torch.as_tensor(eeg_Xt, dtype=torch.float32).to(device)).cpu()
        eeg_emb_v = eeg_enc(torch.as_tensor(eeg_Xv, dtype=torch.float32).to(device)).cpu()
        sp_emb_t = speech_enc(torch.as_tensor(sp_Xt, dtype=torch.float32).to(device)).cpu()
        sp_emb_v = speech_enc(torch.as_tensor(sp_Xv, dtype=torch.float32).to(device)).cpu()

    n_t = min(len(eeg_emb_t), len(sp_emb_t))
    n_v = min(len(eeg_emb_v), len(sp_emb_v))

    # Train
    save_dir = ckpt / "fusion"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = FusionTrainer(cfg)
    history = trainer.fit(
        eeg_emb_t[:n_t], sp_emb_t[:n_t],
        torch.as_tensor(eeg_yt[:n_t], dtype=torch.long),
        val_eeg_emb=eeg_emb_v[:n_v],
        val_speech_emb=sp_emb_v[:n_v],
        val_labels=torch.as_tensor(eeg_yv[:n_v], dtype=torch.long),
        save_dir=save_dir,
    )

    # Plot
    out = Path(paths["outputs"])
    plot_loss_curves({"train_loss": history["train_loss"]}, save_path=str(out / "fusion_loss.png"))
    plot_accuracy_curves(
        {"train": history["train_acc"], "val": history["val_acc"]},
        save_path=str(out / "fusion_acc.png"),
    )
    print("Fusion training complete. Checkpoint:", save_dir)


if __name__ == "__main__":
    main()
