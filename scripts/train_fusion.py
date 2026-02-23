#!/usr/bin/env python3
"""Train the late-fusion classifier (baseline, no RL).

Key change: uses LABEL-ALIGNED pairing — EEG and speech samples are
matched by emotion class, not by index.  This creates semantically
meaningful cross-modal pairs from two separate datasets (DEAP + IEMOCAP).
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

from src.data.deap_loader import DEAPLoader
from src.data.iemocap_loader import IEMOCAPLoader

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
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

    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Load data
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_feat, eeg_lbl, _ = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_feat, sp_lbl, _ = iemocap.load_all()

    # Split EACH modality independently (stratified)
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
        dropout=cfg.model.eeg_encoder.dropout,
    ).to(device)
    eeg_ckpt = ckpt / "eeg" / "eeg_encoder_final.pt"
    if eeg_ckpt.exists():
        eeg_enc.load_state_dict(torch.load(eeg_ckpt, map_location=device))
    eeg_enc.eval()

    speech_enc = SpeechEncoder(
        n_features=cfg.model.speech_encoder.n_mfcc,
        embedding_dim=cfg.model.speech_encoder.embedding_dim,
    ).to(device)
    sp_ckpt = ckpt / "speech" / "speech_encoder_final.pt"
    if sp_ckpt.exists():
        speech_enc.load_state_dict(torch.load(sp_ckpt, map_location=device))
    speech_enc.eval()

    # Encode in batches to avoid OOM
    def _encode_batched(encoder, data, batch_size=512):
        parts = []
        t = torch.as_tensor(data, dtype=torch.float32)
        for i in range(0, len(t), batch_size):
            chunk = t[i : i + batch_size].to(device)
            parts.append(encoder(chunk).cpu())
        return torch.cat(parts, dim=0)

    with torch.no_grad():
        eeg_emb_t = _encode_batched(eeg_enc, eeg_Xt)
        eeg_emb_v = _encode_batched(eeg_enc, eeg_Xv)
        sp_emb_t = _encode_batched(speech_enc, sp_Xt)
        sp_emb_v = _encode_batched(speech_enc, sp_Xv)

    # Convert labels to tensors
    eeg_yt_t = torch.as_tensor(eeg_yt, dtype=torch.long)
    eeg_yv_t = torch.as_tensor(eeg_yv, dtype=torch.long)
    sp_yt_t = torch.as_tensor(sp_yt, dtype=torch.long)
    sp_yv_t = torch.as_tensor(sp_yv, dtype=torch.long)

    print(f"\nLabel-aligned fusion training:")
    print(f"  EEG train: {len(eeg_emb_t)} embeddings, Speech train: {len(sp_emb_t)} embeddings")
    print(f"  EEG val:   {len(eeg_emb_v)} embeddings, Speech val:   {len(sp_emb_v)} embeddings")
    print(f"  Pairing strategy: LABEL-ALIGNED (same emotion class)\n")

    # Train using label-aligned pairing
    save_dir = ckpt / "fusion"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = FusionTrainer(cfg)
    history = trainer.fit(
        eeg_emb=eeg_emb_t,
        eeg_labels=eeg_yt_t,
        speech_emb=sp_emb_t,
        speech_labels=sp_yt_t,
        val_eeg_emb=eeg_emb_v,
        val_eeg_labels=eeg_yv_t,
        val_speech_emb=sp_emb_v,
        val_speech_labels=sp_yv_t,
        save_dir=save_dir,
    )

    # Plot
    out = Path(paths["outputs"])
    plot_loss_curves(
        history["train_loss"],
        history.get("val_loss", []),
        title="Fusion Loss",
        save_path=str(out / "fusion_loss.png"),
    )
    plot_accuracy_curves(
        history["train_acc"],
        history["val_acc"],
        title="Fusion Accuracy",
        save_path=str(out / "fusion_acc.png"),
    )
    print("Fusion training complete. Checkpoint:", save_dir)


if __name__ == "__main__":
    main()
