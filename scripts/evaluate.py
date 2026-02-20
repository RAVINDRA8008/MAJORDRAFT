#!/usr/bin/env python3
"""Evaluate a trained AMERS model on test data.

Generates full metrics, confusion matrix, t-SNE plots, and a
Markdown report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging
from src.utils.device import get_device

from src.data.deap_loader import DEAPLoader
from src.data.iemocap_loader import IEMOCAPLoader
from src.data.label_mapper import LabelMapper

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report_generator import generate_report
from src.utils.visualization import plot_confusion_matrix, plot_tsne


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AMERS model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    device = get_device()

    ckpt = Path(paths["checkpoints"])
    out = Path(paths["outputs"])

    # Load data (full → split → use test portion)
    deap = DEAPLoader(processed_dir=paths["deap_processed"], label_mapper=LabelMapper())
    eeg_feat, eeg_lbl = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"], label_mapper=LabelMapper())
    sp_feat, sp_lbl = iemocap.load_all()

    _, eeg_Xv, _, eeg_yv = train_test_split(
        eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
    )
    _, sp_Xv, _, sp_yv = train_test_split(
        sp_feat, sp_lbl, test_size=0.2, stratify=sp_lbl, random_state=cfg.seed,
    )

    # Load models
    eeg_enc = EEGEncoder(
        input_dim=cfg.model.eeg_encoder.input_dim,
        hidden_dims=list(cfg.model.eeg_encoder.hidden_dims),
        embedding_dim=cfg.model.eeg_encoder.embedding_dim,
    ).to(device)
    eeg_enc.load_state_dict(torch.load(ckpt / "eeg" / "eeg_encoder_final.pt", map_location=device))
    eeg_enc.eval()

    speech_enc = SpeechEncoder(
        n_mfcc=cfg.model.speech_encoder.n_mfcc,
        embedding_dim=cfg.model.speech_encoder.embedding_dim,
    ).to(device)
    speech_enc.load_state_dict(torch.load(ckpt / "speech" / "speech_encoder_final.pt", map_location=device))
    speech_enc.eval()

    fusion = FusionClassifier(
        eeg_dim=cfg.model.fusion.eeg_dim,
        speech_dim=cfg.model.fusion.speech_dim,
        num_classes=cfg.model.num_classes,
    ).to(device)
    # Load best RL checkpoint if available, else baseline
    rl_path = ckpt / "rl" / "best_fusion.pt"
    bl_path = ckpt / "fusion" / "best_fusion_baseline.pt"
    if rl_path.exists():
        sd = torch.load(rl_path, map_location=device)
        fusion.load_state_dict(sd.get("fusion", sd))
    elif bl_path.exists():
        sd = torch.load(bl_path, map_location=device)
        fusion.load_state_dict(sd.get("fusion", sd))
    fusion.eval()

    # Encode & predict
    with torch.no_grad():
        eeg_emb = eeg_enc(torch.as_tensor(eeg_Xv, dtype=torch.float32).to(device))
        sp_emb = speech_enc(torch.as_tensor(sp_Xv, dtype=torch.float32).to(device))
        n = min(len(eeg_emb), len(sp_emb))
        logits = fusion(eeg_emb[:n], sp_emb[:n])
        preds = logits.argmax(1).cpu().numpy()

    labels = eeg_yv[:n]
    metrics = compute_all_metrics(labels, preds)

    print("\n" + metrics["report_str"])

    # Plots
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names=["Happy", "Sad", "Angry", "Neutral"],
        save_path=str(out / "confusion_matrix.png"),
    )

    # t-SNE on fusion embeddings
    with torch.no_grad():
        fused = torch.cat([eeg_emb[:n], sp_emb[:n]], dim=1).cpu().numpy()
    plot_tsne(fused, labels, save_path=str(out / "tsne_embeddings.png"))

    # Report
    results = {"overall_metrics": metrics}
    generate_report(results, output_dir=out)
    print(f"\nReport saved to {out / 'report.md'}")


if __name__ == "__main__":
    main()
