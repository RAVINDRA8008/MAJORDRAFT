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

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report_generator import generate_report
from src.utils.visualization import plot_confusion_matrix, plot_tsne

LABEL_NAMES = ["Angry", "Happy", "Sad", "Neutral"]


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
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_feat, eeg_lbl, _ = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_feat, sp_lbl, _ = iemocap.load_all()

    _, eeg_Xv, _, eeg_yv = train_test_split(
        eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
    )
    _, sp_Xv, _, sp_yv = train_test_split(
        sp_feat, sp_lbl, test_size=0.2, stratify=sp_lbl, random_state=cfg.seed,
    )

    # Load models
    ecfg = cfg.model.eeg_encoder
    eeg_enc = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)
    eeg_enc.load_state_dict(torch.load(ckpt / "eeg" / "eeg_encoder_final.pt", map_location=device))
    eeg_enc.eval()

    scfg = cfg.model.speech_encoder
    speech_enc = SpeechEncoder(
        n_features=scfg.n_mfcc,
        embedding_dim=scfg.embedding_dim,
    ).to(device)
    speech_enc.load_state_dict(torch.load(ckpt / "speech" / "speech_encoder_final.pt", map_location=device))
    speech_enc.eval()

    fcfg = cfg.model.fusion
    fusion = FusionClassifier(
        eeg_embed_dim=fcfg.eeg_dim,
        speech_embed_dim=fcfg.speech_dim,
        hidden_dims=list(fcfg.hidden_dims),
        num_classes=cfg.model.num_classes,
        dropout=[fcfg.dropout, fcfg.dropout] if not isinstance(fcfg.dropout, list) else list(fcfg.dropout),
        modality_dropout_prob=fcfg.modality_dropout,
    ).to(device)
    # Load best RL checkpoint if available, else baseline
    rl_path = ckpt / "rl" / "best_fusion.pt"
    bl_path = ckpt / "fusion" / "best_fusion_baseline.pt"
    if rl_path.exists():
        sd = torch.load(rl_path, map_location=device)
        fusion.load_state_dict(sd.get("fusion", sd))
        print(f"Loaded RL-optimised fusion from {rl_path}")
    elif bl_path.exists():
        sd = torch.load(bl_path, map_location=device)
        fusion.load_state_dict(sd.get("fusion", sd))
        print(f"Loaded baseline fusion from {bl_path}")
    else:
        print("WARNING: No fusion checkpoint found — using random weights")
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

    print("\n" + "=" * 60)
    print("AMERS EVALUATION RESULTS")
    print("=" * 60)
    print(metrics["report_str"])
    print(f"Overall accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"Weighted F1:       {metrics['f1_weighted']:.4f}")
    print(f"Cohen's Kappa:     {metrics['cohens_kappa']:.4f}")
    print("=" * 60)

    # Plots
    plot_confusion_matrix(
        labels,
        preds,
        labels=LABEL_NAMES,
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
