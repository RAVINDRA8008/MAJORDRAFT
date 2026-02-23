#!/usr/bin/env python3
"""Evaluate a trained AMERS model on test data.

Uses LABEL-ALIGNED pairing: EEG and speech samples are matched by
emotion class, not by index.  Also evaluates each modality individually.

Generates: confusion matrix, t-SNE plots, Markdown report.
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

# Maps numeric labels to emotion names
LABEL_NAMES = ["Angry", "Happy", "Sad", "Neutral"]


def _encode_batched(encoder, data, device, batch_size=512):
    """Encode data through an encoder in batches to avoid OOM."""
    parts = []
    t = torch.as_tensor(data, dtype=torch.float32)
    for i in range(0, len(t), batch_size):
        parts.append(encoder(t[i : i + batch_size].to(device)).cpu())
    return torch.cat(parts, dim=0)


def _label_aligned_pairs(eeg_emb, eeg_labels, sp_emb, sp_labels, num_classes=4):
    """Create label-aligned (eeg, speech, label) pairs for evaluation.

    For each class, pairs up to min(n_eeg, n_speech) samples randomly.
    Returns flat tensors ready for evaluation.
    """
    eeg_list, sp_list, lbl_list = [], [], []
    for c in range(num_classes):
        eeg_mask = eeg_labels == c
        sp_mask = sp_labels == c
        eeg_c = eeg_emb[eeg_mask]
        sp_c = sp_emb[sp_mask]
        n = min(len(eeg_c), len(sp_c))
        if n == 0:
            continue
        # Shuffle and take n
        eeg_perm = torch.randperm(len(eeg_c))[:n]
        sp_perm = torch.randperm(len(sp_c))[:n]
        eeg_list.append(eeg_c[eeg_perm])
        sp_list.append(sp_c[sp_perm])
        lbl_list.append(torch.full((n,), c, dtype=torch.long))

    return torch.cat(eeg_list), torch.cat(sp_list), torch.cat(lbl_list)


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

    # ── Load data (full → split → use test portion) ──
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

    # ── Load models ──
    ecfg = cfg.model.eeg_encoder
    eeg_enc = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)
    eeg_enc.load_state_dict(
        torch.load(ckpt / "eeg" / "eeg_encoder_final.pt", map_location=device)
    )
    eeg_enc.eval()

    scfg = cfg.model.speech_encoder
    speech_enc = SpeechEncoder(
        n_features=scfg.n_mfcc,
        embedding_dim=scfg.embedding_dim,
    ).to(device)
    speech_enc.load_state_dict(
        torch.load(ckpt / "speech" / "speech_encoder_final.pt", map_location=device)
    )
    speech_enc.eval()

    fcfg = cfg.model.fusion
    dropout_val = fcfg.dropout
    if isinstance(dropout_val, (int, float)):
        dropout_list = [float(dropout_val)] * len(list(fcfg.hidden_dims))
    else:
        dropout_list = list(dropout_val)

    fusion = FusionClassifier(
        eeg_embed_dim=fcfg.eeg_dim,
        speech_embed_dim=fcfg.speech_dim,
        hidden_dims=list(fcfg.hidden_dims),
        num_classes=cfg.model.num_classes,
        dropout=dropout_list,
        modality_dropout_prob=0.0,  # no dropout at eval
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

    # ── Encode validation data ──
    with torch.no_grad():
        eeg_emb = _encode_batched(eeg_enc, eeg_Xv, device)
        sp_emb = _encode_batched(speech_enc, sp_Xv, device)

    eeg_yv_t = torch.as_tensor(eeg_yv, dtype=torch.long)
    sp_yv_t = torch.as_tensor(sp_yv, dtype=torch.long)

    # ── Label-aligned evaluation ──
    eeg_paired, sp_paired, labels_paired = _label_aligned_pairs(
        eeg_emb, eeg_yv_t, sp_emb, sp_yv_t, num_classes=cfg.model.num_classes,
    )

    print(f"\nLabel-aligned evaluation: {len(labels_paired)} paired samples")
    from collections import Counter
    dist = Counter(labels_paired.numpy().tolist())
    for c in sorted(dist):
        print(f"  Class {c} ({LABEL_NAMES[c]}): {dist[c]} pairs")

    # ── Predict in batches ──
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(labels_paired), 512):
            end = min(i + 512, len(labels_paired))
            logits = fusion(
                eeg_paired[i:end].to(device),
                sp_paired[i:end].to(device),
            )
            all_preds.append(logits.argmax(1).cpu())
        preds = torch.cat(all_preds).numpy()

    labels = labels_paired.numpy()
    metrics = compute_all_metrics(labels, preds, label_names=LABEL_NAMES)

    # ── Print results ──
    print("\n" + "=" * 60)
    print("AMERS EVALUATION RESULTS (label-aligned)")
    print("=" * 60)
    print(metrics["report_str"])
    print(f"Overall accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"Weighted F1:       {metrics['f1_weighted']:.4f}")
    print(f"Cohen's Kappa:     {metrics['kappa']:.4f}")

    # ── Per-modality accuracy (using simple classifier heads if available) ──
    # EEG standalone: predict most-common class per embedding cluster
    eeg_only_path = ckpt / "eeg" / "eeg_encoder_final.pt"
    if eeg_only_path.exists():
        # Simple nearest-centroid classifier for EEG
        eeg_train_feat, _, eeg_train_lbl, _ = train_test_split(
            eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
        )
        with torch.no_grad():
            eeg_train_emb = _encode_batched(eeg_enc, eeg_train_feat, device)
        eeg_train_lbl_t = torch.as_tensor(eeg_train_lbl, dtype=torch.long)
        # Compute class centroids
        centroids = []
        for c in range(cfg.model.num_classes):
            mask = eeg_train_lbl_t == c
            if mask.any():
                centroids.append(eeg_train_emb[mask].mean(0))
            else:
                centroids.append(torch.zeros(eeg_emb.shape[1]))
        centroids = torch.stack(centroids)  # (num_classes, embed_dim)
        # Classify val by nearest centroid
        dists = torch.cdist(eeg_emb, centroids)  # (n_val, num_classes)
        eeg_preds = dists.argmin(1).numpy()
        eeg_acc = float((eeg_preds == eeg_yv).mean())
        print(f"\nEEG-only accuracy (nearest centroid): {eeg_acc:.4f}")

    print("=" * 60)

    # ── Plots ──
    plot_confusion_matrix(
        labels, preds, labels=LABEL_NAMES,
        save_path=str(out / "confusion_matrix.png"),
    )

    # t-SNE on fused embeddings
    with torch.no_grad():
        fused = torch.cat([eeg_paired, sp_paired], dim=1).numpy()
    plot_tsne(fused, labels, save_path=str(out / "tsne_embeddings.png"))

    # ── Report ──
    results = {"overall_metrics": metrics}
    generate_report(results, output_dir=out)
    print(f"\nReport saved to {out / 'report.md'}")


if __name__ == "__main__":
    main()
