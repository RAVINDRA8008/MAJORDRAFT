#!/usr/bin/env python3
"""v3 — Evaluate the full v3 pipeline (transformer fusion + DANN encoders).

Generates: confusion matrix, t-SNE, classification report, Markdown report.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from src.models.transformer_fusion import TransformerFusionClassifier

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report_generator import generate_report
from src.utils.visualization import plot_confusion_matrix, plot_tsne

LABEL_NAMES = ["Angry", "Happy", "Sad", "Neutral"]


def _encode_batched(encoder, data, device, batch_size=512):
    parts = []
    t = torch.as_tensor(data, dtype=torch.float32)
    for i in range(0, len(t), batch_size):
        parts.append(encoder(t[i : i + batch_size].to(device)).cpu())
    return torch.cat(parts, dim=0)


def _label_aligned_pairs(eeg_emb, eeg_labels, sp_emb, sp_labels, num_classes=4):
    eeg_list, sp_list, lbl_list = [], [], []
    for c in range(num_classes):
        eeg_c = eeg_emb[eeg_labels == c]
        sp_c = sp_emb[sp_labels == c]
        n = min(len(eeg_c), len(sp_c))
        if n == 0:
            continue
        eeg_perm = torch.randperm(len(eeg_c))[:n]
        sp_perm = torch.randperm(len(sp_c))[:n]
        eeg_list.append(eeg_c[eeg_perm])
        sp_list.append(sp_c[sp_perm])
        lbl_list.append(torch.full((n,), c, dtype=torch.long))
    return torch.cat(eeg_list), torch.cat(sp_list), torch.cat(lbl_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="v3: Evaluate full pipeline")
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

    # ── Load data ──
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

    # ── Load encoders (DANN > contrastive > v2) ──
    ecfg = cfg.model.eeg_encoder
    scfg = cfg.model.speech_encoder

    eeg_enc = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)

    encoder_source = "random"
    for name in ["v3/eeg_encoder_dann.pt", "v3/eeg_encoder_contrastive.pt", "eeg/eeg_encoder_final.pt"]:
        if (ckpt / name).exists():
            try:
                eeg_enc.load_state_dict(torch.load(ckpt / name, map_location=device))
                encoder_source = name
                break
            except RuntimeError as e:
                print(f"  SKIP {name}: architecture mismatch (retrain needed)")
                continue
    print(f"EEG encoder: {encoder_source}")
    eeg_enc.eval()

    speech_enc = SpeechEncoder(
        n_features=scfg.n_mfcc,
        embedding_dim=scfg.embedding_dim,
    ).to(device)

    sp_source = "random"
    for name in ["v3/speech_encoder_dann.pt", "v3/speech_encoder_contrastive.pt", "speech/speech_encoder_final.pt"]:
        if (ckpt / name).exists():
            try:
                speech_enc.load_state_dict(torch.load(ckpt / name, map_location=device))
                sp_source = name
                break
            except RuntimeError as e:
                print(f"  SKIP {name}: architecture mismatch (retrain needed)")
                continue
    print(f"Speech encoder: {sp_source}")
    speech_enc.eval()

    # ── Load transformer fusion ──
    v3 = getattr(cfg, "v3", {})
    tf = v3.get("transformer_fusion", {}) if isinstance(v3, dict) else getattr(v3, "transformer_fusion", {})

    n_tokens = tf.get("n_tokens", 8) if isinstance(tf, dict) else getattr(tf, "n_tokens", 8)
    d_model = tf.get("d_model", 64) if isinstance(tf, dict) else getattr(tf, "d_model", 64)
    n_heads = tf.get("n_heads", 4) if isinstance(tf, dict) else getattr(tf, "n_heads", 4)
    n_layers = tf.get("n_layers", 2) if isinstance(tf, dict) else getattr(tf, "n_layers", 2)

    fusion = TransformerFusionClassifier(
        eeg_embed_dim=ecfg.embedding_dim,
        speech_embed_dim=scfg.embedding_dim,
        n_tokens=n_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        num_classes=cfg.model.num_classes,
        dropout=0.0,   # no dropout at eval
        modality_dropout_prob=0.0,
    ).to(device)

    # Try: transformer fusion → RL v3 best → v2 RL best
    # Prefer transformer fusion over RL because RL often degrades performance
    fusion_source = "random"
    for name in ["v3/best_transformer_fusion.pt", "v3/best_fusion_v3.pt", "rl/best_fusion.pt"]:
        if (ckpt / name).exists():
            try:
                sd = torch.load(ckpt / name, map_location=device)
                fusion.load_state_dict(sd.get("fusion", sd))
                fusion_source = name
                break
            except RuntimeError:
                print(f"  SKIP {name}: architecture mismatch (retrain needed)")
                continue
    print(f"Fusion model: {fusion_source}")
    fusion.eval()

    # ── Encode ──
    with torch.no_grad():
        eeg_emb = _encode_batched(eeg_enc, eeg_Xv, device)
        sp_emb = _encode_batched(speech_enc, sp_Xv, device)

    eeg_yv_t = torch.as_tensor(eeg_yv, dtype=torch.long)
    sp_yv_t = torch.as_tensor(sp_yv, dtype=torch.long)

    # ── Label-aligned evaluation ──
    eeg_paired, sp_paired, labels_paired = _label_aligned_pairs(
        eeg_emb, eeg_yv_t, sp_emb, sp_yv_t, num_classes=cfg.model.num_classes,
    )

    print(f"\nv3 Evaluation: {len(labels_paired)} label-aligned pairs")
    dist = Counter(labels_paired.numpy().tolist())
    for c in sorted(dist):
        print(f"  {LABEL_NAMES[c]}: {dist[c]}")

    # ── Predict ──
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
    print("AMERS v3 EVALUATION RESULTS")
    print("=" * 60)
    print(metrics["report_str"])
    print(f"Overall accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"Weighted F1:       {metrics['f1_weighted']:.4f}")
    print(f"Cohen's Kappa:     {metrics['kappa']:.4f}")
    print(f"\nEncoder sources: EEG={encoder_source}, Speech={sp_source}")
    print(f"Fusion source:   {fusion_source}")
    print("=" * 60)

    # ── Plots ──
    plot_confusion_matrix(
        labels, preds, labels=LABEL_NAMES,
        save_path=str(out / "v3_confusion_matrix.png"),
    )

    with torch.no_grad():
        fused = fusion.fusion(
            eeg_paired.to(device), sp_paired.to(device)
        ).cpu().numpy()
    plot_tsne(fused, labels, save_path=str(out / "v3_tsne_embeddings.png"))

    # ── Report ──
    results = {
        "overall_metrics": metrics,
        "version": "v3",
        "encoder_sources": {"eeg": encoder_source, "speech": sp_source},
        "fusion_source": fusion_source,
    }
    generate_report(results, output_dir=out)
    print(f"\nReport saved to {out / 'report.md'}")


if __name__ == "__main__":
    main()
