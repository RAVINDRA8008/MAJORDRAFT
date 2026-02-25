#!/usr/bin/env python3
"""v5 LOSO — Leave-One-Subject-Out cross-validation on DEAP.

Evaluates the v5.3 CMMA fusion model's EEG subject-independence by
holding out one DEAP subject per fold (32 folds total).

Per fold:
  - Test:  held-out subject  (completely unseen during training)
  - Val:   next subject      (used only for early stopping)
  - Train: remaining 30 subjects
  - Speech: fixed 80/20 IEMOCAP split (constant across all folds)

Pretrained DANN encoder weights are loaded per fold (slight data leak
acknowledged — encoder pretraining used all subjects).

Uses reduced epochs (40) and patience (10) to keep total runtime
under ~3 hours on an NVIDIA L4 GPU.

Usage:
    python scripts/v5_loso.py
    python scripts/v5_loso.py --start-fold 0 --end-fold 16   # first half
    python scripts/v5_loso.py --start-fold 16 --end-fold 32  # second half
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.amp import autocast

# ── Project root on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging

from src.data.deap_loader import DEAPLoader
from src.data.iemocap_loader import IEMOCAPLoader
from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.cmma_fusion import CMMAFusionClassifier
from src.evaluation.metrics import compute_all_metrics

# Import trainer from the v5 training script (safe — guarded by __name__)
_scripts_dir = str(PROJECT_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from v5_train_cmma import CMMATrainer  # noqa: E402

logger = logging.getLogger(__name__)

# ── LOSO hyperparameter overrides (reduced for speed) ──
LOSO_OVERRIDES = {
    "epochs": 40,
    "patience": 10,
    "samples_per_epoch": 8000,
    "warmup_epochs": 3,
    "freeze_encoder_epochs": 5,
}


# ======================================================================
# Model factory
# ======================================================================

def build_models(cfg, ckpt_dir: str, device: torch.device):
    """Create fresh EEG encoder, speech encoder, CMMA, with pretrained weights."""
    ecfg = cfg.model.eeg_encoder
    scfg = cfg.model.speech_encoder

    eeg_enc = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)

    speech_enc = SpeechEncoder(
        n_features=scfg.n_mfcc,
        embedding_dim=scfg.embedding_dim,
    ).to(device)

    # ── Load DANN-pretrained encoder weights ──
    ckpt = Path(ckpt_dir)

    eeg_loaded = False
    for name in ["v3/eeg_encoder_dann.pt", "v3/speech_encoder_dann.pt",
                  "v3/eeg_encoder_contrastive.pt", "eeg/eeg_encoder_final.pt"]:
        p = ckpt / name
        if p.exists():
            try:
                eeg_enc.load_state_dict(
                    torch.load(p, map_location=device, weights_only=False)
                )
                eeg_loaded = True
                break
            except RuntimeError:
                continue
    if not eeg_loaded:
        logger.warning("No compatible EEG encoder checkpoint — training from scratch")

    sp_loaded = False
    for name in ["v3/speech_encoder_dann.pt", "v3/eeg_encoder_dann.pt",
                  "v3/speech_encoder_contrastive.pt", "speech/speech_encoder_final.pt"]:
        p = ckpt / name
        if p.exists():
            try:
                speech_enc.load_state_dict(
                    torch.load(p, map_location=device, weights_only=False)
                )
                sp_loaded = True
                break
            except RuntimeError:
                continue
    if not sp_loaded:
        logger.warning("No compatible speech encoder checkpoint — training from scratch")

    # ── Build CMMA fusion classifier ──
    v5 = getattr(cfg, "v5", {})
    _g = (lambda k, d: v5.get(k, d)) if isinstance(v5, dict) else (lambda k, d: getattr(v5, k, d))

    cmma = CMMAFusionClassifier(
        eeg_embed_dim=ecfg.embedding_dim,
        speech_embed_dim=scfg.embedding_dim,
        n_tokens=_g("n_tokens", 8),
        d_model=_g("d_model", 128),
        n_heads=_g("n_heads", 4),
        n_cmma_layers=_g("n_cmma_layers", 3),
        ff_dim=_g("ff_dim", 512),
        num_classes=cfg.model.num_classes,
        dropout=_g("dropout", 0.1),
        modality_dropout_prob=_g("modality_dropout", 0.1),
    ).to(device)

    return eeg_enc, speech_enc, cmma


# ======================================================================
# Test-set evaluation (all held-out EEG samples)
# ======================================================================

def evaluate_test(
    eeg_enc, speech_enc, cmma,
    eeg_test: np.ndarray,
    eeg_lbl_test: np.ndarray,
    sp_pool: np.ndarray,
    sp_lbl_pool: np.ndarray,
    device: torch.device,
    num_classes: int = 4,
    seed: int = 42,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate trained model on ALL held-out EEG test samples.

    Each EEG sample is paired with a random IEMOCAP speech sample
    of the same emotion label (label-aligned pairing, consistent
    with training protocol).

    Returns:
        Predicted labels array, shape ``(n_test,)``.
    """
    rng = np.random.RandomState(seed)
    sp_by_class = {c: sp_pool[sp_lbl_pool == c] for c in range(num_classes)}

    # Pre-compute speech pair index for each test sample
    pair_info = []  # list of (class, index-into-class-pool)
    for lbl in eeg_lbl_test:
        c = int(lbl)
        pool = sp_by_class[c]
        idx = rng.randint(0, len(pool))
        pair_info.append((c, idx))

    eeg_enc.eval()
    speech_enc.eval()
    cmma.eval()

    all_preds = []
    n = len(eeg_test)
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            eeg_b = torch.as_tensor(
                eeg_test[start:end], dtype=torch.float32
            ).to(device)

            # Construct speech batch on the fly (memory-efficient)
            sp_batch = [sp_by_class[c][idx] for c, idx in pair_info[start:end]]
            sp_b = torch.as_tensor(
                np.stack(sp_batch), dtype=torch.float32
            ).to(device)

            with autocast("cuda", enabled=use_amp):
                eeg_emb = eeg_enc(eeg_b)
                sp_emb = speech_enc(sp_b)
                logits = cmma(eeg_emb, sp_emb)

            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v5 LOSO — subject-independent evaluation on DEAP"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--start-fold", type=int, default=0,
        help="First fold index (0-based, inclusive)"
    )
    parser.add_argument(
        "--end-fold", type=int, default=32,
        help="Last fold index (exclusive)"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # ── Setup ──
    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths()
    ensure_dirs(paths)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    ckpt_dir = paths["checkpoints"]
    loso_dir = Path(paths["outputs"]) / "loso"
    loso_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n" + "=" * 60)
    print("  AMERS v5 LOSO — Leave-One-Subject-Out Evaluation")
    print("=" * 60)

    print("\nLoading DEAP data (with subject IDs)...")
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_all, eeg_lbl_all, eeg_sids = deap.load_all(flatten=True)

    print("Loading IEMOCAP data...")
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_all, sp_lbl_all, _ = iemocap.load_all()

    # ── Fixed IEMOCAP split (identical across all 32 folds) ──
    sp_Xt, sp_Xv, sp_yt, sp_yv = train_test_split(
        sp_all, sp_lbl_all,
        test_size=0.2,
        stratify=sp_lbl_all,
        random_state=cfg.seed,
    )

    # ── Group DEAP by subject ──
    subjects = sorted(set(eeg_sids.tolist()))
    eeg_by_subj: dict[int, np.ndarray] = {}
    lbl_by_subj: dict[int, np.ndarray] = {}
    for s in subjects:
        mask = eeg_sids == s
        eeg_by_subj[s] = eeg_all[mask]
        lbl_by_subj[s] = eeg_lbl_all[mask]

    print(f"\n  DEAP:    {len(eeg_all):,} samples, {len(subjects)} subjects")
    print(f"  IEMOCAP: {len(sp_all):,} total → {len(sp_Xt):,} train, {len(sp_Xv):,} val")
    for s in subjects:
        print(f"    Subject {s:2d}: {len(eeg_by_subj[s]):,} samples")

    # ── LOSO loop ──
    start_fold = max(0, args.start_fold)
    end_fold = min(args.end_fold, len(subjects))

    fold_results: list[dict] = []
    all_test_preds: list[np.ndarray] = []
    all_test_labels: list[np.ndarray] = []
    total_t0 = time.time()

    for fold_idx in range(start_fold, end_fold):
        test_subj = subjects[fold_idx]
        val_subj = subjects[(fold_idx + 1) % len(subjects)]
        train_subjs = [s for s in subjects if s != test_subj and s != val_subj]

        print(f"\n{'=' * 60}")
        print(f"  LOSO Fold {fold_idx + 1}/{len(subjects)}")
        print(f"  Test: subject {test_subj}  |  Val: subject {val_subj}")
        print(f"  Train: {len(train_subjs)} subjects")
        print(f"{'=' * 60}")

        # ── Check cached results (resume support) ──
        fold_json = loso_dir / f"fold_{fold_idx:02d}.json"
        if fold_json.exists():
            print("  → Cached result found — skipping training")
            with open(fold_json) as f:
                cached = json.load(f)
            fold_results.append(cached["metrics"])
            all_test_preds.append(np.array(cached["predictions"]))
            all_test_labels.append(np.array(cached["true_labels"]))
            continue

        fold_t0 = time.time()

        # ── Build EEG splits ──
        eeg_train = np.concatenate([eeg_by_subj[s] for s in train_subjs])
        eeg_train_lbl = np.concatenate([lbl_by_subj[s] for s in train_subjs])
        eeg_val = eeg_by_subj[val_subj]
        eeg_val_lbl = lbl_by_subj[val_subj]
        eeg_test = eeg_by_subj[test_subj]
        eeg_test_lbl = lbl_by_subj[test_subj]

        print(f"  EEG split: train={len(eeg_train):,}  val={len(eeg_val):,}  test={len(eeg_test):,}")

        # ── Build fresh models (reload pretrained encoders each fold) ──
        eeg_enc, speech_enc, cmma = build_models(cfg, ckpt_dir, device)

        # ── Train with reduced hyperparams ──
        trainer = CMMATrainer(cfg)
        for k, v in LOSO_OVERRIDES.items():
            setattr(trainer, k, v)

        fold_save = loso_dir / f"fold_{fold_idx:02d}"
        fold_save.mkdir(parents=True, exist_ok=True)

        history = trainer.fit(
            eeg_enc, speech_enc, cmma,
            eeg_train, eeg_train_lbl,
            sp_Xt, sp_yt,
            eeg_val, eeg_val_lbl,
            sp_Xv, sp_yv,
            save_dir=str(fold_save),
        )

        # ── Reload best checkpoint ──
        best_ckpt = fold_save / "best_cmma_v5.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device, weights_only=False)
            eeg_enc.load_state_dict(state["eeg_encoder"])
            speech_enc.load_state_dict(state["speech_encoder"])
            cmma.load_state_dict(state["cmma_fusion"])
            saved_val_acc = state.get("val_acc", 0)
            print(f"  Loaded best checkpoint (val_acc={saved_val_acc:.2%})")
        else:
            print("  ⚠ No best checkpoint saved — using last epoch weights")

        # ── Evaluate on held-out test subject ──
        preds = evaluate_test(
            eeg_enc, speech_enc, cmma,
            eeg_test, eeg_test_lbl,
            sp_Xv, sp_yv,   # pair with IEMOCAP val split
            device,
            num_classes=cfg.model.num_classes,
        )

        metrics = compute_all_metrics(eeg_test_lbl, preds)
        fold_elapsed = time.time() - fold_t0

        print(f"\n  ✅ Fold {fold_idx + 1} complete in {fold_elapsed:.0f}s")
        print(f"     test_acc  = {metrics['accuracy']:.2%}")
        print(f"     f1_macro  = {metrics['f1_macro']:.3f}")
        print(f"     kappa     = {metrics['kappa']:.3f}")

        # ── Save fold results (JSON) ──
        serializable_metrics = {
            k: v for k, v in metrics.items()
            if k not in ("confusion_matrix", "report_str")
        }
        fold_data = {
            "fold": fold_idx,
            "test_subject": int(test_subj),
            "val_subject": int(val_subj),
            "n_train": len(eeg_train),
            "n_val": len(eeg_val),
            "n_test": len(eeg_test),
            "metrics": serializable_metrics,
            "predictions": preds.tolist(),
            "true_labels": eeg_test_lbl.tolist(),
            "confusion_matrix": metrics["confusion_matrix"],
            "classification_report": metrics["report_str"],
            "elapsed_seconds": round(fold_elapsed, 1),
        }
        with open(fold_json, "w") as f:
            json.dump(fold_data, f, indent=2)

        fold_results.append(serializable_metrics)
        all_test_preds.append(preds)
        all_test_labels.append(eeg_test_lbl)

        # ── Free GPU memory for next fold ──
        del eeg_enc, speech_enc, cmma, trainer
        torch.cuda.empty_cache()

    # ==================================================================
    # Aggregate results
    # ==================================================================
    total_elapsed = time.time() - total_t0

    if not fold_results:
        print("\nNo folds completed — nothing to aggregate.")
        return

    all_preds_flat = np.concatenate(all_test_preds)
    all_labels_flat = np.concatenate(all_test_labels)

    # Overall metrics (all predictions pooled across folds)
    overall = compute_all_metrics(all_labels_flat, all_preds_flat)

    # Per-fold mean ± std
    agg_keys = [
        "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "kappa",
    ]
    mean_m = {k: float(np.mean([m[k] for m in fold_results])) for k in agg_keys}
    std_m = {k: float(np.std([m[k] for m in fold_results])) for k in agg_keys}

    per_fold_acc = [m["accuracy"] for m in fold_results]

    print(f"\n{'=' * 60}")
    print(f"  LOSO RESULTS  ({len(fold_results)} folds, {total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    print(f"  Overall pooled accuracy : {overall['accuracy']:.2%}")
    print(f"  Mean fold accuracy      : {mean_m['accuracy']:.2%} ± {std_m['accuracy']:.2%}")
    print(f"  Mean fold F1 (macro)    : {mean_m['f1_macro']:.3f} ± {std_m['f1_macro']:.3f}")
    print(f"  Mean fold F1 (weighted) : {mean_m['f1_weighted']:.3f} ± {std_m['f1_weighted']:.3f}")
    print(f"  Mean Cohen's kappa      : {mean_m['kappa']:.3f} ± {std_m['kappa']:.3f}")
    print(f"\n  Per-fold accuracy range : [{min(per_fold_acc):.2%}, {max(per_fold_acc):.2%}]")
    print(f"\n{overall['report_str']}")

    # ── Save summary ──
    summary = {
        "n_folds": len(fold_results),
        "total_time_seconds": round(total_elapsed, 1),
        "overall_metrics": {
            k: v for k, v in overall.items() if k != "report_str"
        },
        "mean_metrics": mean_m,
        "std_metrics": std_m,
        "per_fold_accuracy": per_fold_acc,
        "per_fold_subjects": [
            subjects[i] for i in range(start_fold, end_fold)
        ],
        "classification_report": overall["report_str"],
        "loso_overrides": LOSO_OVERRIDES,
    }
    summary_path = loso_dir / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print(f"  Per-fold JSONs in {loso_dir}/fold_*.json")
    print(f"\n{'=' * 60}")
    print("  LOSO evaluation complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
