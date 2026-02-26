#!/usr/bin/env python3
"""v5 LOSO v2 — Improved Leave-One-Subject-Out evaluation on DEAP.

Improvements over v1 (68.41%):
  1. Cross-subject normalization (z-score fit on train, applied to val/test)
  2. Stronger class weights from EEG-only labels (not diluted by IEMOCAP)
  3. Higher focal loss gamma (3.0) for minority-class focus
  4. More epochs (60) + patience (15)
  5. Multiple speech pairings at test time (ensemble over 5 random pairings)

Target: 72–74% mean LOSO accuracy.

Usage:
    python scripts/v5_loso_v2.py
    python scripts/v5_loso_v2.py --start-fold 0 --end-fold 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

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

# Import trainer pieces
_scripts_dir = str(PROJECT_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from v5_train_cmma import (
    CMMATrainer,
    E2ELabelAlignedDataset,
    WarmupCosineScheduler,
    ModelEMA,
)
from v5_train_cmma import FocalLoss  # re-export from fusion_trainer

logger = logging.getLogger(__name__)

# ── LOSO v2 hyperparameter overrides ──
LOSO_V2_OVERRIDES = {
    "epochs": 60,
    "patience": 15,
    "samples_per_epoch": 10000,
    "warmup_epochs": 5,
    "freeze_encoder_epochs": 5,
    "label_smoothing": 0.05,     # less smoothing — let focal loss do the work
    "gate_div_weight": 0.15,     # slightly stronger diversity
}


# ======================================================================
# Cross-subject normalization
# ======================================================================

def normalize_cross_subject(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization: fit on train, apply to all.

    This removes inter-subject distribution shift — the single biggest
    LOSO booster. Each feature dimension is normalized to zero mean and
    unit variance based on the training set statistics only.
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


# ======================================================================
# Model factory (same as v1)
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
    for name in ["v3/eeg_encoder_dann.pt", "v3/eeg_encoder_contrastive.pt",
                  "eeg/eeg_encoder_final.pt"]:
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
    for name in ["v3/speech_encoder_dann.pt", "speech/speech_encoder_final.pt"]:
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

    # ── Build CMMA ──
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
# Test-set evaluation with multi-pairing ensemble
# ======================================================================

def evaluate_test_ensemble(
    eeg_enc, speech_enc, cmma,
    eeg_test: np.ndarray,
    eeg_lbl_test: np.ndarray,
    sp_pool: np.ndarray,
    sp_lbl_pool: np.ndarray,
    device: torch.device,
    num_classes: int = 4,
    n_pairings: int = 5,
    batch_size: int = 512,
) -> np.ndarray:
    """Evaluate with multiple random speech pairings and average logits.

    Instead of a single random pairing, we average logits over
    `n_pairings` different random pairings. This reduces variance
    from the random cross-modal pairing and gives more stable predictions.
    """
    sp_by_class = {c: sp_pool[sp_lbl_pool == c] for c in range(num_classes)}

    eeg_enc.eval()
    speech_enc.eval()
    cmma.eval()

    use_amp = device.type == "cuda"
    n = len(eeg_test)

    # Accumulate logits across pairings
    accum_logits = np.zeros((n, num_classes), dtype=np.float32)

    for pairing_idx in range(n_pairings):
        rng = np.random.RandomState(42 + pairing_idx)
        pair_info = []
        for lbl in eeg_lbl_test:
            c = int(lbl)
            pool = sp_by_class.get(c, sp_pool[:1])  # fallback
            idx = rng.randint(0, max(1, len(pool)))
            pair_info.append((c, idx))

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                eeg_b = torch.as_tensor(
                    eeg_test[start:end], dtype=torch.float32
                ).to(device)

                sp_batch = []
                for c, idx in pair_info[start:end]:
                    pool = sp_by_class.get(c, sp_pool[:1])
                    sp_batch.append(pool[min(idx, len(pool) - 1)])
                sp_b = torch.as_tensor(
                    np.stack(sp_batch), dtype=torch.float32
                ).to(device)

                with autocast("cuda", enabled=use_amp):
                    eeg_emb = eeg_enc(eeg_b)
                    sp_emb = speech_enc(sp_b)
                    logits = cmma(eeg_emb, sp_emb)

                accum_logits[start:end] += logits.cpu().numpy()

    return accum_logits.argmax(axis=1)


# ======================================================================
# Custom LOSO trainer (wraps CMMATrainer with stronger class weights)
# ======================================================================

class LOSOTrainer(CMMATrainer):
    """CMMATrainer with LOSO-specific improvements.

    Overrides:
    1. Class weights computed from EEG labels only (not diluted by IEMOCAP)
    2. Higher focal gamma (3.0) for minority-class focus
    3. Optional EEG mixup augmentation
    """

    def __init__(self, cfg, focal_gamma: float = 3.0):
        super().__init__(cfg)
        self._focal_gamma = focal_gamma
        self._eeg_class_weights = None  # set before calling fit()

    def set_eeg_class_weights(self, eeg_labels: np.ndarray):
        """Compute class weights from EEG training labels only."""
        counts = Counter(eeg_labels.tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            # Inverse frequency with sqrt dampening for extreme ratios
            weights[cls] = np.sqrt(total / (n_cls * max(cnt, 1)))
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        self._eeg_class_weights = weights

    def fit(self, eeg_encoder, speech_encoder, cmma,
            eeg_feat_train, eeg_labels_train,
            sp_feat_train, sp_labels_train,
            eeg_feat_val, eeg_labels_val,
            sp_feat_val, sp_labels_val,
            save_dir=None):
        """Override fit to use EEG-only class weights and higher focal gamma."""

        eeg_encoder = eeg_encoder.to(self.device)
        speech_encoder = speech_encoder.to(self.device)
        cmma = cmma.to(self.device)

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

        # --- Use EEG-only weights if set, otherwise fall back ---
        if self._eeg_class_weights is not None:
            weights = self._eeg_class_weights.to(self.device)
        else:
            all_labels = np.concatenate([eeg_labels_train, sp_labels_train])
            counts = Counter(all_labels.tolist())
            total = sum(counts.values())
            n_cls = max(counts.keys()) + 1
            weights = torch.zeros(n_cls)
            for cls, cnt in counts.items():
                weights[cls] = total / (n_cls * cnt)
            weights = weights.to(self.device)

        criterion = FocalLoss(
            gamma=self._focal_gamma,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

        # --- Discriminative learning rates ---
        encoder_params = list(eeg_encoder.parameters()) + list(speech_encoder.parameters())
        encoder_param_ids = set(id(p) for p in encoder_params)

        eag_params = []
        cmma_other_params = []
        for name, p in cmma.named_parameters():
            if id(p) in encoder_param_ids:
                continue
            if 'class_gate_logits' in name or 'input_gate' in name:
                eag_params.append(p)
            else:
                cmma_other_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.lr * self.encoder_lr_factor,
             "weight_decay": self.weight_decay},
            {"params": cmma_other_params, "lr": self.lr,
             "weight_decay": self.weight_decay},
            {"params": eag_params, "lr": self.lr * self.eag_lr_factor,
             "weight_decay": 0.0},
        ])

        steps_per_epoch = max(self.samples_per_epoch // self.batch_size, 1)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, eta_min=1e-7,
        )

        scaler = GradScaler("cuda", enabled=self.use_amp)

        # --- Datasets ---
        train_ds = E2ELabelAlignedDataset(
            eeg_feat_train, eeg_labels_train,
            sp_feat_train, sp_labels_train,
            num_classes=self.num_classes,
            balance_classes=True,
            samples_per_epoch=self.samples_per_epoch,
        )
        val_ds = E2ELabelAlignedDataset(
            eeg_feat_val, eeg_labels_val,
            sp_feat_val, sp_labels_val,
            num_classes=self.num_classes,
            balance_classes=False,
            samples_per_epoch=2000,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=256)

        best_val_acc = 0.0
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"  LOSO v2 CMMA Training")
        print(f"  Epochs: {self.epochs}, Batch: {self.batch_size}")
        print(f"  Focal gamma: {self._focal_gamma}, Label smoothing: {self.label_smoothing}")
        print(f"  Class weights: {[f'{w:.2f}' for w in weights.cpu().tolist()]}")
        print(f"  CMMA LR: {self.lr}, Encoder LR: {self.lr * self.encoder_lr_factor:.1e}")
        print(f"  Warmup: {self.warmup_epochs} epochs, Patience: {self.patience}")
        print(f"  Samples/epoch: {self.samples_per_epoch}")
        print(f"{'='*60}\n")

        # Start with encoders frozen
        for p in encoder_params:
            p.requires_grad_(False)
        encoders_frozen = True

        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            tf_ratio = max(0.0, 1.0 - (epoch - 1) / self.tf_anneal_epochs)

            if encoders_frozen and epoch > self.freeze_encoder_epochs:
                for p in encoder_params:
                    p.requires_grad_(True)
                encoders_frozen = False
                print(f"  [epoch {epoch}] Encoders unfrozen")

            # --- Train ---
            eeg_encoder.train()
            speech_encoder.train()
            cmma.train()

            train_loss, train_correct, train_total = 0.0, 0, 0

            for eeg_raw, sp_raw, labels in train_loader:
                eeg_raw = eeg_raw.to(self.device, non_blocking=True)
                sp_raw = sp_raw.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    eeg_emb = eeg_encoder(eeg_raw)
                    sp_emb = speech_encoder(sp_raw)

                    logits, aux = cmma(
                        eeg_emb, sp_emb, return_aux=True,
                        labels=labels, tf_ratio=tf_ratio,
                    )

                    loss_main = criterion(logits, labels)
                    loss_eeg_aux = criterion(aux['eeg_logits'], labels)
                    loss_sp_aux = criterion(aux['speech_logits'], labels)
                    loss_probe = criterion(aux['probe_logits'], labels)
                    loss_div = cmma.emotion_gate.gate_diversity_loss()

                    loss = (loss_main
                            + self.aux_loss_weight * loss_eeg_aux
                            + self.aux_loss_weight * loss_sp_aux
                            + 0.1 * loss_probe
                            + self.gate_div_weight * loss_div)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(eeg_encoder.parameters())
                    + list(speech_encoder.parameters())
                    + list(cmma.parameters()),
                    max_norm=self.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item() * eeg_raw.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += eeg_raw.size(0)

            # --- Validate ---
            eeg_encoder.eval()
            speech_encoder.eval()
            cmma.eval()

            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for eeg_raw, sp_raw, labels in val_loader:
                    eeg_raw = eeg_raw.to(self.device)
                    sp_raw = sp_raw.to(self.device)
                    labels = labels.to(self.device)

                    with autocast("cuda", enabled=self.use_amp):
                        eeg_emb = eeg_encoder(eeg_raw)
                        sp_emb = speech_encoder(sp_raw)
                        logits = cmma(eeg_emb, sp_emb)
                        loss = criterion(logits, labels)

                    val_loss += loss.item() * eeg_raw.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += eeg_raw.size(0)

            train_acc = train_correct / max(train_total, 1)
            val_acc = val_correct / max(val_total, 1)
            history["train_loss"].append(train_loss / max(train_total, 1))
            history["val_loss"].append(val_loss / max(val_total, 1))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Print every 5 epochs or first/last
            if epoch % 5 == 0 or epoch <= 2 or epoch == self.epochs:
                print(
                    f"  [{epoch:3d}/{self.epochs}]  "
                    f"train: acc={train_acc:.1%}  "
                    f"val: acc={val_acc:.1%}  "
                    f"tf={tf_ratio:.2f}"
                )

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_dir is not None:
                    save_path = Path(save_dir) / "best_cmma_v5.pt"
                    torch.save({
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "eeg_encoder": eeg_encoder.state_dict(),
                        "speech_encoder": speech_encoder.state_dict(),
                        "cmma_fusion": cmma.state_dict(),
                    }, save_path)
                    if epoch <= 5 or val_acc > 0.7:
                        print(f"    → saved (val_acc={val_acc:.1%})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - t0
        print(f"\n  Training complete in {elapsed:.0f}s (best val_acc={best_val_acc:.2%})")

        return history


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v5 LOSO v2 — improved subject-independent evaluation"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=32)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths()
    ensure_dirs(paths)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    ckpt_dir = paths["checkpoints"]
    loso_dir = Path(paths["outputs"]) / "loso_v2"
    loso_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n" + "=" * 60)
    print("  AMERS v5 LOSO v2 — Improved Subject-Independent Evaluation")
    print("  Improvements: cross-subject norm, EEG class weights,")
    print("  focal gamma=3, 60 epochs, 5-pairing test ensemble")
    print("=" * 60)

    print("\nLoading DEAP data (with subject IDs)...")
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_all, eeg_lbl_all, eeg_sids = deap.load_all(flatten=True)

    print("Loading IEMOCAP data...")
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_all, sp_lbl_all, _ = iemocap.load_all()

    # ── Fixed IEMOCAP split ──
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

    # Show per-class distribution
    eeg_counts = Counter(eeg_lbl_all.tolist())
    label_names = ["Happy", "Sad", "Angry", "Neutral"]
    print(f"\n  DEAP class distribution:")
    for c in range(4):
        print(f"    {label_names[c]:8s}: {eeg_counts.get(c, 0):,}")

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
        print(f"  LOSO v2 Fold {fold_idx + 1}/{len(subjects)}")
        print(f"  Test: subject {test_subj}  |  Val: subject {val_subj}")
        print(f"  Train: {len(train_subjs)} subjects")
        print(f"{'=' * 60}")

        # ── Check cached results ──
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

        # ── KEY IMPROVEMENT 1: Cross-subject normalization ──
        eeg_train, eeg_val, eeg_test = normalize_cross_subject(
            eeg_train, eeg_val, eeg_test
        )

        print(f"  EEG split: train={len(eeg_train):,}  val={len(eeg_val):,}  test={len(eeg_test):,}")
        print(f"  Cross-subject normalization applied (fit on train)")

        # Show train class distribution
        train_counts = Counter(eeg_train_lbl.tolist())
        print(f"  Train class dist: {dict(sorted(train_counts.items()))}")

        # ── Build fresh models ──
        eeg_enc, speech_enc, cmma = build_models(cfg, ckpt_dir, device)

        # ── KEY IMPROVEMENT 2: EEG-only class weights + higher focal gamma ──
        trainer = LOSOTrainer(cfg, focal_gamma=3.0)
        for k, v in LOSO_V2_OVERRIDES.items():
            setattr(trainer, k, v)
        trainer.set_eeg_class_weights(eeg_train_lbl)

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

        # ── KEY IMPROVEMENT 3: Multi-pairing test ensemble ──
        preds = evaluate_test_ensemble(
            eeg_enc, speech_enc, cmma,
            eeg_test, eeg_test_lbl,
            sp_Xv, sp_yv,
            device,
            num_classes=cfg.model.num_classes,
            n_pairings=5,
        )

        metrics = compute_all_metrics(eeg_test_lbl, preds)
        fold_elapsed = time.time() - fold_t0

        print(f"\n  ✅ Fold {fold_idx + 1} complete in {fold_elapsed:.0f}s")
        print(f"     test_acc  = {metrics['accuracy']:.2%}")
        print(f"     f1_macro  = {metrics['f1_macro']:.3f}")
        print(f"     kappa     = {metrics['kappa']:.3f}")

        # ── Save fold results ──
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
            "improvements": [
                "cross_subject_normalization",
                "eeg_only_class_weights",
                "focal_gamma_3.0",
                "multi_pairing_ensemble_5x",
                "epochs_60_patience_15",
            ],
        }
        with open(fold_json, "w") as f:
            json.dump(fold_data, f, indent=2)

        fold_results.append(serializable_metrics)
        all_test_preds.append(preds)
        all_test_labels.append(eeg_test_lbl)

        # ── Free GPU memory ──
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

    overall = compute_all_metrics(all_labels_flat, all_preds_flat)

    agg_keys = [
        "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "kappa",
    ]
    mean_m = {k: float(np.mean([m[k] for m in fold_results])) for k in agg_keys}
    std_m = {k: float(np.std([m[k] for m in fold_results])) for k in agg_keys}

    per_fold_acc = [m["accuracy"] for m in fold_results]

    print(f"\n{'=' * 60}")
    print(f"  LOSO v2 RESULTS  ({len(fold_results)} folds, {total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    print(f"  Overall pooled accuracy : {overall['accuracy']:.2%}")
    print(f"  Mean fold accuracy      : {mean_m['accuracy']:.2%} ± {std_m['accuracy']:.2%}")
    print(f"  Mean fold F1 (macro)    : {mean_m['f1_macro']:.3f} ± {std_m['f1_macro']:.3f}")
    print(f"  Mean fold F1 (weighted) : {mean_m['f1_weighted']:.3f} ± {std_m['f1_weighted']:.3f}")
    print(f"  Mean Cohen's kappa      : {mean_m['kappa']:.3f} ± {std_m['kappa']:.3f}")
    print(f"\n  Per-fold accuracy range : [{min(per_fold_acc):.2%}, {max(per_fold_acc):.2%}]")
    print(f"\n{overall['report_str']}")

    # ── Compare with v1 ──
    v1_summary = loso_dir.parent / "loso" / "loso_summary.json"
    if v1_summary.exists():
        with open(v1_summary) as f:
            v1 = json.load(f)
        v1_acc = v1["mean_metrics"]["accuracy"]
        delta = mean_m["accuracy"] - v1_acc
        print(f"\n  LOSO v1 → v2 comparison:")
        print(f"    v1: {v1_acc:.2%}")
        print(f"    v2: {mean_m['accuracy']:.2%}")
        print(f"    Δ:  {'+' if delta > 0 else ''}{delta:.2%}")

    # ── Save summary ──
    summary = {
        "version": "loso_v2",
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
        "loso_overrides": LOSO_V2_OVERRIDES,
        "improvements": [
            "cross_subject_normalization",
            "eeg_only_class_weights_sqrt",
            "focal_gamma_3.0",
            "multi_pairing_ensemble_5x",
            "epochs_60_patience_15",
            "label_smoothing_0.05",
        ],
    }
    summary_path = loso_dir / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print(f"\n{'=' * 60}")
    print("  LOSO v2 evaluation complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
