#!/usr/bin/env python3
"""v5.6 — Train CMMA (Cross-Modal Mutual Attention) fusion end-to-end.

What's new in v5.6
──────────────────
1. End-to-end training: encoders fine-tuned jointly with CMMA layers
2. Discriminative LR: encoders get 0.05x LR to preserve representations
3. Cross-modal mutual attention with gated cross-attention
4. Emotion-aware gating: per-class modality weights via annealed TF
5. Best-of-two EMA: validate both raw & EMA models, save whichever wins
6. Deterministic validation: fixed pre-computed pairs for stable eval
7. Gate diversity loss: prevents modality weight collapse

Usage:
    python scripts/v5_train_cmma.py
    python scripts/v5_train_cmma.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

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
from src.models.cmma_fusion import CMMAFusionClassifier
from src.training.fusion_trainer import LabelAlignedDataset, FocalLoss
from src.utils.visualization import plot_loss_curves, plot_accuracy_curves


# ======================================================================
# Raw-data dataset for end-to-end training
# ======================================================================

class E2ELabelAlignedDataset(Dataset):
    """Label-aligned dataset that returns RAW features (not embeddings).

    For each sample, picks one EEG and one speech sample with the same
    emotion label.  Balances classes by oversampling minority classes.
    Uses random pairing each call (for training — new pairs each epoch).
    """

    def __init__(
        self,
        eeg_features: np.ndarray,
        eeg_labels: np.ndarray,
        speech_features: np.ndarray,
        speech_labels: np.ndarray,
        num_classes: int = 4,
        balance_classes: bool = True,
        samples_per_epoch: int = 5000,
    ):
        self.num_classes = num_classes
        self.samples_per_epoch = samples_per_epoch

        # Group by class
        self.eeg_by_class = {}
        self.sp_by_class = {}
        for c in range(num_classes):
            self.eeg_by_class[c] = torch.as_tensor(
                eeg_features[eeg_labels == c], dtype=torch.float32
            )
            self.sp_by_class[c] = torch.as_tensor(
                speech_features[speech_labels == c], dtype=torch.float32
            )

        # Compute class weights for balanced sampling
        if balance_classes:
            min_n = min(
                min(len(self.eeg_by_class[c]) for c in range(num_classes)),
                min(len(self.sp_by_class[c]) for c in range(num_classes)),
            )
            self.class_weights = [1.0] * num_classes
        else:
            self.class_weights = [
                min(len(self.eeg_by_class[c]), len(self.sp_by_class[c]))
                for c in range(num_classes)
            ]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Pick a random class (balanced)
        c = idx % self.num_classes

        eeg_pool = self.eeg_by_class[c]
        sp_pool = self.sp_by_class[c]

        eeg_idx = torch.randint(len(eeg_pool), (1,)).item()
        sp_idx = torch.randint(len(sp_pool), (1,)).item()

        return eeg_pool[eeg_idx], sp_pool[sp_idx], c


class FixedPairValDataset(Dataset):
    """Deterministic validation dataset with pre-computed fixed pairs.

    v5.6 fix: random pairing in validation caused noisy accuracy and
    made raw-vs-EMA comparison unreliable (different data each pass).
    This dataset generates all pairs once with a fixed seed.
    """

    def __init__(
        self,
        eeg_features: np.ndarray,
        eeg_labels: np.ndarray,
        speech_features: np.ndarray,
        speech_labels: np.ndarray,
        num_classes: int = 4,
        samples: int = 2000,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        per_class = samples // num_classes

        eeg_list, sp_list, lbl_list = [], [], []
        for c in range(num_classes):
            eeg_pool = eeg_features[eeg_labels == c]
            sp_pool = speech_features[speech_labels == c]
            eeg_idxs = rng.randint(0, len(eeg_pool), size=per_class)
            sp_idxs = rng.randint(0, len(sp_pool), size=per_class)
            eeg_list.append(torch.as_tensor(eeg_pool[eeg_idxs], dtype=torch.float32))
            sp_list.append(torch.as_tensor(sp_pool[sp_idxs], dtype=torch.float32))
            lbl_list.append(torch.full((per_class,), c, dtype=torch.long))

        self.eeg = torch.cat(eeg_list, dim=0)
        self.sp = torch.cat(sp_list, dim=0)
        self.labels = torch.cat(lbl_list, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.sp[idx], self.labels[idx].item()


# ======================================================================
# Learning rate warmup + cosine decay scheduler
# ======================================================================

class WarmupCosineScheduler:
    """Linear warmup for `warmup_steps`, then cosine decay to `eta_min`."""

    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.eta_min, base_lr * scale)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ======================================================================
# Exponential Moving Average (v5.4)
# ======================================================================

class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies updated as:
        shadow = decay * shadow + (1 - decay) * current
    with warmup: actual_decay = min(decay, (1+steps)/(10+steps)).
    """

    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(d).add_(v, alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# ======================================================================
# V5 Trainer
# ======================================================================

class CMMATrainer:
    """End-to-end trainer for v5 CMMA fusion.

    Key features:
    - Discriminative LR: encoder params get cfg.lr * 0.1
    - Focal loss with label smoothing
    - AMP mixed precision
    - Warmup + cosine decay
    - Early stopping with best model save
    - Gradient clipping
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"
        self.num_classes = cfg.model.num_classes

        v5 = getattr(cfg, "v5", {})
        if isinstance(v5, dict):
            _g = lambda k, default: v5.get(k, default)
        else:
            _g = lambda k, default: getattr(v5, k, default)

        self.epochs = _g("epochs", 80)
        self.batch_size = _g("batch_size", 64)
        self.lr = _g("lr", 3e-4)
        self.encoder_lr_factor = _g("encoder_lr_factor", 0.05)  # v5.1: more conservative
        self.eag_lr_factor = _g("eag_lr_factor", 3.0)  # v5.1: higher LR for EAG
        self.patience = _g("patience", 20)
        self.warmup_epochs = _g("warmup_epochs", 5)
        self.freeze_encoder_epochs = _g("freeze_encoder_epochs", 8)  # v5.1: freeze encoders initially
        self.weight_decay = _g("weight_decay", 3e-4)  # v5.2: stronger regularization
        self.samples_per_epoch = _g("samples_per_epoch", 10000)  # v5.1: more samples
        self.grad_clip = _g("grad_clip", 1.0)
        self.aux_loss_weight = _g("aux_loss_weight", 0.2)  # v5.1: auxiliary unimodal loss weight
        self.tf_anneal_epochs = _g("tf_anneal_epochs", 25)  # v5.3: anneal TF from 1→0
        self.gate_div_weight = _g("gate_div_weight", 0.1)  # v5.3: gate diversity loss
        self.ema_decay = _g("ema_decay", 0.999)  # v5.5: EMA (best-of-two validation)
        self.label_smoothing = _g("label_smoothing", 0.15)  # v5.6: increased from 0.1

    def fit(
        self,
        eeg_encoder: EEGEncoder,
        speech_encoder: SpeechEncoder,
        cmma: CMMAFusionClassifier,
        eeg_feat_train: np.ndarray,
        eeg_labels_train: np.ndarray,
        sp_feat_train: np.ndarray,
        sp_labels_train: np.ndarray,
        eeg_feat_val: np.ndarray,
        eeg_labels_val: np.ndarray,
        sp_feat_val: np.ndarray,
        sp_labels_val: np.ndarray,
        save_dir: str | Path | None = None,
    ) -> dict[str, list]:
        """Train end-to-end: encoders + CMMA + classifier."""

        eeg_encoder = eeg_encoder.to(self.device)
        speech_encoder = speech_encoder.to(self.device)
        cmma = cmma.to(self.device)

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

        # --- Class weights for focal loss ---
        all_labels = np.concatenate([eeg_labels_train, sp_labels_train])
        counts = Counter(all_labels.tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            weights[cls] = total / (n_cls * cnt)

        criterion = FocalLoss(
            gamma=2.0, weight=weights.to(self.device),
            label_smoothing=self.label_smoothing,
        )

        # --- Discriminative learning rates (3 groups) ---
        encoder_params = list(eeg_encoder.parameters()) + list(speech_encoder.parameters())
        encoder_param_ids = set(id(p) for p in encoder_params)

        # Separate EAG gate logits (need higher LR) from rest of CMMA
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
             "weight_decay": 0.0},  # no weight decay on gate params
        ])

        # --- Scheduler ---
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
        # v5.6: Deterministic validation — fixed pairs, same data every epoch
        val_ds = FixedPairValDataset(
            eeg_feat_val, eeg_labels_val,
            sp_feat_val, sp_labels_val,
            num_classes=self.num_classes,
            samples=2000,
            seed=42,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=256)

        best_val_acc = 0.0
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"  v5.6 CMMA End-to-End Training (deterministic val + EMA)")
        print(f"  Epochs: {self.epochs}, Batch: {self.batch_size}")
        print(f"  CMMA LR: {self.lr}, Encoder LR: {self.lr * self.encoder_lr_factor}")
        print(f"  EAG LR: {self.lr * self.eag_lr_factor}")
        print(f"  Freeze encoders: first {self.freeze_encoder_epochs} epochs")
        print(f"  Warmup: {self.warmup_epochs} epochs, Patience: {self.patience}")
        print(f"  TF anneal: 1.0 → 0.0 over {self.tf_anneal_epochs} epochs")
        print(f"  EMA decay: {self.ema_decay}, Label smoothing: {self.label_smoothing}")
        print(f"  Gate diversity weight: {self.gate_div_weight}")
        print(f"  Aux loss weight: {self.aux_loss_weight}")
        print(f"  Samples/epoch: {self.samples_per_epoch}")
        print(f"  Validation: deterministic fixed pairs (2000 samples)")
        print(f"{'='*60}\n")

        # Start with encoders frozen
        for p in encoder_params:
            p.requires_grad_(False)
        encoders_frozen = True

        t0 = time.time()

        # v5.4: Initialize EMA shadow models
        if self.ema_decay > 0:
            ema_eeg = ModelEMA(eeg_encoder, decay=self.ema_decay)
            ema_sp = ModelEMA(speech_encoder, decay=self.ema_decay)
            ema_cmma = ModelEMA(cmma, decay=self.ema_decay)

        for epoch in range(1, self.epochs + 1):
            # --- Compute annealed teacher forcing ratio ---
            tf_ratio = max(0.0, 1.0 - (epoch - 1) / self.tf_anneal_epochs)

            # --- Unfreeze encoders after warmup phase ---
            if encoders_frozen and epoch > self.freeze_encoder_epochs:
                for p in encoder_params:
                    p.requires_grad_(True)
                encoders_frozen = False
                print(f"  [epoch {epoch}] Encoders unfrozen (lr={self.lr * self.encoder_lr_factor:.1e})")

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
                    # End-to-end: raw → encode → CMMA → classify
                    eeg_emb = eeg_encoder(eeg_raw)
                    sp_emb = speech_encoder(sp_raw)

                    logits, aux = cmma(
                        eeg_emb, sp_emb, return_aux=True,
                        labels=labels, tf_ratio=tf_ratio,
                    )

                    # Main classification loss
                    loss_main = criterion(logits, labels)

                    # Auxiliary unimodal losses
                    loss_eeg_aux = criterion(aux['eeg_logits'], labels)
                    loss_sp_aux = criterion(aux['speech_logits'], labels)
                    loss_probe = criterion(aux['probe_logits'], labels)

                    # Gate diversity loss (prevents gate logits from collapsing)
                    loss_div = cmma.emotion_gate.gate_diversity_loss()

                    # Total loss (v5.6: removed confidence penalty — it was backwards)
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

                # v5.4: Update EMA
                if self.ema_decay > 0:
                    ema_eeg.update(eeg_encoder)
                    ema_sp.update(speech_encoder)
                    ema_cmma.update(cmma)

                train_loss += loss.item() * eeg_raw.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += eeg_raw.size(0)

            # --- Validate: cache batches for fair raw-vs-EMA comparison ---
            eeg_encoder.eval()
            speech_encoder.eval()
            cmma.eval()

            # v5.6: Collect all val batches once so raw and EMA see identical data
            val_batches = [
                (eeg_raw.to(self.device), sp_raw.to(self.device), labels.to(self.device))
                for eeg_raw, sp_raw, labels in val_loader
            ]

            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for eeg_raw, sp_raw, labels in val_batches:

                    with autocast("cuda", enabled=self.use_amp):
                        eeg_emb = eeg_encoder(eeg_raw)
                        sp_emb = speech_encoder(sp_raw)
                        logits = cmma(eeg_emb, sp_emb)
                        loss = criterion(logits, labels)

                    val_loss += loss.item() * eeg_raw.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += eeg_raw.size(0)

            raw_v_acc = val_correct / max(val_total, 1)
            raw_v_loss = val_loss / max(val_total, 1)

            # --- v5.5: Validate EMA model (best-of-two) ---
            ema_v_acc = 0.0
            use_ema = False
            if self.ema_decay > 0:
                _bk_eeg = {k: v.clone() for k, v in eeg_encoder.state_dict().items()}
                _bk_sp = {k: v.clone() for k, v in speech_encoder.state_dict().items()}
                _bk_cmma = {k: v.clone() for k, v in cmma.state_dict().items()}
                eeg_encoder.load_state_dict(ema_eeg.state_dict())
                speech_encoder.load_state_dict(ema_sp.state_dict())
                cmma.load_state_dict(ema_cmma.state_dict())

                ema_val_correct, ema_val_total = 0, 0
                with torch.no_grad():
                    for eeg_raw, sp_raw, labels in val_batches:  # same cached batches
                        with autocast("cuda", enabled=self.use_amp):
                            eeg_emb = eeg_encoder(eeg_raw)
                            sp_emb = speech_encoder(sp_raw)
                            logits = cmma(eeg_emb, sp_emb)
                        ema_val_correct += (logits.argmax(1) == labels).sum().item()
                        ema_val_total += eeg_raw.size(0)

                ema_v_acc = ema_val_correct / max(ema_val_total, 1)
                use_ema = ema_v_acc > raw_v_acc

                # Restore raw weights for training
                eeg_encoder.load_state_dict(_bk_eeg)
                speech_encoder.load_state_dict(_bk_sp)
                cmma.load_state_dict(_bk_cmma)

            # Pick the better model's accuracy for this epoch
            v_acc = max(raw_v_acc, ema_v_acc)
            v_loss = raw_v_loss  # always use raw loss for monitoring

            t_loss = train_loss / max(train_total, 1)
            t_acc = train_correct / max(train_total, 1)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            lrs = scheduler.get_lr()
            lr_str = f"enc_lr={lrs[0]:.1e} cmma_lr={lrs[1]:.1e} eag_lr={lrs[2]:.1e}"

            ema_tag = " [EMA]" if use_ema else ""
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                print(
                    f"  [{epoch:3d}/{self.epochs}]  "
                    f"train: loss={t_loss:.4f} acc={t_acc:.1%}  "
                    f"val: loss={v_loss:.4f} acc={v_acc:.1%}{ema_tag}  "
                    f"tf={tf_ratio:.2f}  {lr_str}"
                )

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience_counter = 0
                if save_dir:
                    sd = Path(save_dir)
                    sd.mkdir(parents=True, exist_ok=True)
                    # v5.5: Save whichever model (raw or EMA) scored higher
                    if use_ema:
                        save_eeg_sd = ema_eeg.state_dict()
                        save_sp_sd = ema_sp.state_dict()
                        save_cmma_sd = ema_cmma.state_dict()
                    else:
                        save_eeg_sd = eeg_encoder.state_dict()
                        save_sp_sd = speech_encoder.state_dict()
                        save_cmma_sd = cmma.state_dict()
                    torch.save({
                        "eeg_encoder": save_eeg_sd,
                        "speech_encoder": save_sp_sd,
                        "cmma_fusion": save_cmma_sd,
                        "val_acc": v_acc,
                        "epoch": epoch,
                        "ema": use_ema,
                    }, sd / "best_cmma_v5.pt")
                    # Show modality weights from saved model
                    # Temporarily load saved weights to read gate logits
                    _orig_cmma = {k: v.clone() for k, v in cmma.state_dict().items()}
                    cmma.load_state_dict(save_cmma_sd)
                    mw = cmma.get_modality_weights()
                    cmma.load_state_dict(_orig_cmma)  # restore
                    src = "EMA" if use_ema else "raw"
                    print(f"    → saved {src} (val_acc={v_acc:.1%}) | "
                          f"modality weights: EEG={mw[:, 0].tolist()}, "
                          f"Speech={mw[:, 1].tolist()}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  v5 CMMA training complete in {elapsed:.0f}s")
        print(f"  Best val accuracy: {best_val_acc:.2%}")

        # Print final modality weights
        mw = cmma.get_modality_weights()
        labels = ["Angry", "Happy", "Sad", "Neutral"]
        print(f"\n  Learned modality weights (EEG / Speech):")
        for c in range(min(len(labels), mw.size(0))):
            print(f"    {labels[c]:8s}: EEG={mw[c, 0]:.3f}  Speech={mw[c, 1]:.3f}")

        print(f"{'='*60}")

        return history


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="v5: CMMA end-to-end training")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    # --- Load data ---
    print("Loading data...")
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_feat, eeg_lbl, _ = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_feat, sp_lbl, _ = iemocap.load_all()

    print(f"  DEAP:    {len(eeg_feat)} samples, {eeg_feat.shape[1:]}")
    print(f"  IEMOCAP: {len(sp_feat)} samples, {sp_feat.shape[1:]}")

    # --- Split ---
    eeg_Xt, eeg_Xv, eeg_yt, eeg_yv = train_test_split(
        eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
    )
    sp_Xt, sp_Xv, sp_yt, sp_yv = train_test_split(
        sp_feat, sp_lbl, test_size=0.2, stratify=sp_lbl, random_state=cfg.seed,
    )

    print(f"  Train: {len(eeg_Xt)} EEG, {len(sp_Xt)} speech")
    print(f"  Val:   {len(eeg_Xv)} EEG, {len(sp_Xv)} speech")

    # --- Build encoders (load DANN-pretrained weights) ---
    ckpt = Path(paths["checkpoints"])
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

    # Load DANN-pretrained (try both files — they may be swapped on Drive)
    print("\nLoading pretrained encoders...")
    for name in ["v3/eeg_encoder_dann.pt", "v3/speech_encoder_dann.pt",
                  "v3/eeg_encoder_contrastive.pt", "eeg/eeg_encoder_final.pt"]:
        p = ckpt / name
        if p.exists():
            try:
                eeg_enc.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                print(f"  ✅ EEG encoder  ← {name}")
                break
            except RuntimeError:
                continue
    else:
        print("  ⚠️ No compatible EEG encoder checkpoint — training from scratch")

    for name in ["v3/speech_encoder_dann.pt", "v3/eeg_encoder_dann.pt",
                  "v3/speech_encoder_contrastive.pt", "speech/speech_encoder_final.pt"]:
        p = ckpt / name
        if p.exists():
            try:
                speech_enc.load_state_dict(torch.load(p, map_location=device, weights_only=False))
                print(f"  ✅ Speech enc   ← {name}")
                break
            except RuntimeError:
                continue
    else:
        print("  ⚠️ No compatible speech encoder checkpoint — training from scratch")

    # --- Build CMMA fusion ---
    v5 = getattr(cfg, "v5", {})
    if isinstance(v5, dict):
        _g = lambda k, default: v5.get(k, default)
    else:
        _g = lambda k, default: getattr(v5, k, default)

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

    # Count parameters
    n_eeg = sum(p.numel() for p in eeg_enc.parameters())
    n_sp = sum(p.numel() for p in speech_enc.parameters())
    n_cmma = sum(p.numel() for p in cmma.parameters())
    print(f"\n  Model params: EEG={n_eeg:,}, Speech={n_sp:,}, CMMA={n_cmma:,}")
    print(f"  Total: {n_eeg + n_sp + n_cmma:,}")

    # --- Train ---
    save_dir = ckpt / "v5"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = CMMATrainer(cfg)
    history = trainer.fit(
        eeg_enc, speech_enc, cmma,
        eeg_Xt, eeg_yt,
        sp_Xt, sp_yt,
        eeg_Xv, eeg_yv,
        sp_Xv, sp_yv,
        save_dir=save_dir,
    )

    # --- Plots ---
    out = Path(paths["outputs"])
    plot_loss_curves(
        history["train_loss"], history["val_loss"],
        title="v5 CMMA End-to-End Loss",
        save_path=str(out / "v5_cmma_loss.png"),
    )
    plot_accuracy_curves(
        history["train_acc"], history["val_acc"],
        title="v5 CMMA End-to-End Accuracy",
        save_path=str(out / "v5_cmma_acc.png"),
    )

    print("\nv5 CMMA training complete!")
    print(f"  Checkpoint: {save_dir / 'best_cmma_v5.pt'}")


if __name__ == "__main__":
    main()
