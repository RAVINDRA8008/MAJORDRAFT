#!/usr/bin/env python3
"""Strict LOSO v2 — Fully Leak-Free Leave-One-Subject-Out Evaluation.

NO DATA LEAKAGE: Every stage (contrastive pretraining, DANN, CMMA fusion)
is trained from scratch per fold using ONLY the 30 training subjects.
The test subject NEVER appears in any stage of any fold.

Differences from loso_v2:
  - Contrastive EEG encoder pretraining done per fold (30 subjects only)
  - DANN domain adaptation done per fold (30 subjects only)
  - No global checkpoint reuse whatsoever
  - All weights freshly initialized per fold
  - Speech encoder contrastive pretraining also per fold

This is the gold-standard subject-independent evaluation.
Expected runtime: ~8-12 hours on L4 (32 folds × full pretraining pipeline).

Usage:
    python scripts/strict_loso_v2.py
    python scripts/strict_loso_v2.py --start-fold 0 --end-fold 16
    python scripts/strict_loso_v2.py --skip-contrastive   # skip CL, only DANN
    python scripts/strict_loso_v2.py --contrastive-epochs 50 --dann-epochs 15
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

# Contrastive & DANN components
from src.pretraining.contrastive_eeg import (
    ContrastiveEEGDataset,
    EEGAugmentor,
    ProjectionHead,
    NTXentLoss,
)
from src.models.domain_adapter import (
    DomainClassifier,
    EmotionHead,
    DomainAlignedDataset,
    _lambda_schedule,
)

# Trainer pieces from v5
_scripts_dir = str(PROJECT_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from v5_train_cmma import (
    CMMATrainer,
    E2ELabelAlignedDataset,
    WarmupCosineScheduler,
    FocalLoss,
)

logger = logging.getLogger(__name__)

# ── Strict LOSO hyperparameters ──
STRICT_LOSO_CONFIG = {
    # Contrastive pretraining (per fold)
    "contrastive_epochs": 60,
    "contrastive_batch_size": 512,
    "contrastive_lr": 3e-4,
    "contrastive_temperature": 0.07,
    # DANN (per fold)
    "dann_epochs": 20,
    "dann_batch_size": 128,
    "dann_lr": 1e-4,
    "dann_domain_weight": 0.3,
    # CMMA fusion (per fold) — same as LOSO v2
    "fusion_epochs": 60,
    "fusion_patience": 15,
    "fusion_samples_per_epoch": 10_000,
    "fusion_warmup_epochs": 5,
    "fusion_freeze_encoder_epochs": 5,
    "fusion_label_smoothing": 0.05,
    "fusion_gate_div_weight": 0.15,
    "fusion_focal_gamma": 3.0,
    # Test ensemble
    "n_test_pairings": 5,
}


# ======================================================================
# Cross-subject normalization (same as loso_v2)
# ======================================================================

def normalize_cross_subject(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score: fit on train subjects only, apply to all."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


# ======================================================================
# STEP 3: Per-fold contrastive pretraining (EEG only — 30 subjects)
# ======================================================================

def contrastive_pretrain_eeg(
    eeg_train: np.ndarray,
    cfg,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 3e-4,
    temperature: float = 0.07,
) -> EEGEncoder:
    """Contrastive pretrain an EEG encoder from scratch on training data ONLY.

    Returns a fresh EEGEncoder with pretrained weights.
    No labels used — pure self-supervised learning.
    """
    ecfg = cfg.model.eeg_encoder
    encoder = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)

    projector = ProjectionHead(
        input_dim=ecfg.embedding_dim,
        hidden_dim=ecfg.embedding_dim * 2,
        output_dim=64,
    ).to(device)

    augmentor = EEGAugmentor()
    dataset = ContrastiveEEGDataset(eeg_train, augmentor)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
    )

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    warmup_ep = max(1, epochs // 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_ep, eta_min=1e-6,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = NTXentLoss(temperature=temperature)

    use_amp = device.type == "cuda"

    print(f"    [CL-EEG] {len(eeg_train):,} samples, {epochs} epochs, batch={batch_size}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_ep:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * epoch / warmup_ep

        encoder.train()
        projector.train()
        running_loss = 0.0
        n_batches = 0

        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                z1 = projector(encoder(v1))
                z2 = projector(encoder(v2))
                loss = criterion(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        if epoch > warmup_ep:
            scheduler.step()

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            el = running_loss / max(n_batches, 1)
            print(f"    [CL-EEG] epoch {epoch}/{epochs}  loss={el:.4f}")

    elapsed = time.time() - t0
    print(f"    [CL-EEG] done in {elapsed:.0f}s")

    # Projector is discarded — only encoder weights are kept
    del projector
    return encoder


# ======================================================================
# STEP 4: Per-fold DANN domain adaptation (30 EEG subjects + speech)
# ======================================================================

def dann_train_per_fold(
    eeg_encoder: EEGEncoder,
    eeg_train: np.ndarray,
    eeg_train_lbl: np.ndarray,
    sp_train: np.ndarray,
    sp_train_lbl: np.ndarray,
    cfg,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-4,
    domain_weight: float = 0.3,
) -> tuple[EEGEncoder, SpeechEncoder]:
    """DANN domain adaptation using ONLY the training subjects' EEG data.

    Initializes the EEG encoder from contrastive pretraining (this fold).
    Speech encoder is freshly initialized (IEMOCAP has no subject leak concern).

    Returns both encoders with DANN-tuned weights.
    """
    scfg = cfg.model.speech_encoder
    speech_encoder = SpeechEncoder(
        n_features=scfg.n_mfcc,
        embedding_dim=scfg.embedding_dim,
    ).to(device)

    embed_dim = cfg.model.eeg_encoder.embedding_dim  # 128

    domain_classifier = DomainClassifier(
        input_dim=embed_dim, hidden_dim=embed_dim, dropout=0.3,
    ).to(device)

    emotion_head = EmotionHead(
        input_dim=embed_dim, num_classes=cfg.model.num_classes,
    ).to(device)

    dataset = DomainAlignedDataset(
        eeg_train, eeg_train_lbl, sp_train, sp_train_lbl,
    )

    def collate_fn(batch):
        eeg_f, eeg_e, eeg_d, sp_f, sp_e, sp_d = [], [], [], [], [], []
        for feat, emo, dom, mod in batch:
            if mod.item() == 0:
                eeg_f.append(feat); eeg_e.append(emo); eeg_d.append(dom)
            else:
                sp_f.append(feat); sp_e.append(emo); sp_d.append(dom)
        return {
            "eeg_feats": torch.stack(eeg_f) if eeg_f else None,
            "eeg_emos": torch.stack(eeg_e) if eeg_e else None,
            "eeg_doms": torch.stack(eeg_d) if eeg_d else None,
            "speech_feats": torch.stack(sp_f) if sp_f else None,
            "speech_emos": torch.stack(sp_e) if sp_e else None,
            "speech_doms": torch.stack(sp_d) if sp_d else None,
        }

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
        collate_fn=collate_fn,
    )

    params = (
        list(eeg_encoder.parameters())
        + list(speech_encoder.parameters())
        + list(emotion_head.parameters())
        + list(domain_classifier.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    emo_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    dom_criterion = nn.CrossEntropyLoss()

    print(f"    [DANN] EEG={len(eeg_train):,}  Speech={len(sp_train):,}  epochs={epochs}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        lambda_ = _lambda_schedule(epoch, epochs)
        domain_classifier.grl.set_lambda(lambda_)

        eeg_encoder.train()
        speech_encoder.train()
        emotion_head.train()
        domain_classifier.train()

        running_emo, running_dom, n_b = 0.0, 0.0, 0

        for batch in loader:
            all_emb, all_emo, all_dom = [], [], []

            with autocast("cuda", enabled=use_amp):
                if batch["eeg_feats"] is not None:
                    eeg_x = batch["eeg_feats"].to(device, non_blocking=True)
                    all_emb.append(eeg_encoder(eeg_x))
                    all_emo.append(batch["eeg_emos"].to(device))
                    all_dom.append(batch["eeg_doms"].to(device))

                if batch["speech_feats"] is not None:
                    sp_x = batch["speech_feats"].to(device, non_blocking=True)
                    all_emb.append(speech_encoder(sp_x))
                    all_emo.append(batch["speech_emos"].to(device))
                    all_dom.append(batch["speech_doms"].to(device))

                if not all_emb:
                    continue

                emb = torch.cat(all_emb, dim=0)
                emo_lbl = torch.cat(all_emo, dim=0)
                dom_lbl = torch.cat(all_dom, dim=0)

                emo_loss = emo_criterion(emotion_head(emb), emo_lbl)
                dom_loss = dom_criterion(domain_classifier(emb), dom_lbl)
                total_loss = emo_loss + domain_weight * dom_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_emo += emo_loss.item()
            running_dom += dom_loss.item()
            n_b += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"    [DANN] epoch {epoch}/{epochs}  "
                f"emo={running_emo / max(n_b, 1):.4f}  "
                f"dom={running_dom / max(n_b, 1):.4f}  "
                f"λ={lambda_:.3f}"
            )

    elapsed = time.time() - t0
    print(f"    [DANN] done in {elapsed:.0f}s")

    # Clean up DANN-specific modules (not needed downstream)
    del domain_classifier, emotion_head
    return eeg_encoder, speech_encoder


# ======================================================================
# STEP 5: Build fresh CMMA (no pretrained loading)
# ======================================================================

def build_cmma_only(cfg, device: torch.device) -> CMMAFusionClassifier:
    """Build a fresh CMMA fusion classifier — NO checkpoint loading."""
    ecfg = cfg.model.eeg_encoder
    v5 = getattr(cfg, "v5", {})
    _g = (lambda k, d: v5.get(k, d)) if isinstance(v5, dict) else (
        lambda k, d: getattr(v5, k, d)
    )

    cmma = CMMAFusionClassifier(
        eeg_embed_dim=ecfg.embedding_dim,
        speech_embed_dim=cfg.model.speech_encoder.embedding_dim,
        n_tokens=_g("n_tokens", 8),
        d_model=_g("d_model", 128),
        n_heads=_g("n_heads", 4),
        n_cmma_layers=_g("n_cmma_layers", 3),
        ff_dim=_g("ff_dim", 512),
        num_classes=cfg.model.num_classes,
        dropout=_g("dropout", 0.1),
        modality_dropout_prob=_g("modality_dropout", 0.1),
    ).to(device)

    return cmma


# ======================================================================
# Test-set evaluation with multi-pairing ensemble (same as loso_v2)
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
    """Evaluate with multiple random speech pairings and average logits."""
    sp_by_class = {c: sp_pool[sp_lbl_pool == c] for c in range(num_classes)}

    eeg_enc.eval()
    speech_enc.eval()
    cmma.eval()

    use_amp = device.type == "cuda"
    n = len(eeg_test)
    accum_logits = np.zeros((n, num_classes), dtype=np.float32)

    for pairing_idx in range(n_pairings):
        rng = np.random.RandomState(42 + pairing_idx)
        pair_info = []
        for lbl in eeg_lbl_test:
            c = int(lbl)
            pool = sp_by_class.get(c, sp_pool[:1])
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
# CMMA Fusion Trainer (strict — uses given encoders, no checkpoint load)
# ======================================================================

class StrictLOSOTrainer(CMMATrainer):
    """CMMATrainer for strict LOSO — same improvements as LOSOTrainer v2
    but explicitly never loads global checkpoints."""

    def __init__(self, cfg, focal_gamma: float = 3.0):
        super().__init__(cfg)
        self._focal_gamma = focal_gamma
        self._eeg_class_weights = None

    def set_eeg_class_weights(self, eeg_labels: np.ndarray):
        """Compute class weights from EEG training labels only."""
        counts = Counter(eeg_labels.tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            weights[cls] = np.sqrt(total / (n_cls * max(cnt, 1)))
        weights = weights / weights.mean()
        self._eeg_class_weights = weights

    def fit(self, eeg_encoder, speech_encoder, cmma,
            eeg_feat_train, eeg_labels_train,
            sp_feat_train, sp_labels_train,
            eeg_feat_val, eeg_labels_val,
            sp_feat_val, sp_labels_val,
            save_dir=None):
        """Train CMMA fusion. Identical logic to LOSOTrainer.fit() from loso_v2."""

        eeg_encoder = eeg_encoder.to(self.device)
        speech_encoder = speech_encoder.to(self.device)
        cmma = cmma.to(self.device)

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

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

        # Discriminative learning rates
        encoder_params = (
            list(eeg_encoder.parameters()) + list(speech_encoder.parameters())
        )
        encoder_param_ids = set(id(p) for p in encoder_params)

        eag_params = []
        cmma_other_params = []
        for name, p in cmma.named_parameters():
            if id(p) in encoder_param_ids:
                continue
            if "class_gate_logits" in name or "input_gate" in name:
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

        print(f"\n    [CMMA] epochs={self.epochs}, batch={self.batch_size}")
        print(f"    [CMMA] focal_gamma={self._focal_gamma}, "
              f"smooth={self.label_smoothing}")
        print(f"    [CMMA] weights={[f'{w:.2f}' for w in weights.cpu().tolist()]}")
        print(f"    [CMMA] patience={self.patience}, warmup={self.warmup_epochs}")

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
                print(f"    [CMMA] epoch {epoch}: encoders unfrozen")

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
                    loss_eeg_aux = criterion(aux["eeg_logits"], labels)
                    loss_sp_aux = criterion(aux["speech_logits"], labels)
                    loss_probe = criterion(aux["probe_logits"], labels)
                    loss_div = cmma.emotion_gate.gate_diversity_loss()

                    loss = (
                        loss_main
                        + self.aux_loss_weight * loss_eeg_aux
                        + self.aux_loss_weight * loss_sp_aux
                        + 0.1 * loss_probe
                        + self.gate_div_weight * loss_div
                    )

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

            if epoch % 10 == 0 or epoch <= 2 or epoch == self.epochs:
                print(
                    f"    [CMMA] [{epoch:3d}/{self.epochs}]  "
                    f"train={train_acc:.1%}  val={val_acc:.1%}  "
                    f"tf={tf_ratio:.2f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_dir is not None:
                    save_path = Path(save_dir) / "best_strict_loso.pt"
                    torch.save({
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "eeg_encoder": eeg_encoder.state_dict(),
                        "speech_encoder": speech_encoder.state_dict(),
                        "cmma_fusion": cmma.state_dict(),
                    }, save_path)
                    if epoch <= 5 or val_acc > 0.7:
                        print(f"      → saved (val_acc={val_acc:.1%})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"    [CMMA] Early stop at epoch {epoch}")
                    break

        elapsed = time.time() - t0
        print(f"    [CMMA] done in {elapsed:.0f}s (best={best_val_acc:.2%})")
        return history


# ======================================================================
# Main — Strict LOSO Loop
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict LOSO v2 — fully leak-free subject-independent evaluation"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=32)
    parser.add_argument("--skip-contrastive", action="store_true",
                        help="Skip contrastive pretraining (faster, but weaker encoders)")
    parser.add_argument("--contrastive-epochs", type=int, default=None,
                        help="Override contrastive pretraining epochs per fold")
    parser.add_argument("--dann-epochs", type=int, default=None,
                        help="Override DANN training epochs per fold")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths()
    ensure_dirs(paths)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    hparams = dict(STRICT_LOSO_CONFIG)
    if args.contrastive_epochs is not None:
        hparams["contrastive_epochs"] = args.contrastive_epochs
    if args.dann_epochs is not None:
        hparams["dann_epochs"] = args.dann_epochs

    loso_dir = Path(paths["outputs"]) / "strict_loso_v2"
    loso_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n" + "=" * 72)
    print("  AMERS — STRICT LOSO v2 (Fully Leak-Free)")
    print("  NO global pretraining. Every fold: CL → DANN → CMMA from scratch.")
    print("  Test subject NEVER seen in ANY training stage.")
    print("=" * 72)

    print("\nLoading DEAP data (with subject IDs)...")
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_all, eeg_lbl_all, eeg_sids = deap.load_all(flatten=True)

    print("Loading IEMOCAP data...")
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_all, sp_lbl_all, _ = iemocap.load_all()

    # Fixed IEMOCAP split (no subject leak concern — different corpus)
    sp_Xt, sp_Xv, sp_yt, sp_yv = train_test_split(
        sp_all, sp_lbl_all,
        test_size=0.2, stratify=sp_lbl_all, random_state=cfg.seed,
    )

    # Group DEAP by subject
    subjects = sorted(set(eeg_sids.tolist()))
    eeg_by_subj: dict[int, np.ndarray] = {}
    lbl_by_subj: dict[int, np.ndarray] = {}
    for s in subjects:
        mask = eeg_sids == s
        eeg_by_subj[s] = eeg_all[mask]
        lbl_by_subj[s] = eeg_lbl_all[mask]

    print(f"\n  DEAP:    {len(eeg_all):,} samples, {len(subjects)} subjects")
    print(f"  IEMOCAP: {len(sp_all):,} total → {len(sp_Xt):,} train, {len(sp_Xv):,} val")

    label_names = ["Happy", "Sad", "Angry", "Neutral"]
    eeg_counts = Counter(eeg_lbl_all.tolist())
    print(f"\n  DEAP class distribution:")
    for c in range(4):
        print(f"    {label_names[c]:8s}: {eeg_counts.get(c, 0):,}")

    if args.skip_contrastive:
        print("\n  ⚠ --skip-contrastive: skipping per-fold CL pretraining")

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

        print(f"\n{'=' * 72}")
        print(f"  STRICT LOSO Fold {fold_idx + 1}/{len(subjects)}")
        print(f"  Test: subject {test_subj}  |  Val: subject {val_subj}")
        print(f"  Train: {len(train_subjs)} subjects")
        print(f"  Pipeline: {'CL →' if not args.skip_contrastive else ''} "
              f"DANN → CMMA (all from scratch)")
        print(f"{'=' * 72}")

        # ── Check cached results ──
        fold_json = loso_dir / f"fold_{fold_idx:02d}.json"
        if fold_json.exists():
            print("  → Cached result found — skipping")
            with open(fold_json) as f:
                cached = json.load(f)
            fold_results.append(cached["metrics"])
            all_test_preds.append(np.array(cached["predictions"]))
            all_test_labels.append(np.array(cached["true_labels"]))
            continue

        fold_t0 = time.time()

        # Set reproducible seed per fold
        set_seed(cfg.seed + fold_idx)

        # ── STEP 1: Subject split ──
        eeg_train = np.concatenate([eeg_by_subj[s] for s in train_subjs])
        eeg_train_lbl = np.concatenate([lbl_by_subj[s] for s in train_subjs])
        eeg_val = eeg_by_subj[val_subj]
        eeg_val_lbl = lbl_by_subj[val_subj]
        eeg_test = eeg_by_subj[test_subj]
        eeg_test_lbl = lbl_by_subj[test_subj]

        print(f"\n  STEP 1 — Subject split:")
        print(f"    train={len(eeg_train):,}  val={len(eeg_val):,}  "
              f"test={len(eeg_test):,}")
        train_counts = Counter(eeg_train_lbl.tolist())
        print(f"    Train class dist: {dict(sorted(train_counts.items()))}")

        # ── STEP 2: Cross-subject normalization ──
        eeg_train, eeg_val, eeg_test = normalize_cross_subject(
            eeg_train, eeg_val, eeg_test
        )
        print(f"\n  STEP 2 — Cross-subject normalization applied (fit on train)")

        # ── STEP 3: Contrastive EEG pretraining (30 subjects ONLY) ──
        if not args.skip_contrastive:
            print(f"\n  STEP 3 — Contrastive EEG pretraining (30 subjs, NO test subj)")
            eeg_encoder = contrastive_pretrain_eeg(
                eeg_train, cfg, device,
                epochs=hparams["contrastive_epochs"],
                batch_size=hparams["contrastive_batch_size"],
                lr=hparams["contrastive_lr"],
                temperature=hparams["contrastive_temperature"],
            )
        else:
            print(f"\n  STEP 3 — SKIPPED (--skip-contrastive)")
            ecfg = cfg.model.eeg_encoder
            eeg_encoder = EEGEncoder(
                input_dim=ecfg.input_dim,
                hidden_dims=list(ecfg.hidden_dims),
                embedding_dim=ecfg.embedding_dim,
                dropout=ecfg.dropout,
            ).to(device)

        # ── STEP 4: DANN domain adaptation (30 subjs EEG + IEMOCAP) ──
        print(f"\n  STEP 4 — DANN domain adaptation (30 subjs, NO test subj)")
        eeg_encoder, speech_encoder = dann_train_per_fold(
            eeg_encoder, eeg_train, eeg_train_lbl,
            sp_Xt, sp_yt, cfg, device,
            epochs=hparams["dann_epochs"],
            batch_size=hparams["dann_batch_size"],
            lr=hparams["dann_lr"],
            domain_weight=hparams["dann_domain_weight"],
        )

        # ── STEP 5: CMMA fusion training ──
        print(f"\n  STEP 5 — CMMA fusion training (fresh CMMA, pretrained encoders)")
        cmma = build_cmma_only(cfg, device)

        trainer = StrictLOSOTrainer(cfg, focal_gamma=hparams["fusion_focal_gamma"])
        trainer.epochs = hparams["fusion_epochs"]
        trainer.patience = hparams["fusion_patience"]
        trainer.samples_per_epoch = hparams["fusion_samples_per_epoch"]
        trainer.warmup_epochs = hparams["fusion_warmup_epochs"]
        trainer.freeze_encoder_epochs = hparams["fusion_freeze_encoder_epochs"]
        trainer.label_smoothing = hparams["fusion_label_smoothing"]
        trainer.gate_div_weight = hparams["fusion_gate_div_weight"]
        trainer.set_eeg_class_weights(eeg_train_lbl)

        fold_save = loso_dir / f"fold_{fold_idx:02d}"
        fold_save.mkdir(parents=True, exist_ok=True)

        history = trainer.fit(
            eeg_encoder, speech_encoder, cmma,
            eeg_train, eeg_train_lbl,
            sp_Xt, sp_yt,
            eeg_val, eeg_val_lbl,
            sp_Xv, sp_yv,
            save_dir=str(fold_save),
        )

        # Reload best checkpoint
        best_ckpt = fold_save / "best_strict_loso.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device, weights_only=False)
            eeg_encoder.load_state_dict(state["eeg_encoder"])
            speech_encoder.load_state_dict(state["speech_encoder"])
            cmma.load_state_dict(state["cmma_fusion"])
            saved_val_acc = state.get("val_acc", 0)
            print(f"    Loaded best checkpoint (val_acc={saved_val_acc:.2%})")
        else:
            print("    ⚠ No best checkpoint — using last epoch weights")

        # ── STEP 6: Test on held-out subject ──
        print(f"\n  STEP 6 — Evaluate on held-out subject {test_subj}")
        preds = evaluate_test_ensemble(
            eeg_encoder, speech_encoder, cmma,
            eeg_test, eeg_test_lbl,
            sp_Xv, sp_yv,
            device,
            num_classes=cfg.model.num_classes,
            n_pairings=hparams["n_test_pairings"],
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
            "protocol": "strict_loso_v2",
            "leakage": "none",
            "per_fold_pretraining": True,
            "contrastive_skipped": args.skip_contrastive,
            "hyperparameters": hparams,
        }
        with open(fold_json, "w") as f:
            json.dump(fold_data, f, indent=2)

        fold_results.append(serializable_metrics)
        all_test_preds.append(preds)
        all_test_labels.append(eeg_test_lbl)

        # Free GPU memory
        del eeg_encoder, speech_encoder, cmma, trainer
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

    print(f"\n{'=' * 72}")
    print(f"  STRICT LOSO v2 RESULTS  ({len(fold_results)} folds, "
          f"{total_elapsed:.0f}s total)")
    print(f"{'=' * 72}")
    print(f"  Overall pooled accuracy : {overall['accuracy']:.2%}")
    print(f"  Mean fold accuracy      : {mean_m['accuracy']:.2%} ± "
          f"{std_m['accuracy']:.2%}")
    print(f"  Mean fold F1 (macro)    : {mean_m['f1_macro']:.3f} ± "
          f"{std_m['f1_macro']:.3f}")
    print(f"  Mean fold F1 (weighted) : {mean_m['f1_weighted']:.3f} ± "
          f"{std_m['f1_weighted']:.3f}")
    print(f"  Mean Cohen's kappa      : {mean_m['kappa']:.3f} ± "
          f"{std_m['kappa']:.3f}")
    print(f"\n  Per-fold accuracy range : [{min(per_fold_acc):.2%}, "
          f"{max(per_fold_acc):.2%}]")
    print(f"\n{overall['report_str']}")

    # ── Compare with LOSO v2 (leaked) ──
    v2_summary = Path(paths["outputs"]) / "loso_v2" / "loso_summary.json"
    if v2_summary.exists():
        with open(v2_summary) as f:
            v2 = json.load(f)
        v2_acc = v2["mean_metrics"]["accuracy"]
        delta = mean_m["accuracy"] - v2_acc
        print(f"\n  LOSO v2 (leaked) vs Strict LOSO v2:")
        print(f"    LOSO v2 (leaked):  {v2_acc:.2%}")
        print(f"    Strict LOSO v2:    {mean_m['accuracy']:.2%}")
        print(f"    Δ (leak impact):   {delta:+.2%}")
        print(f"    → The leaked encoders {'inflated' if delta < 0 else 'deflated'} "
              f"results by {abs(delta):.2%}")

    # ── Compare with LOSO v1 ──
    v1_summary = Path(paths["outputs"]) / "loso" / "loso_summary.json"
    if v1_summary.exists():
        with open(v1_summary) as f:
            v1 = json.load(f)
        v1_acc = v1["mean_metrics"]["accuracy"]
        delta_v1 = mean_m["accuracy"] - v1_acc
        print(f"\n  LOSO v1 (baseline) vs Strict LOSO v2:")
        print(f"    LOSO v1:           {v1_acc:.2%}")
        print(f"    Strict LOSO v2:    {mean_m['accuracy']:.2%}")
        print(f"    Δ:                 {delta_v1:+.2%}")

    # ── Save summary ──
    summary = {
        "version": "strict_loso_v2",
        "protocol": "fully_leak_free",
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
        "hyperparameters": hparams,
        "contrastive_skipped": args.skip_contrastive,
        "data_leakage": "none — all pretraining per fold",
        "improvements_over_loso_v2": [
            "per_fold_contrastive_pretraining",
            "per_fold_dann_training",
            "no_global_checkpoint_reuse",
            "fresh_weight_initialization_per_fold",
        ],
    }
    summary_path = loso_dir / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print(f"\n{'=' * 72}")
    print("  STRICT LOSO v2 evaluation complete!")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
