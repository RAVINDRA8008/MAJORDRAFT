#!/usr/bin/env python3
"""Pre-train the EEG encoder on DEAP differential-entropy features.

v4 optimisations:
- **Aggressive class rebalancing**: caps each class to max_samples_per_class
  (default 5000) BEFORE train/val split to eliminate 37x imbalance
- Focal loss (gamma=2) for remaining mild imbalance
- Balanced batch sampling (WeightedRandomSampler)
- Mixup augmentation for regularisation
- Mixed-precision training (AMP)
- Cosine LR scheduling with warmup
- Early stopping + best-checkpoint selection
- Gradient clipping
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.paths import get_paths, ensure_dirs
from src.utils.logging_setup import setup_logging
from src.utils.device import get_device
from src.data.deap_loader import DEAPLoader
from src.models.eeg_encoder import EEGEncoder
from src.utils.visualization import plot_loss_curves, plot_accuracy_curves

LABEL_NAMES = {0: "angry", 1: "happy", 2: "sad", 3: "neutral"}


def balance_classes(
    features: np.ndarray,
    labels: np.ndarray,
    max_per_class: int = 5000,
    min_oversample: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggressively rebalance classes via under+oversampling.

    1. Cap each class at *max_per_class* (undersample majority).
    2. Oversample minority classes to at least *min_oversample* via
       repetition + noise jitter.

    This reduces 37x imbalance to at most 2.5x.
    """
    rng = np.random.RandomState(seed)
    balanced_feats, balanced_labels = [], []

    for cls in sorted(set(labels.tolist())):
        mask = labels == cls
        cls_feats = features[mask]
        n = len(cls_feats)

        if n > max_per_class:
            # Undersample: random selection without replacement
            idx = rng.choice(n, max_per_class, replace=False)
            cls_feats = cls_feats[idx]
        elif n < min_oversample:
            # Oversample with noise jitter to avoid exact duplicates
            deficit = min_oversample - n
            extra_idx = rng.choice(n, deficit, replace=True)
            extra = cls_feats[extra_idx] + rng.normal(0, 0.01, cls_feats[extra_idx].shape).astype(cls_feats.dtype)
            cls_feats = np.concatenate([cls_feats, extra], axis=0)

        balanced_feats.append(cls_feats)
        balanced_labels.append(np.full(len(cls_feats), cls, dtype=labels.dtype))

    features_out = np.concatenate(balanced_feats, axis=0)
    labels_out = np.concatenate(balanced_labels, axis=0)

    # Shuffle
    perm = rng.permutation(len(features_out))
    return features_out[perm], labels_out[perm]


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation: returns mixed inputs and pairs of targets.

    Returns:
        (mixed_x, y_a, y_b, lam)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification: FL = -(1-pt)^gamma * log(pt)"""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = Counter(labels.numpy().tolist())
    total = sum(counts.values())
    n_classes = max(counts.keys()) + 1
    weights = torch.zeros(n_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (n_classes * cnt)
    return weights


def make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    counts = Counter(labels.numpy().tolist())
    sample_weights = torch.tensor([1.0 / counts[int(l)] for l in labels])
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train EEG encoder on DEAP")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    setup_logging(cfg)
    set_seed(cfg.seed)
    paths = get_paths(cfg)
    ensure_dirs(paths)
    device = get_device()
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    # ── Load data ──
    loader = DEAPLoader(processed_dir=paths["deap_processed"])
    features, labels, _ = loader.load_all(flatten=True)

    # ── Aggressive class rebalancing BEFORE train/val split ──
    ecfg = cfg.model.eeg_encoder
    max_per_cls = getattr(ecfg, "max_samples_per_class", 5000)
    min_per_cls = getattr(ecfg, "min_samples_per_class", 2000)
    print(f"Original data: {len(labels)} samples")
    dist_orig = Counter(labels.tolist())
    imbalance_orig = max(dist_orig.values()) / max(min(dist_orig.values()), 1)
    print(f"  Original distribution: {dict(sorted(dist_orig.items()))} ({imbalance_orig:.0f}x imbalance)")

    features, labels = balance_classes(
        features, labels,
        max_per_class=max_per_cls,
        min_oversample=min_per_cls,
        seed=cfg.seed,
    )
    print(f"After rebalancing: {len(labels)} samples")
    dist_bal = Counter(labels.tolist())
    imbalance_bal = max(dist_bal.values()) / max(min(dist_bal.values()), 1)
    print(f"  Balanced distribution: {dict(sorted(dist_bal.items()))} ({imbalance_bal:.1f}x imbalance)")

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=cfg.seed,
    )
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.long)

    # ── Class balance analysis (post-split) ──
    dist = Counter(y_train.numpy().tolist())
    imbalance = max(dist.values()) / max(min(dist.values()), 1)
    dist_str = ", ".join(f"{LABEL_NAMES.get(k, k)}: {v}" for k, v in sorted(dist.items()))
    print(f"Train distribution: {dist_str}  (imbalance {imbalance:.1f}x)")

    use_balanced = imbalance > 2.0
    class_weights = compute_class_weights(y_train).to(device)

    ecfg = cfg.model.eeg_encoder
    batch_size = getattr(ecfg, "batch_size", 64)
    lr = getattr(ecfg, "lr", 1e-3)
    epochs = getattr(ecfg, "pretrain_epochs", 50)
    patience = getattr(ecfg, "patience", 15)
    use_mixup = getattr(ecfg, "mixup", True)
    mixup_alpha = getattr(ecfg, "mixup_alpha", 0.4)

    # ── DataLoaders ──
    train_ds = TensorDataset(X_train, y_train)
    sampler = make_balanced_sampler(y_train) if use_balanced else None
    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        drop_last=True, num_workers=2, pin_memory=True,
    )
    val_ds = TensorDataset(X_val, y_val)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    # ── Build model ──
    encoder = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)

    head = nn.Linear(ecfg.embedding_dim, cfg.model.num_classes).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(
        gamma=2.0,
        weight=class_weights,
        label_smoothing=0.05,
    )

    # Cosine LR scheduler with warmup
    warmup_epochs = max(1, epochs // 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )

    scaler = GradScaler(enabled=use_amp)

    # ── Training ──
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    save_dir = Path(paths["checkpoints"]) / "eeg"
    save_dir.mkdir(parents=True, exist_ok=True)

    log_every = cfg.training.get("log_every", 5)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Warmup
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ── Train ──
        encoder.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                if use_mixup and np.random.rand() < 0.5:
                    xb_mix, ya, yb_mix, lam = mixup_batch(xb, yb, mixup_alpha)
                    emb = encoder(xb_mix)
                    logits = head(emb)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_mix)
                else:
                    emb = encoder(xb)
                    logits = head(emb)
                    loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)

        if epoch > warmup_epochs:
            scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ── Validate ──
        encoder.eval()
        head.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = head(encoder(xb))
                    vloss += criterion(logits, yb).item() * len(xb)
                vcorrect += (logits.argmax(1) == yb).sum().item()
                vtotal += len(xb)

        val_loss = vloss / max(vtotal, 1)
        val_acc = vcorrect / max(vtotal, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Best checkpoint ──
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[EEG] {epoch}/{epochs}  loss={train_loss:.4f}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                f"best={best_val_acc:.3f}@{best_epoch}  lr={cur_lr:.6f}  "
                f"patience={patience_counter}/{patience}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── Restore best & save ──
    if best_state is not None:
        encoder.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch} (val_acc={best_val_acc:.4f})")

    torch.save(encoder.state_dict(), save_dir / "eeg_encoder_final.pt")

    elapsed = time.time() - t0
    print(f"EEG pre-training complete in {elapsed:.0f}s. Best val acc: {best_val_acc:.3f}")

    # ── Plot ──
    out = Path(paths["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(
        history["train_loss"], history["val_loss"],
        title="EEG Encoder Pre-Training Loss",
        save_path=str(out / "eeg_loss.png"),
    )
    plot_accuracy_curves(
        history["train_acc"], history["val_acc"],
        title="EEG Encoder Accuracy",
        save_path=str(out / "eeg_acc.png"),
    )


if __name__ == "__main__":
    main()
