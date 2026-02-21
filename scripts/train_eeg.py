#!/usr/bin/env python3
"""Pre-train the EEG (FC) encoder on DEAP differential-entropy features.

Trains a simple classification head on top of the EEGEncoder so that
the encoder learns meaningful emotion embeddings before fusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

    # ── Load data ──
    loader = DEAPLoader(processed_dir=paths["deap_processed"])
    features, labels, _ = loader.load_all(flatten=True)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=cfg.seed,
    )
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    ecfg = cfg.model.eeg_encoder
    batch_size = getattr(ecfg, "batch_size", 64)
    lr = getattr(ecfg, "lr", 1e-3)
    epochs = getattr(ecfg, "pretrain_epochs", 20)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Build model ──
    encoder = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    ).to(device)

    # Classification head for pre-training
    head = nn.Linear(ecfg.embedding_dim, cfg.model.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    save_dir = Path(paths["checkpoints"]) / "eeg"
    save_dir.mkdir(parents=True, exist_ok=True)

    log_every = cfg.training.get("log_every", 5)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        encoder.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            emb = encoder(xb)
            logits = head(emb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ── Validate ──
        encoder.eval()
        head.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                emb = encoder(xb)
                logits = head(emb)
                vloss += criterion(logits, yb).item() * len(xb)
                vcorrect += (logits.argmax(1) == yb).sum().item()
                vtotal += len(xb)

        val_loss = vloss / max(vtotal, 1)
        val_acc = vcorrect / max(vtotal, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % log_every == 0 or epoch == epochs:
            print(
                f"[EEG Encoder] Epoch {epoch}/{epochs}  "
                f"TrLoss={train_loss:.4f}  TrAcc={train_acc:.3f}  "
                f"VaLoss={val_loss:.4f}  VaAcc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), save_dir / "eeg_encoder_final.pt")

    # Final save (always overwrite with last epoch too)
    torch.save(encoder.state_dict(), save_dir / "eeg_encoder_final.pt")

    # ── Plot ──
    out = Path(paths["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(
        history["train_loss"],
        history["val_loss"],
        title="EEG Encoder Pre-Training Loss",
        save_path=str(out / "eeg_loss.png"),
    )
    plot_accuracy_curves(
        history["train_acc"],
        history["val_acc"],
        title="EEG Encoder Accuracy",
        save_path=str(out / "eeg_acc.png"),
    )
    print(f"EEG pre-training complete. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {save_dir / 'eeg_encoder_final.pt'}")


if __name__ == "__main__":
    main()
