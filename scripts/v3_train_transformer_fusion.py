#!/usr/bin/env python3
"""v3 — Train transformer fusion classifier with DANN-pretrained encoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from src.training.fusion_trainer import FusionTrainer, LabelAlignedDataset, FocalLoss
from src.utils.visualization import plot_loss_curves, plot_accuracy_curves


def _encode_batched(encoder, data, device, batch_size=512):
    parts = []
    t = torch.as_tensor(data, dtype=torch.float32)
    for i in range(0, len(t), batch_size):
        parts.append(encoder(t[i : i + batch_size].to(device)).cpu())
    return torch.cat(parts, dim=0)


class TransformerFusionTrainer:
    """Standalone trainer for TransformerFusionClassifier (v3).

    Uses label-aligned data + focal loss + AMP + early stopping.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = self.device.type == "cuda"
        self.num_classes = cfg.model.num_classes

        v3 = getattr(cfg, "v3", {})
        tf = v3.get("transformer_fusion", {}) if isinstance(v3, dict) else getattr(v3, "transformer_fusion", {})

        self.epochs = tf.get("epochs", 60) if isinstance(tf, dict) else getattr(tf, "epochs", 60)
        self.batch_size = tf.get("batch_size", 128) if isinstance(tf, dict) else getattr(tf, "batch_size", 128)
        self.lr = tf.get("lr", 5e-4) if isinstance(tf, dict) else getattr(tf, "lr", 5e-4)
        self.patience = tf.get("patience", 15) if isinstance(tf, dict) else getattr(tf, "patience", 15)

    def fit(
        self,
        model: TransformerFusionClassifier,
        eeg_emb: torch.Tensor,
        eeg_labels: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_labels: torch.Tensor,
        val_eeg_emb: torch.Tensor,
        val_eeg_labels: torch.Tensor,
        val_speech_emb: torch.Tensor,
        val_speech_labels: torch.Tensor,
        save_dir: str | Path | None = None,
    ) -> dict[str, list]:
        from collections import Counter
        from torch.cuda.amp import GradScaler, autocast
        from torch.utils.data import DataLoader
        import torch.nn as nn

        model = model.to(self.device)
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Class weights
        all_labels = torch.cat([eeg_labels, speech_labels])
        counts = Counter(all_labels.numpy().tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            weights[cls] = total / (n_cls * cnt)

        criterion = FocalLoss(gamma=2.0, weight=weights.to(self.device), label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6,
        )
        scaler = GradScaler(enabled=self.use_amp)

        train_ds = LabelAlignedDataset(
            eeg_emb, eeg_labels, speech_emb, speech_labels,
            num_classes=self.num_classes, balance_classes=True,
        )
        val_ds = LabelAlignedDataset(
            val_eeg_emb, val_eeg_labels, val_speech_emb, val_speech_labels,
            num_classes=self.num_classes, balance_classes=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(val_ds, batch_size=512)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for eeg_b, sp_b, lbl_b in train_loader:
                eeg_b = eeg_b.to(self.device, non_blocking=True)
                sp_b = sp_b.to(self.device, non_blocking=True)
                lbl_b = lbl_b.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = model(eeg_b, sp_b)
                    loss = criterion(logits, lbl_b)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * eeg_b.size(0)
                train_correct += (logits.argmax(1) == lbl_b).sum().item()
                train_total += eeg_b.size(0)

            scheduler.step()

            # Validate
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for eeg_b, sp_b, lbl_b in val_loader:
                    eeg_b = eeg_b.to(self.device)
                    sp_b = sp_b.to(self.device)
                    lbl_b = lbl_b.to(self.device)
                    logits = model(eeg_b, sp_b)
                    val_loss += criterion(logits, lbl_b).item() * eeg_b.size(0)
                    val_correct += (logits.argmax(1) == lbl_b).sum().item()
                    val_total += eeg_b.size(0)

            t_loss = train_loss / max(train_total, 1)
            v_loss = val_loss / max(val_total, 1)
            t_acc = train_correct / max(train_total, 1)
            v_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                print(
                    f"TFusion {epoch}/{self.epochs}  "
                    f"train: loss={t_loss:.4f} acc={t_acc:.3f}  "
                    f"val: loss={v_loss:.4f} acc={v_acc:.3f}"
                )

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                patience_counter = 0
                if save_dir:
                    sd = Path(save_dir)
                    sd.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"fusion": model.state_dict(), "val_acc": v_acc, "epoch": epoch},
                        sd / "best_transformer_fusion.pt",
                    )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Transformer fusion training complete — best val_acc: {best_val_acc:.4f}")
        return history


def main() -> None:
    parser = argparse.ArgumentParser(description="v3: Transformer fusion training")
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

    # Load data
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_feat, eeg_lbl, _ = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_feat, sp_lbl, _ = iemocap.load_all()

    # Split
    eeg_Xt, eeg_Xv, eeg_yt, eeg_yv = train_test_split(
        eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
    )
    sp_Xt, sp_Xv, sp_yt, sp_yv = train_test_split(
        sp_feat, sp_lbl, test_size=0.2, stratify=sp_lbl, random_state=cfg.seed,
    )

    # Load DANN-pretrained encoders (fallback to v2 if DANN not available)
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

    # Try DANN → contrastive → v2 supervised checkpoints (in priority order)
    for eeg_ckpt_name in ["v3/eeg_encoder_dann.pt", "v3/eeg_encoder_contrastive.pt", "eeg/eeg_encoder_final.pt"]:
        p = ckpt / eeg_ckpt_name
        if p.exists():
            eeg_enc.load_state_dict(torch.load(p, map_location=device))
            print(f"Loaded EEG encoder from {p}")
            break
    else:
        print("WARNING: No EEG encoder checkpoint found")

    for sp_ckpt_name in ["v3/speech_encoder_dann.pt", "v3/speech_encoder_contrastive.pt", "speech/speech_encoder_final.pt"]:
        p = ckpt / sp_ckpt_name
        if p.exists():
            speech_enc.load_state_dict(torch.load(p, map_location=device))
            print(f"Loaded speech encoder from {p}")
            break
    else:
        print("WARNING: No speech encoder checkpoint found")

    eeg_enc.eval()
    speech_enc.eval()

    # Encode in batches
    with torch.no_grad():
        eeg_emb_t = _encode_batched(eeg_enc, eeg_Xt, device)
        eeg_emb_v = _encode_batched(eeg_enc, eeg_Xv, device)
        sp_emb_t = _encode_batched(speech_enc, sp_Xt, device)
        sp_emb_v = _encode_batched(speech_enc, sp_Xv, device)

    eeg_yt_t = torch.as_tensor(eeg_yt, dtype=torch.long)
    eeg_yv_t = torch.as_tensor(eeg_yv, dtype=torch.long)
    sp_yt_t = torch.as_tensor(sp_yt, dtype=torch.long)
    sp_yv_t = torch.as_tensor(sp_yv, dtype=torch.long)

    print(f"\nTransformer fusion training:")
    print(f"  EEG train: {len(eeg_emb_t)}, Speech train: {len(sp_emb_t)}")
    print(f"  EEG val:   {len(eeg_emb_v)}, Speech val:   {len(sp_emb_v)}")

    # Build model
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
        dropout=0.1,
        modality_dropout_prob=0.1,
    )

    save_dir = ckpt / "v3"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerFusionTrainer(cfg)
    history = trainer.fit(
        fusion,
        eeg_emb_t, eeg_yt_t,
        sp_emb_t, sp_yt_t,
        eeg_emb_v, eeg_yv_t,
        sp_emb_v, sp_yv_t,
        save_dir=save_dir,
    )

    # Plot
    out = Path(paths["outputs"])
    plot_loss_curves(
        history["train_loss"], history["val_loss"],
        title="v3 Transformer Fusion Loss",
        save_path=str(out / "v3_transformer_fusion_loss.png"),
    )
    plot_accuracy_curves(
        history["train_acc"], history["val_acc"],
        title="v3 Transformer Fusion Accuracy",
        save_path=str(out / "v3_transformer_fusion_acc.png"),
    )
    print("Transformer fusion training complete.")


if __name__ == "__main__":
    main()
