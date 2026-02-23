#!/usr/bin/env python3
"""v3 — RL-augmented transformer fusion training (PPO v2)."""

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
from src.models.gan import ConditionalGAN
from src.models.transformer_fusion import TransformerFusionClassifier
from src.training.rl_v2_trainer import RLv2Trainer
from src.utils.visualization import plot_augmentation_ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="v3: RL-augmented transformer fusion")
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

    ckpt = Path(paths["checkpoints"])
    ecfg = cfg.model.eeg_encoder
    scfg = cfg.model.speech_encoder

    # ── Load GAN ──
    gan_config = {
        "feature_dim": cfg.model.gan.feature_dim,
        "latent_dim": cfg.model.gan.noise_dim,
        "num_classes": cfg.model.num_classes,
        "generator_hidden_dims": [cfg.model.gan.hidden_dim, cfg.model.gan.hidden_dim * 2, cfg.model.gan.hidden_dim],
        "discriminator_hidden_dims": [cfg.model.gan.hidden_dim, cfg.model.gan.hidden_dim * 2, cfg.model.gan.hidden_dim],
        "lr": cfg.model.gan.lr_g,
    }
    gan = ConditionalGAN(gan_config, torch.device("cpu"))
    gan_path = ckpt / "gan" / "gan_final.pt"
    if gan_path.exists():
        gan.load_state_dict(torch.load(gan_path, map_location="cpu"))
        print(f"Loaded GAN from {gan_path}")

    # ── Load encoders (DANN > contrastive > v2) ──
    eeg_enc = EEGEncoder(
        input_dim=ecfg.input_dim,
        hidden_dims=list(ecfg.hidden_dims),
        embedding_dim=ecfg.embedding_dim,
        dropout=ecfg.dropout,
    )
    for p in ["v3/eeg_encoder_dann.pt", "v3/eeg_encoder_contrastive.pt", "eeg/eeg_encoder_final.pt"]:
        if (ckpt / p).exists():
            eeg_enc.load_state_dict(torch.load(ckpt / p, map_location="cpu"))
            print(f"Loaded EEG encoder: {p}")
            break

    speech_enc = SpeechEncoder(
        n_features=scfg.n_mfcc,
        embedding_dim=scfg.embedding_dim,
    )
    for p in ["v3/speech_encoder_dann.pt", "v3/speech_encoder_contrastive.pt", "speech/speech_encoder_final.pt"]:
        if (ckpt / p).exists():
            speech_enc.load_state_dict(torch.load(ckpt / p, map_location="cpu"))
            print(f"Loaded speech encoder: {p}")
            break

    # ── Load transformer fusion (warm start from v3_train_transformer_fusion) ──
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

    tf_ckpt = ckpt / "v3" / "best_transformer_fusion.pt"
    if tf_ckpt.exists():
        sd = torch.load(tf_ckpt, map_location="cpu")
        fusion.load_state_dict(sd.get("fusion", sd))
        print(f"Loaded transformer fusion from {tf_ckpt}")

    # ── Load data ──
    deap = DEAPLoader(processed_dir=paths["deap_processed"])
    eeg_feat, eeg_lbl, _ = deap.load_all(flatten=True)
    iemocap = IEMOCAPLoader(processed_dir=paths["iemocap_processed"])
    sp_feat, sp_lbl, _ = iemocap.load_all()

    eeg_Xt, eeg_Xv, eeg_yt, eeg_yv = train_test_split(
        eeg_feat, eeg_lbl, test_size=0.2, stratify=eeg_lbl, random_state=cfg.seed,
    )
    sp_Xt, sp_Xv, sp_yt, sp_yv = train_test_split(
        sp_feat, sp_lbl, test_size=0.2, stratify=sp_lbl, random_state=cfg.seed,
    )

    # ── Train RL v2 ──
    rl_save = ckpt / "v3"
    rl_save.mkdir(parents=True, exist_ok=True)

    trainer = RLv2Trainer(
        cfg, gan=gan,
        eeg_encoder=eeg_enc, speech_encoder=speech_enc, fusion=fusion,
    )
    history = trainer.train(
        torch.as_tensor(eeg_Xt, dtype=torch.float32),
        torch.as_tensor(eeg_yt, dtype=torch.long),
        torch.as_tensor(sp_Xt, dtype=torch.float32),
        torch.as_tensor(sp_yt, dtype=torch.long),
        torch.as_tensor(eeg_Xv, dtype=torch.float32),
        torch.as_tensor(eeg_yv, dtype=torch.long),
        torch.as_tensor(sp_Xv, dtype=torch.float32),
        torch.as_tensor(sp_yv, dtype=torch.long),
        save_dir=rl_save,
    )

    # Plot
    out = Path(paths["outputs"])
    plot_augmentation_ratios(
        history["aug_ratio"],
        save_path=str(out / "v3_rl_aug_ratios.png"),
    )
    print("RL v2 training complete. Best macro F1:", max(history["macro_f1"]))


if __name__ == "__main__":
    main()
