"""End-to-end training pipeline.

Orchestrates the full AMERS workflow:
  1. Load & preprocess DEAP + IEMOCAP
  2. Train cGAN on EEG features
  3. Pre-train speech encoder on IEMOCAP
  4. Warm-up fusion classifier (no RL)
  5. RL-augmented fusion training (PPO controls GAN ratio)
  6. Evaluate and save results
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.seed import set_seed
from src.utils.device import get_device, log_gpu_memory
from src.utils.paths import get_paths, ensure_dirs
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging_setup import setup_logging

from src.data.deap_loader import DEAPLoader
from src.data.iemocap_loader import IEMOCAPLoader
from src.data.label_mapper import LabelMapper

from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.gan import ConditionalGAN
from src.models.fusion import FusionClassifier

from src.training.gan_trainer import GANTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.fusion_trainer import FusionTrainer
from src.training.rl_trainer import RLTrainer

logger = logging.getLogger(__name__)


def run_pipeline(cfg: DictConfig) -> dict:
    """Execute the full AMERS training pipeline.

    Args:
        cfg: Merged OmegaConf config.

    Returns:
        Dict with all history and final metrics.
    """
    # ----- Setup -----
    setup_logging(cfg)
    set_seed(cfg.seed)
    device = get_device()
    paths = get_paths(cfg)
    ensure_dirs(paths)

    logger.info("=" * 60)
    logger.info("AMERS Training Pipeline — device=%s", device)
    logger.info("=" * 60)

    results: dict = {}

    # ----- 1. Load data -----
    logger.info("[1/6] Loading data ...")

    deap_loader = DEAPLoader(
        processed_dir=paths["deap_processed"],
        label_mapper=LabelMapper(),
    )
    eeg_features, eeg_labels = deap_loader.load_all(flatten=True)
    logger.info("  DEAP: %d samples, feature_dim=%d", len(eeg_features), eeg_features.shape[1])

    iemocap_loader = IEMOCAPLoader(
        processed_dir=paths["iemocap_processed"],
        label_mapper=LabelMapper(),
    )
    speech_features, speech_labels = iemocap_loader.load_all()
    logger.info("  IEMOCAP: %d samples, shape=%s", len(speech_features), speech_features.shape[1:])

    # Simple train/val split (will be replaced by LOSO CV in evaluation)
    from sklearn.model_selection import train_test_split

    eeg_X_train, eeg_X_val, eeg_y_train, eeg_y_val = train_test_split(
        eeg_features, eeg_labels, test_size=0.2, stratify=eeg_labels,
        random_state=cfg.seed,
    )
    sp_X_train, sp_X_val, sp_y_train, sp_y_val = train_test_split(
        speech_features, speech_labels, test_size=0.2, stratify=speech_labels,
        random_state=cfg.seed,
    )

    # Convert to tensors
    eeg_X_train = torch.as_tensor(eeg_X_train, dtype=torch.float32)
    eeg_y_train = torch.as_tensor(eeg_y_train, dtype=torch.long)
    eeg_X_val = torch.as_tensor(eeg_X_val, dtype=torch.float32)
    eeg_y_val = torch.as_tensor(eeg_y_val, dtype=torch.long)
    sp_X_train = torch.as_tensor(sp_X_train, dtype=torch.float32)
    sp_y_train = torch.as_tensor(sp_y_train, dtype=torch.long)
    sp_X_val = torch.as_tensor(sp_X_val, dtype=torch.float32)
    sp_y_val = torch.as_tensor(sp_y_val, dtype=torch.long)

    log_gpu_memory()

    # ----- 2. Train cGAN -----
    logger.info("[2/6] Training conditional GAN on EEG features ...")
    gan_trainer = GANTrainer(cfg)
    gan_history = gan_trainer.fit(
        eeg_X_train, eeg_y_train,
        save_dir=Path(paths["checkpoints"]) / "gan",
    )
    results["gan"] = gan_history

    # ----- 3. Pre-train speech encoder -----
    logger.info("[3/6] Pre-training speech encoder on IEMOCAP ...")
    speech_trainer = SpeechTrainer(cfg)
    speech_history = speech_trainer.fit(
        sp_X_train, sp_y_train,
        val_features=sp_X_val, val_labels=sp_y_val,
        save_dir=Path(paths["checkpoints"]) / "speech",
    )
    results["speech"] = speech_history

    # ----- 4. EEG encoder (trained alongside GAN discriminator) -----
    logger.info("[4/6] Initialising EEG encoder ...")
    eeg_encoder = EEGEncoder(
        input_dim=cfg.model.eeg_encoder.input_dim,
        hidden_dims=list(cfg.model.eeg_encoder.hidden_dims),
        embedding_dim=cfg.model.eeg_encoder.embedding_dim,
        dropout=cfg.model.eeg_encoder.dropout,
    ).to(device)
    # Quick train with cross-entropy
    from src.models.classifier import EncoderClassifier
    enc_clf = EncoderClassifier(eeg_encoder, cfg.model.eeg_encoder.embedding_dim, cfg.model.num_classes).to(device)
    enc_opt = torch.optim.Adam(enc_clf.parameters(), lr=1e-3)
    enc_criterion = torch.nn.CrossEntropyLoss()
    for ep in range(cfg.model.eeg_encoder.get("pretrain_epochs", 20)):
        enc_clf.train()
        idx = torch.randperm(len(eeg_X_train))
        for start in range(0, len(idx), 128):
            batch_idx = idx[start:start+128]
            x = eeg_X_train[batch_idx].to(device)
            y = eeg_y_train[batch_idx].to(device)
            loss = enc_criterion(enc_clf(x), y)
            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()
    logger.info("  EEG encoder pre-trained.")

    # ----- 5. Warm-up fusion (no RL) -----
    logger.info("[5/6] Warm-up fusion training ...")
    with torch.no_grad():
        eeg_encoder.eval()
        speech_trainer.encoder.eval()
        eeg_emb_train = eeg_encoder(eeg_X_train.to(device)).cpu()
        eeg_emb_val = eeg_encoder(eeg_X_val.to(device)).cpu()
        sp_emb_train = speech_trainer.encoder(sp_X_train.to(device)).cpu()
        sp_emb_val = speech_trainer.encoder(sp_X_val.to(device)).cpu()

    # Align lengths
    n_train = min(len(eeg_emb_train), len(sp_emb_train))
    n_val = min(len(eeg_emb_val), len(sp_emb_val))

    fusion_trainer = FusionTrainer(cfg)
    fusion_history = fusion_trainer.fit(
        eeg_emb_train[:n_train], sp_emb_train[:n_train], eeg_y_train[:n_train],
        val_eeg_emb=eeg_emb_val[:n_val],
        val_speech_emb=sp_emb_val[:n_val],
        val_labels=eeg_y_val[:n_val],
        save_dir=Path(paths["checkpoints"]) / "fusion",
    )
    results["fusion_warmup"] = fusion_history

    # ----- 6. RL-augmented fusion -----
    logger.info("[6/6] RL-augmented fusion training ...")
    rl_trainer = RLTrainer(
        cfg,
        gan=gan_trainer.gan,
        eeg_encoder=eeg_encoder,
        speech_encoder=speech_trainer.encoder,
        fusion=fusion_trainer.fusion,
    )
    rl_history = rl_trainer.train(
        eeg_X_train, eeg_y_train,
        sp_X_train, sp_y_train,
        eeg_X_val, eeg_y_val,
        sp_X_val, sp_y_val,
        save_dir=Path(paths["checkpoints"]) / "rl",
    )
    results["rl"] = rl_history

    # ----- Save final results -----
    results_path = Path(paths["outputs"]) / "training_results.pt"
    torch.save(results, results_path)
    logger.info("Results saved to %s", results_path)
    logger.info("Pipeline complete!")

    return results
