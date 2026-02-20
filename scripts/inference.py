#!/usr/bin/env python3
"""Run inference on new EEG + speech input.

Loads exported (or checkpoint) models and predicts emotion class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.paths import get_paths
from src.utils.device import get_device
from src.data.eeg_preprocessor import EEGPreprocessor
from src.data.speech_preprocessor import SpeechPreprocessor
from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier

LABEL_NAMES = ["Happy", "Sad", "Angry", "Neutral"]


def main() -> None:
    parser = argparse.ArgumentParser(description="AMERS inference")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--eeg-file", required=True, help="Path to EEG .npy file (feature vector)")
    parser.add_argument("--wav-file", required=True, help="Path to speech .wav file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    device = get_device()
    ckpt = Path(paths["checkpoints"])

    # --- Preprocess speech ---
    speech_config = {
        "target_sr": cfg.data.iemocap.sr,
        "n_mfcc": cfg.data.iemocap.n_mfcc,
    }
    sp_proc = SpeechPreprocessor(speech_config)
    speech_feat = sp_proc.process_utterance(args.wav_file)
    speech_tensor = torch.as_tensor(speech_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Load EEG feature ---
    eeg_feat = np.load(args.eeg_file)
    eeg_tensor = torch.as_tensor(eeg_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Load models ---
    eeg_enc = EEGEncoder(
        input_dim=cfg.model.eeg_encoder.input_dim,
        hidden_dims=list(cfg.model.eeg_encoder.hidden_dims),
        embedding_dim=cfg.model.eeg_encoder.embedding_dim,
    ).to(device)
    eeg_enc.load_state_dict(torch.load(ckpt / "eeg" / "eeg_encoder_final.pt", map_location=device))
    eeg_enc.eval()

    speech_enc = SpeechEncoder(
        n_features=cfg.model.speech_encoder.n_mfcc,
        embedding_dim=cfg.model.speech_encoder.embedding_dim,
    ).to(device)
    speech_enc.load_state_dict(torch.load(ckpt / "speech" / "speech_encoder_final.pt", map_location=device))
    speech_enc.eval()

    fusion = FusionClassifier(
        eeg_embed_dim=cfg.model.fusion.eeg_dim,
        speech_embed_dim=cfg.model.fusion.speech_dim,
        num_classes=cfg.model.num_classes,
    ).to(device)
    rl_path = ckpt / "rl" / "best_fusion.pt"
    bl_path = ckpt / "fusion" / "best_fusion_baseline.pt"
    for p in [rl_path, bl_path]:
        if p.exists():
            sd = torch.load(p, map_location=device)
            fusion.load_state_dict(sd.get("fusion", sd))
            break
    fusion.eval()

    # --- Predict ---
    with torch.no_grad():
        eeg_emb = eeg_enc(eeg_tensor)
        speech_emb = speech_enc(speech_tensor)
        logits = fusion(eeg_emb, speech_emb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(1).item())

    print(f"\nPredicted emotion: {LABEL_NAMES[pred]}")
    print("Probabilities:")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:8s}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
