#!/usr/bin/env python3
"""Export trained model to TorchScript for deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.paths import get_paths
from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AMERS model to TorchScript")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    ckpt = Path(paths["checkpoints"])
    out_dir = Path(args.output_dir or paths["outputs"]) / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"

    # EEG encoder
    eeg_enc = EEGEncoder(
        input_dim=cfg.model.eeg_encoder.input_dim,
        hidden_dims=list(cfg.model.eeg_encoder.hidden_dims),
        embedding_dim=cfg.model.eeg_encoder.embedding_dim,
    )
    eeg_path = ckpt / "eeg" / "eeg_encoder_final.pt"
    if eeg_path.exists():
        eeg_enc.load_state_dict(torch.load(eeg_path, map_location=device))
    eeg_enc.eval()
    scripted_eeg = torch.jit.script(eeg_enc)
    scripted_eeg.save(str(out_dir / "eeg_encoder.pt"))
    print(f"Exported EEG encoder → {out_dir / 'eeg_encoder.pt'}")

    # Speech encoder
    speech_enc = SpeechEncoder(
        n_features=cfg.model.speech_encoder.n_mfcc,
        embedding_dim=cfg.model.speech_encoder.embedding_dim,
    )
    sp_path = ckpt / "speech" / "speech_encoder_final.pt"
    if sp_path.exists():
        speech_enc.load_state_dict(torch.load(sp_path, map_location=device))
    speech_enc.eval()
    # Use tracing for speech encoder (dynamic control flow in LSTM)
    dummy_input = torch.randn(1, cfg.data.iemocap.max_len, cfg.data.iemocap.n_mfcc * 3)
    traced_speech = torch.jit.trace(speech_enc, dummy_input)
    traced_speech.save(str(out_dir / "speech_encoder.pt"))
    print(f"Exported speech encoder → {out_dir / 'speech_encoder.pt'}")

    # Fusion
    fusion = FusionClassifier(
        eeg_embed_dim=cfg.model.fusion.eeg_dim,
        speech_embed_dim=cfg.model.fusion.speech_dim,
        num_classes=cfg.model.num_classes,
    )
    rl_path = ckpt / "rl" / "best_fusion.pt"
    bl_path = ckpt / "fusion" / "best_fusion_baseline.pt"
    for p in [rl_path, bl_path]:
        if p.exists():
            sd = torch.load(p, map_location=device)
            fusion.load_state_dict(sd.get("fusion", sd))
            break
    fusion.eval()
    dummy_eeg = torch.randn(1, cfg.model.fusion.eeg_dim)
    dummy_sp = torch.randn(1, cfg.model.fusion.speech_dim)
    traced_fusion = torch.jit.trace(fusion, (dummy_eeg, dummy_sp))
    traced_fusion.save(str(out_dir / "fusion_classifier.pt"))
    print(f"Exported fusion classifier → {out_dir / 'fusion_classifier.pt'}")

    print("\nAll models exported to:", out_dir)


if __name__ == "__main__":
    main()
