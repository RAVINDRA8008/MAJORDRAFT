from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cmma_fusion import CMMAFusionClassifier
from src.models.transformer_fusion import TransformerFusionClassifier
from src.models.fusion import FusionClassifier
from src.utils.config import load_config

LABELS = ["Angry", "Happy", "Sad", "Neutral"]


class ModelService:
    """Loads best available fusion model for inference.

    Priority:
    1. v5 CMMA checkpoint
    2. v3 Transformer checkpoint
    3. v2/v4 gated-fusion checkpoint
    4. Mock mode fallback (if no checkpoint found)
    """

    def __init__(self, config_path: str | Path = "config/default.yaml") -> None:
        self.device = torch.device("cpu")
        self.cfg = load_config(config_path)

        ckpt_dir_env = os.getenv("AMERS_CHECKPOINT_DIR", "")
        if ckpt_dir_env:
            self.ckpt_root = Path(ckpt_dir_env)
        else:
            self.ckpt_root = Path.cwd() / "checkpoints"

        self.model: torch.nn.Module | None = None
        self.model_type = "mock"

        self._init_model()

    def _init_model(self) -> None:
        v5_candidates = [
            self.ckpt_root / "v5" / "best_cmma.pt",
            self.ckpt_root / "v5" / "best_v5_cmma.pt",
            self.ckpt_root / "v5" / "best_model.pt",
        ]
        v3_candidates = [
            self.ckpt_root / "v3" / "best_transformer_fusion.pt",
            self.ckpt_root / "v3" / "best_fusion_v3.pt",
        ]
        v2_candidates = [
            self.ckpt_root / "rl" / "best_fusion.pt",
            self.ckpt_root / "fusion" / "best_fusion_baseline.pt",
        ]

        model = self._try_load_v5(v5_candidates)
        if model is not None:
            self.model = model
            self.model_type = "v5-cmma"
            return

        model = self._try_load_v3(v3_candidates)
        if model is not None:
            self.model = model
            self.model_type = "v3-transformer"
            return

        model = self._try_load_v2(v2_candidates)
        if model is not None:
            self.model = model
            self.model_type = "v2-v4-gated"
            return

    def _load_state(self, path: Path) -> dict[str, Any]:
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "fusion" in state:
            return state["fusion"]
        return state

    def _try_load_v5(self, candidates: list[Path]) -> torch.nn.Module | None:
        v5 = getattr(self.cfg, "v5", {})
        getv = (lambda k, d: v5.get(k, d)) if isinstance(v5, dict) else (lambda k, d: getattr(v5, k, d))

        model = CMMAFusionClassifier(
            eeg_embed_dim=self.cfg.model.fusion.eeg_dim,
            speech_embed_dim=self.cfg.model.fusion.speech_dim,
            n_tokens=getv("n_tokens", 8),
            d_model=getv("d_model", 128),
            n_heads=getv("n_heads", 4),
            n_cmma_layers=getv("n_cmma_layers", 3),
            ff_dim=getv("ff_dim", 512),
            num_classes=self.cfg.model.num_classes,
            dropout=getv("dropout", 0.15),
            modality_dropout_prob=0.0,
        ).to(self.device)

        for path in candidates:
            if path.exists():
                try:
                    model.load_state_dict(self._load_state(path), strict=False)
                    model.eval()
                    return model
                except Exception:
                    continue
        return None

    def _try_load_v3(self, candidates: list[Path]) -> torch.nn.Module | None:
        tf = getattr(getattr(self.cfg, "v3", {}), "transformer_fusion", {})
        getv = (lambda k, d: tf.get(k, d)) if isinstance(tf, dict) else (lambda k, d: getattr(tf, k, d))

        model = TransformerFusionClassifier(
            eeg_embed_dim=self.cfg.model.fusion.eeg_dim,
            speech_embed_dim=self.cfg.model.fusion.speech_dim,
            n_tokens=getv("n_tokens", 8),
            d_model=getv("d_model", 64),
            n_heads=getv("n_heads", 4),
            n_layers=getv("n_layers", 2),
            ff_dim=getv("ff_dim", 256),
            num_classes=self.cfg.model.num_classes,
            dropout=0.0,
            modality_dropout_prob=0.0,
        ).to(self.device)

        for path in candidates:
            if path.exists():
                try:
                    model.load_state_dict(self._load_state(path), strict=False)
                    model.eval()
                    return model
                except Exception:
                    continue
        return None

    def _try_load_v2(self, candidates: list[Path]) -> torch.nn.Module | None:
        model = FusionClassifier(
            eeg_embed_dim=self.cfg.model.fusion.eeg_dim,
            speech_embed_dim=self.cfg.model.fusion.speech_dim,
            hidden_dims=list(self.cfg.model.fusion.hidden_dims),
            num_classes=self.cfg.model.num_classes,
            dropout=self.cfg.model.fusion.dropout,
            modality_dropout_prob=0.0,
        ).to(self.device)

        for path in candidates:
            if path.exists():
                try:
                    model.load_state_dict(self._load_state(path), strict=False)
                    model.eval()
                    return model
                except Exception:
                    continue
        return None

    def predict(self, eeg_features: list[float], speech_features: list[float]) -> dict[str, Any]:
        eeg_dim = int(self.cfg.model.fusion.eeg_dim)
        speech_dim = int(self.cfg.model.fusion.speech_dim)

        eeg = np.array(eeg_features, dtype=np.float32)
        speech = np.array(speech_features, dtype=np.float32)

        if eeg.size != eeg_dim:
            raise ValueError(f"eeg_features must have length {eeg_dim}")
        if speech.size != speech_dim:
            raise ValueError(f"speech_features must have length {speech_dim}")

        if self.model is None:
            # deterministic mock fallback for demo continuity
            logits = np.array([
                eeg.mean() - speech.mean(),
                speech.mean(),
                -abs(eeg.mean() - speech.mean()),
                0.2,
            ], dtype=np.float32)
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
        else:
            with torch.no_grad():
                eeg_t = torch.from_numpy(eeg).unsqueeze(0).to(self.device)
                speech_t = torch.from_numpy(speech).unsqueeze(0).to(self.device)
                logits_t = self.model(eeg_t, speech_t)
                probs = torch.softmax(logits_t, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidences = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

        return {
            "model_type": self.model_type,
            "label": LABELS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "confidences": confidences,
        }
