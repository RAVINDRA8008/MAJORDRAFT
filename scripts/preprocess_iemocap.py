#!/usr/bin/env python3
"""Pre-process IEMOCAP dataset.

Reads .wav utterances per session from Drive, applies:
  1. Pre-emphasis (0.97)
  2. Silence trimming
  3. MFCC extraction (40 coeffs + Δ + ΔΔ = 120)
  4. Pad / truncate to fixed length (800 frames)
  5. Label mapping (4 classes)

Saves per-session .npy files to Drive processed directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import get_paths
from src.utils.seed import set_seed
from src.data.speech_preprocessor import SpeechPreprocessor
from src.data.label_mapper import LabelMapper


def _parse_iemocap_labels(session_dir: Path) -> dict[str, str]:
    """Parse IEMOCAP evaluation files to get utterance → emotion map.

    Looks for EvalXXX/*.txt files with format:
        [START - END] UTT_ID EMOTION [V, A, D]
    """
    label_map: dict[str, str] = {}
    eval_dir = session_dir / "dialog" / "EmoEvaluation"
    if not eval_dir.exists():
        return label_map

    for txt_file in eval_dir.glob("*.txt"):
        with open(txt_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        utt_id = parts[1].strip()
                        emotion = parts[2].strip()
                        label_map[utt_id] = emotion
    return label_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process IEMOCAP dataset")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config")
    parser.add_argument("--sessions", nargs="*", type=int, default=None,
                        help="Session IDs (1-5). Default: all")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    paths = get_paths(cfg)

    raw_dir = Path(paths["iemocap_raw"])
    out_dir = Path(paths["iemocap_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = args.sessions or list(range(1, 6))

    preprocessor = SpeechPreprocessor(
        sr=cfg.data.iemocap.sr,
        n_mfcc=cfg.data.iemocap.n_mfcc,
        max_len=cfg.data.iemocap.max_len,
        pre_emphasis=cfg.data.iemocap.pre_emphasis,
    )
    mapper = LabelMapper()

    target_emotions = {"hap", "exc", "sad", "ang", "neu"}

    for sess in sessions:
        sess_dir = raw_dir / f"Session{sess}"
        if not sess_dir.exists():
            print(f"  SKIP Session{sess} (not found)")
            continue

        print(f"  Processing Session{sess} ...", end=" ", flush=True)

        label_map = _parse_iemocap_labels(sess_dir)
        wav_dir = sess_dir / "sentences" / "wav"

        features_list: list[np.ndarray] = []
        labels_list: list[int] = []

        for wav_file in sorted(wav_dir.rglob("*.wav")):
            utt_id = wav_file.stem
            emotion = label_map.get(utt_id)
            if emotion not in target_emotions:
                continue
            label = mapper.iemocap_label(emotion)
            if label < 0:
                continue

            feat = preprocessor.extract_features(str(wav_file))
            features_list.append(feat)
            labels_list.append(label)

        if features_list:
            feat_arr = np.stack(features_list)
            lbl_arr = np.array(labels_list, dtype=np.int64)
            np.save(out_dir / f"session{sess}_features.npy", feat_arr)
            np.save(out_dir / f"session{sess}_labels.npy", lbl_arr)
            print(f"→ {len(features_list)} utterances, shape={feat_arr.shape[1:]}")
        else:
            print("→ 0 utterances (check paths)")

    print("IEMOCAP pre-processing complete.")


if __name__ == "__main__":
    main()
