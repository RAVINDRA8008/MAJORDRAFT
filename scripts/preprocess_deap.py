#!/usr/bin/env python3
"""Pre-process DEAP dataset.

Reads raw .dat (pickle) files per subject from Drive, applies:
  1. Bandpass filter (4-45 Hz)
  2. Epoch segmentation (1 s windows)
  3. Differential-Entropy feature extraction (5 bands × 32 ch = 160-d)
  4. Normalisation (z-score per subject)
  5. Label mapping (valence × arousal → 4 classes)

Saves per-subject .npy files to Drive processed directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import get_paths
from src.utils.seed import set_seed
from src.data.eeg_preprocessor import EEGPreprocessor
from src.data.label_mapper import LabelMapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process DEAP dataset")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config")
    parser.add_argument("--subjects", nargs="*", type=int, default=None,
                        help="Subject IDs (1-32). Default: all")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    paths = get_paths(cfg)

    raw_dir = Path(paths["deap_raw"])
    out_dir = Path(paths["deap_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list(range(1, 33))

    eeg_config = {
        "sampling_rate": cfg.data.deap.sfreq,
        "epoch_length_sec": cfg.data.deap.epoch_sec,
        "filter_order": 5,
    }
    preprocessor = EEGPreprocessor(eeg_config)
    mapper = LabelMapper()

    for subj in subjects:
        fname = f"s{subj:02d}.dat"
        dat_path = raw_dir / fname
        if not dat_path.exists():
            print(f"  SKIP {fname} (not found)")
            continue

        print(f"  Processing {fname} ...", end=" ", flush=True)
        features, labels = preprocessor.process_subject(dat_path, mapper)

        np.save(out_dir / f"s{subj:02d}_features.npy", features)
        np.save(out_dir / f"s{subj:02d}_labels.npy", labels)
        print(f"→ {features.shape[0]} samples, dim={features.shape[1]}")

    print("DEAP pre-processing complete.")


if __name__ == "__main__":
    main()
