#!/usr/bin/env python3
"""Verify Colab environment setup.

Run this first to check: GPU availability, library versions,
Drive mount, and dataset paths.
"""

from __future__ import annotations

import importlib
import sys


def check_python() -> None:
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    assert v.major == 3 and v.minor >= 9, "Need Python ≥ 3.9"


def check_gpu() -> None:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected — training will be very slow.")


def check_libraries() -> None:
    required = [
        "torch", "numpy", "scipy", "sklearn", "mne", "librosa",
        "gymnasium", "omegaconf", "tqdm", "tensorboard", "seaborn",
        "matplotlib",
    ]
    for lib in required:
        try:
            mod = importlib.import_module(lib)
            version = getattr(mod, "__version__", "ok")
            print(f"  {lib:20s} {version}")
        except ImportError:
            print(f"  {lib:20s} MISSING ❌")


def check_drive() -> None:
    from pathlib import Path
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        print(f"Drive mounted at {drive}")
    else:
        print("WARNING: Google Drive not mounted.")


def check_datasets() -> None:
    from pathlib import Path
    base = Path("/content/drive/MyDrive/AMERS")
    for name, subdir in [("DEAP", "data/deap/raw"), ("IEMOCAP", "data/iemocap/raw")]:
        p = base / subdir
        if p.exists():
            files = list(p.iterdir())
            print(f"  {name}: {len(files)} items in {p}")
        else:
            print(f"  {name}: NOT FOUND at {p} — run preprocessing first")


def main() -> None:
    print("=" * 50)
    print("AMERS Environment Verification")
    print("=" * 50)

    print("\n[1] Python version")
    check_python()

    print("\n[2] GPU")
    check_gpu()

    print("\n[3] Libraries")
    check_libraries()

    print("\n[4] Google Drive")
    check_drive()

    print("\n[5] Datasets")
    check_datasets()

    print("\n✓ Verification complete.")


if __name__ == "__main__":
    main()
