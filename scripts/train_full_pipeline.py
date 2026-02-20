#!/usr/bin/env python3
"""Run the full end-to-end AMERS training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.training.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Full AMERS training pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.overrides)
    results = run_pipeline(cfg)
    print("Pipeline finished. Keys:", list(results.keys()))


if __name__ == "__main__":
    main()
