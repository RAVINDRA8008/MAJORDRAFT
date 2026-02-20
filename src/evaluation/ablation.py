"""Ablation study runner.

Systematically disables components and re-trains to measure
marginal contribution of each module.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Each ablation variant is a (name, config_overrides) pair.
ABLATION_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("full_model", {}),
    ("no_gan_augmentation", {"rl.max_ratio": 0.0}),
    ("no_rl_fixed_ratio", {"rl.max_steps": 0}),
    ("eeg_only", {"model.fusion.modality_dropout": 1.0}),       # drop speech
    ("speech_only", {"model.fusion.eeg_dropout_only": True}),    # custom flag
    ("no_modality_dropout", {"model.fusion.modality_dropout": 0.0}),
]


def run_ablation(
    train_fn,
    eval_fn,
    base_cfg: DictConfig,
    train_data: dict,
    test_data: dict,
) -> dict[str, dict]:
    """Run all ablation experiments.

    Args:
        train_fn: ``fn(cfg, train_data) -> model``
        eval_fn: ``fn(model, test_data) -> (preds, labels)``
        base_cfg: Base config (will be overridden per variant).
        train_data: Dict with training tensors.
        test_data: Dict with test tensors.

    Returns:
        ``{variant_name: metrics_dict}``
    """
    results: dict[str, dict] = {}

    for name, overrides in ABLATION_VARIANTS:
        logger.info("Ablation: %s", name)
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))

        try:
            model = train_fn(cfg, train_data)
            preds, labels = eval_fn(model, test_data)
            metrics = compute_all_metrics(np.asarray(labels), np.asarray(preds))
            results[name] = metrics
            logger.info(
                "  %s — acc=%.3f  f1=%.3f",
                name, metrics["accuracy"], metrics["f1_macro"],
            )
        except Exception:
            logger.exception("  Ablation '%s' failed:", name)
            results[name] = {"error": True}

    return results


def ablation_summary_table(results: dict[str, dict]) -> str:
    """Format ablation results as a Markdown table."""
    lines = [
        "| Variant | Accuracy | F1 (macro) | F1 (weighted) | Kappa |",
        "|---------|----------|------------|---------------|-------|",
    ]
    for name, m in results.items():
        if m.get("error"):
            lines.append(f"| {name} | ERROR | — | — | — |")
        else:
            lines.append(
                f"| {name} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} "
                f"| {m['f1_weighted']:.4f} | {m['kappa']:.4f} |"
            )
    return "\n".join(lines)
