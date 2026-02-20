"""Leave-One-Subject-Out (LOSO) cross-validation.

DEAP has 32 subjects → 32 folds.
IEMOCAP has 5 sessions → 5 folds.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
from omegaconf import DictConfig

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def loso_cv(
    subjects: list[int],
    features_by_subject: dict[int, np.ndarray],
    labels_by_subject: dict[int, np.ndarray],
    train_and_eval_fn: Callable,
    cfg: DictConfig,
) -> dict:
    """Run Leave-One-Subject-Out cross-validation.

    Args:
        subjects: List of subject/session IDs.
        features_by_subject: ``{subject_id: (N_i, D)}`` feature arrays.
        labels_by_subject: ``{subject_id: (N_i,)}`` label arrays.
        train_and_eval_fn: ``fn(train_X, train_y, test_X, test_y, cfg) -> (preds, metrics)``
            A callable that trains a model on (train_X, train_y) and
            evaluates on (test_X, test_y), returning predictions and
            a metrics dict.
        cfg: Config dict.

    Returns:
        Dict with ``fold_metrics``, ``mean_metrics``, ``std_metrics``,
        ``all_preds``, ``all_labels``.
    """
    fold_metrics: list[dict] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for i, test_subj in enumerate(subjects):
        logger.info("LOSO fold %d/%d — test subject=%d", i + 1, len(subjects), test_subj)

        # Build train/test splits
        train_feats = np.concatenate(
            [features_by_subject[s] for s in subjects if s != test_subj]
        )
        train_lbls = np.concatenate(
            [labels_by_subject[s] for s in subjects if s != test_subj]
        )
        test_feats = features_by_subject[test_subj]
        test_lbls = labels_by_subject[test_subj]

        preds, metrics = train_and_eval_fn(
            train_feats, train_lbls, test_feats, test_lbls, cfg,
        )

        fold_metrics.append(metrics)
        all_preds.append(np.asarray(preds))
        all_labels.append(np.asarray(test_lbls))

        logger.info(
            "  Fold %d — acc=%.3f  f1_macro=%.3f",
            i + 1, metrics["accuracy"], metrics["f1_macro"],
        )

    # Aggregate
    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)
    overall = compute_all_metrics(all_labels_flat, all_preds_flat)

    # Per-fold mean ± std
    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "kappa"]
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys}
    std_metrics = {k: float(np.std([m[k] for m in fold_metrics])) for k in metric_keys}

    logger.info("LOSO complete — mean_acc=%.3f±%.3f  mean_f1=%.3f±%.3f",
                mean_metrics["accuracy"], std_metrics["accuracy"],
                mean_metrics["f1_macro"], std_metrics["f1_macro"])

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "overall_metrics": overall,
        "all_preds": all_preds_flat,
        "all_labels": all_labels_flat,
    }
