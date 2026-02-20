"""Evaluation metrics for emotion classification.

All functions accept NumPy arrays of true/predicted labels and return
scalar or dict results.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)


LABEL_NAMES = ["Happy", "Sad", "Angry", "Neutral"]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Compute a comprehensive set of classification metrics.

    Returns a dict with keys:
        accuracy, f1_macro, f1_weighted, precision_macro,
        recall_macro, kappa, per_class_f1, per_class_precision,
        per_class_recall, confusion_matrix, report_str.
    """
    label_names = label_names or LABEL_NAMES
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_precision": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report_str": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0,
        ),
    }


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))
