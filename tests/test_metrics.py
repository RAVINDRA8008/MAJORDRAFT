"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_all_metrics, compute_accuracy, compute_f1


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = compute_all_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_random_predictions():
    y_true = np.array([0, 1, 2, 3] * 10)
    y_pred = np.array([0, 0, 0, 0] * 10)
    m = compute_all_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.25
    assert m["f1_macro"] < 0.5


def test_compute_accuracy():
    assert compute_accuracy(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0


def test_compute_f1():
    f = compute_f1(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
    assert f == 1.0


def test_confusion_matrix_shape():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = compute_all_metrics(y, y)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (4, 4)
