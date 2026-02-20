"""Reward computation for the RL augmentation agent.

Reward = Δ accuracy  +  class-balance bonus  −  staleness penalty
"""

from __future__ import annotations

import numpy as np


def compute_reward(
    val_acc: float,
    prev_val_acc: float,
    class_f1s: list[float] | np.ndarray,
    stale_count: int = 0,
    *,
    acc_weight: float = 1.0,
    balance_weight: float = 0.5,
    stale_penalty: float = 0.1,
) -> float:
    """Compute scalar reward for the RL agent.

    Args:
        val_acc: Validation accuracy after latest epoch.
        prev_val_acc: Validation accuracy from the previous epoch.
        class_f1s: Per-class F1 scores (length = num_classes).
        stale_count: Number of consecutive epochs with no improvement.
        acc_weight: Weight for the accuracy-delta term.
        balance_weight: Weight for the class-balance bonus.
        stale_penalty: Per-stale-epoch penalty coefficient.

    Returns:
        Scalar reward (float).
    """
    # 1. Accuracy improvement term
    delta_acc = val_acc - prev_val_acc

    # 2. Class-balance bonus: reward higher minimum F1 (anti-collapse)
    class_f1s = np.asarray(class_f1s, dtype=np.float64)
    min_f1 = float(class_f1s.min()) if len(class_f1s) > 0 else 0.0
    std_f1 = float(class_f1s.std()) if len(class_f1s) > 0 else 0.0
    # Higher min F1 and lower std → better balance
    balance_bonus = min_f1 - std_f1

    # 3. Staleness penalty
    stale = stale_penalty * stale_count

    reward = acc_weight * delta_acc + balance_weight * balance_bonus - stale
    return float(reward)


def shaped_reward(
    val_acc: float,
    prev_val_acc: float,
    class_f1s: list[float] | np.ndarray,
    augmentation_ratio: float,
    stale_count: int = 0,
) -> float:
    """Higher-level reward with augmentation cost shaping.

    Adds a small penalty proportional to augmentation ratio to
    discourage needlessly high synthetic data injection.
    """
    base = compute_reward(
        val_acc, prev_val_acc, class_f1s, stale_count,
    )
    # Small cost for using more synthetic data (encourage efficiency)
    cost = 0.05 * augmentation_ratio
    return base - cost
