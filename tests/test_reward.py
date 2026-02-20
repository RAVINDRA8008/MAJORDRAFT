"""Tests for reward computation."""

import numpy as np
import pytest

from src.rl.reward import compute_reward, shaped_reward


def test_positive_improvement():
    r = compute_reward(0.8, 0.7, [0.8, 0.7, 0.75, 0.8], stale_count=0)
    assert r > 0


def test_no_change_no_stale():
    r = compute_reward(0.7, 0.7, [0.7, 0.7, 0.7, 0.7], stale_count=0)
    # Balance bonus should still contribute
    assert isinstance(r, float)


def test_stale_penalty():
    r0 = compute_reward(0.7, 0.7, [0.7, 0.7, 0.7, 0.7], stale_count=0)
    r5 = compute_reward(0.7, 0.7, [0.7, 0.7, 0.7, 0.7], stale_count=5)
    assert r5 < r0


def test_shaped_reward_cost():
    r_low = shaped_reward(0.8, 0.7, [0.8, 0.8, 0.8, 0.8], 0.1, 0)
    r_high = shaped_reward(0.8, 0.7, [0.8, 0.8, 0.8, 0.8], 1.0, 0)
    assert r_low > r_high  # higher ratio → more cost
