"""Tests for RL environment."""

import numpy as np
import pytest

from src.rl.environment import AugmentationEnv


@pytest.fixture
def env():
    return AugmentationEnv(obs_dim=8, max_ratio=1.0, max_steps=10)


def test_reset(env):
    obs, info = env.reset()
    assert obs.shape == (8,)
    assert isinstance(info, dict)


def test_step(env):
    env.reset()
    env.set_observation(np.ones(8, dtype=np.float32))
    env.set_reward(0.5)
    action = np.array([0.3], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (8,)
    assert reward == 0.5
    assert not terminated


def test_truncation(env):
    env.reset()
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([0.1]))
    assert truncated
