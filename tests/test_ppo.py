"""Tests for PPO agent."""

import numpy as np
import pytest

from src.rl.ppo_agent import PPOAgent


@pytest.fixture
def agent():
    return PPOAgent(obs_dim=8, hidden_dim=32, device="cpu")


def test_select_action(agent):
    obs = np.random.randn(8).astype(np.float32)
    action, log_prob, value = agent.select_action(obs)
    assert 0.0 <= action <= 1.0
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_update_empty_buffer(agent):
    info = agent.update()
    assert info["policy_loss"] == 0.0


def test_update_with_data(agent):
    obs = np.random.randn(8).astype(np.float32)
    for _ in range(10):
        action, lp, val = agent.select_action(obs)
        agent.buffer.add(obs, action, lp, 1.0, val, False)
        obs = np.random.randn(8).astype(np.float32)

    info = agent.update()
    assert "policy_loss" in info
    assert "value_loss" in info
    assert len(agent.buffer) == 0  # buffer should be cleared


def test_state_dict_roundtrip(agent):
    sd = agent.state_dict()
    agent2 = PPOAgent(obs_dim=8, hidden_dim=32, device="cpu")
    agent2.load_state_dict(sd)
    # Should not raise
