"""Gymnasium environment for adaptive augmentation control.

The RL agent observes classifier performance metrics and decides
what fraction of synthetic (GAN-generated) EEG samples to mix in
for the next training epoch.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AugmentationEnv(gym.Env):
    """Custom Gymnasium environment for GAN augmentation ratio control.

    Observation (Box, shape=(obs_dim,)):
        A vector of recent classification metrics:
        [val_accuracy, val_loss, class_0_f1, class_1_f1, class_2_f1, class_3_f1,
         current_ratio, epoch_frac]

    Action (Box, shape=(1,)):
        Continuous value in [0, max_ratio] representing the fraction of
        synthetic EEG samples to inject during the next training epoch.

    Reward:
        Computed externally via ``reward.py`` and injected through
        :meth:`set_reward`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_dim: int = 8,
        max_ratio: float = 1.0,
        max_steps: int = 50,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.max_ratio = max_ratio
        self.max_steps = max_steps

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=max_ratio, shape=(1,), dtype=np.float32,
        )

        # Internal state
        self._step_count = 0
        self._current_obs = np.zeros(obs_dim, dtype=np.float32)
        self._external_reward: float = 0.0
        self._done = False

    # ------------------------------------------------------------------
    # External hooks (called by the training loop)
    # ------------------------------------------------------------------

    def set_observation(self, obs: np.ndarray) -> None:
        """Inject the latest classifier metrics as the next observation."""
        self._current_obs = np.asarray(obs, dtype=np.float32)

    def set_reward(self, reward: float) -> None:
        """Inject reward computed by ``reward.py``."""
        self._external_reward = float(reward)

    def set_done(self, done: bool) -> None:
        """Mark episode as finished (e.g., training complete)."""
        self._done = done

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._current_obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._external_reward = 0.0
        self._done = False
        return self._current_obs.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step.

        The *actual* training is done outside this env; here we simply
        return the pre-set observation and reward.
        """
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        terminated = self._done

        info: dict[str, Any] = {
            "step": self._step_count,
            "action": float(action[0]),
        }

        return (
            self._current_obs.copy(),
            self._external_reward,
            terminated,
            truncated,
            info,
        )
