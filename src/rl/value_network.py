"""Value (Critic) network for PPO.

Estimates V(s) — the expected discounted return from a given
observation state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Critic — maps observation to a scalar state-value estimate."""

    def __init__(
        self,
        obs_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: ``(batch, obs_dim)``

        Returns:
            State value ``(batch, 1)``
        """
        return self.net(obs)
