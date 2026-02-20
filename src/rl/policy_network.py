"""Policy (Actor) network for PPO.

Outputs a Beta distribution over [0, max_ratio] for the augmentation
ratio, ensuring bounded continuous actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta


class PolicyNetwork(nn.Module):
    """Actor network — maps observation to a Beta distribution.

    Using a Beta distribution keeps the action naturally bounded in
    (0, 1), which we scale to [0, max_ratio] externally.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Two heads for Beta distribution parameters (alpha, beta)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensure > 0
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, obs: torch.Tensor) -> Beta:
        """Return a ``Beta`` distribution.

        Args:
            obs: ``(batch, obs_dim)``

        Returns:
            ``torch.distributions.Beta``
        """
        h = self.shared(obs)
        # +1 to keep alpha, beta ≥ 1 → unimodal distribution
        alpha = self.alpha_head(h) + 1.0
        beta = self.beta_head(h) + 1.0
        return Beta(alpha, beta)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (or take mode of) an action and return log-prob.

        Returns:
            ``(action, log_prob)`` each of shape ``(batch, 1)``
        """
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-prob and entropy for a given obs–action pair.

        Used during PPO update.
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
