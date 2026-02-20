"""PPO Agent — orchestrates policy + value networks and GAE updates.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


class RolloutBuffer:
    """Stores trajectory data for one PPO update cycle."""

    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[float] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        obs: np.ndarray,
        action: float,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    """Proximal Policy Optimization agent with clipped objective."""

    def __init__(
        self,
        obs_dim: int = 8,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.device = torch.device(device)

        # Networks
        self.policy = PolicyNetwork(obs_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden_dim).to(self.device)

        # Optimisers
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.value_net.parameters(), lr=lr_critic)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> tuple[float, float, float]:
        """Choose augmentation ratio for current state.

        Returns:
            ``(action, log_prob, value)``
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob = self.policy.get_action(obs_t)
        value = self.value_net(obs_t)
        return (
            float(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = np.zeros(len(rewards), dtype=np.float64)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(values)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        return adv_t, ret_t

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run PPO update using collected rollout data.

        Returns:
            Dict with ``policy_loss``, ``value_loss``, ``entropy``.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages, returns = self._compute_gae()

        # Normalise advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tensors
        obs_t = torch.as_tensor(
            np.array(self.buffer.observations), dtype=torch.float32, device=self.device,
        )
        act_t = torch.as_tensor(
            np.array(self.buffer.actions), dtype=torch.float32, device=self.device,
        ).unsqueeze(-1)
        old_log_prob_t = torch.as_tensor(
            np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device,
        ).unsqueeze(-1)

        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0

        for _ in range(self.update_epochs):
            # Evaluate current policy
            log_prob, entropy = self.policy.evaluate_action(obs_t, act_t)
            ratio = torch.exp(log_prob - old_log_prob_t)

            # Clipped surrogate objective
            adv = advantages.unsqueeze(-1)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = entropy.mean()

            # Actor step
            actor_loss = policy_loss - self.entropy_coeff * entropy_bonus
            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.opt_actor.step()

            # Critic step
            value_pred = self.value_net(obs_t).squeeze(-1)
            value_loss = nn.functional.mse_loss(value_pred, returns)
            self.opt_critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.opt_critic.step()

            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            total_ent += entropy_bonus.item()

        n = self.update_epochs
        self.buffer.clear()

        return {
            "policy_loss": total_pl / n,
            "value_loss": total_vl / n,
            "entropy": total_ent / n,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.value_net.load_state_dict(sd["value_net"])
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_critic.load_state_dict(sd["opt_critic"])
