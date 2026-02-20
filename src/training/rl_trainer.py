"""RL trainer — trains the PPO agent inside the AugmentationEnv.

Each "step" in RL corresponds to one fusion-training epoch:
  1. Agent picks augmentation ratio.
  2. GAN generates synthetic data at that ratio.
  3. Fusion model trains one epoch (EEG + synthetic EEG + speech).
  4. Validation metrics form the observation → reward.
  5. Agent stores transition and eventually does PPO update.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig

from src.rl.environment import AugmentationEnv
from src.rl.ppo_agent import PPOAgent
from src.rl.reward import shaped_reward
from src.models.gan import ConditionalGAN
from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier
from src.utils.device import get_device, log_gpu_memory
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class RLTrainer:
    """Train the PPO augmentation agent in the loop with the fusion model."""

    def __init__(
        self,
        cfg: DictConfig,
        gan: ConditionalGAN,
        eeg_encoder: EEGEncoder,
        speech_encoder: SpeechEncoder,
        fusion: FusionClassifier,
    ) -> None:
        self.cfg = cfg
        self.device = get_device()

        # Pre-trained components (frozen or fine-tuned depending on config)
        self.gan = gan.to(self.device)
        self.eeg_encoder = eeg_encoder.to(self.device)
        self.speech_encoder = speech_encoder.to(self.device)
        self.fusion = fusion.to(self.device)

        rcfg = cfg.rl
        self.env = AugmentationEnv(
            obs_dim=rcfg.obs_dim,
            max_ratio=rcfg.max_ratio,
            max_steps=rcfg.max_steps,
        )
        self.agent = PPOAgent(
            obs_dim=rcfg.obs_dim,
            hidden_dim=rcfg.hidden_dim,
            lr_actor=rcfg.lr_actor,
            lr_critic=rcfg.lr_critic,
            gamma=rcfg.gamma,
            gae_lambda=rcfg.gae_lambda,
            clip_eps=rcfg.clip_eps,
            entropy_coeff=rcfg.entropy_coeff,
            update_epochs=rcfg.update_epochs,
            device=str(self.device),
        )

        self.ppo_update_every = rcfg.get("ppo_update_every", 5)
        self.fusion_lr = cfg.model.fusion.get("lr", 1e-3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        eeg_train: torch.Tensor,
        eeg_labels: torch.Tensor,
        speech_train: torch.Tensor,
        speech_labels: torch.Tensor,
        eeg_val: torch.Tensor,
        eeg_val_labels: torch.Tensor,
        speech_val: torch.Tensor,
        speech_val_labels: torch.Tensor,
        save_dir: str | Path | None = None,
    ) -> dict[str, list]:
        """Run the RL-augmented fusion training loop.

        Returns:
            History dict with keys: ``val_acc``, ``val_loss``,
            ``aug_ratio``, ``reward``, ``policy_loss``, ``value_loss``.
        """
        history: dict[str, list] = {
            "val_acc": [], "val_loss": [],
            "aug_ratio": [], "reward": [],
            "policy_loss": [], "value_loss": [],
        }

        # Freeze GAN generator, EEG & speech encoders
        self.gan.eval()
        for p in self.gan.parameters():
            p.requires_grad = False
        self.eeg_encoder.eval()
        for p in self.eeg_encoder.parameters():
            p.requires_grad = False
        self.speech_encoder.eval()
        for p in self.speech_encoder.parameters():
            p.requires_grad = False

        # Fusion optimizer
        fusion_opt = torch.optim.Adam(self.fusion.parameters(), lr=self.fusion_lr)
        criterion = torch.nn.CrossEntropyLoss()

        obs, _ = self.env.reset()
        prev_val_acc = 0.0
        stale_count = 0
        best_val_acc = 0.0

        for step in range(1, self.env.max_steps + 1):
            # 1. Agent selects augmentation ratio
            action, log_prob, value = self.agent.select_action(obs)
            ratio = float(np.clip(action, 0.0, self.cfg.rl.max_ratio))

            # 2. Generate synthetic EEG data
            n_synthetic = max(1, int(ratio * len(eeg_train)))
            syn_labels = eeg_labels[
                torch.randint(0, len(eeg_labels), (n_synthetic,))
            ].to(self.device)
            syn_features = self.gan.generate(syn_labels).detach()

            # 3. Train fusion for one epoch
            train_loss = self._fusion_epoch(
                eeg_train, eeg_labels,
                speech_train, speech_labels,
                syn_features, syn_labels,
                fusion_opt, criterion,
            )

            # 4. Evaluate
            val_acc, val_loss, class_f1s = self._evaluate(
                eeg_val, eeg_val_labels,
                speech_val, speech_val_labels,
                criterion,
            )

            # 5. Reward
            reward = shaped_reward(
                val_acc, prev_val_acc, class_f1s,
                augmentation_ratio=ratio,
                stale_count=stale_count,
            )

            # Staleness tracking
            if val_acc <= prev_val_acc:
                stale_count += 1
            else:
                stale_count = 0

            # 6. RL bookkeeping
            done = step >= self.env.max_steps
            self.env.set_observation(
                np.array(
                    [val_acc, val_loss] + list(class_f1s) + [ratio, step / self.env.max_steps],
                    dtype=np.float32,
                )
            )
            self.env.set_reward(reward)
            self.env.set_done(done)

            next_obs, _, terminated, truncated, _ = self.env.step(np.array([ratio]))
            self.agent.buffer.add(obs, action, log_prob, reward, value, done)
            obs = next_obs

            # 7. PPO update
            ppo_info = {"policy_loss": 0.0, "value_loss": 0.0}
            if step % self.ppo_update_every == 0 or terminated or truncated:
                ppo_info = self.agent.update()

            # Record
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            history["aug_ratio"].append(ratio)
            history["reward"].append(reward)
            history["policy_loss"].append(ppo_info["policy_loss"])
            history["value_loss"].append(ppo_info["value_loss"])

            # Best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_dir:
                    save_checkpoint(
                        {"fusion": self.fusion.state_dict(), "step": step},
                        Path(save_dir) / "best_fusion.pt",
                    )

            prev_val_acc = val_acc
            logger.info(
                "RL step %d/%d — ratio=%.3f  val_acc=%.3f  reward=%.4f",
                step, self.env.max_steps, ratio, val_acc, reward,
            )

            if terminated or truncated:
                break

        # Save agent
        if save_dir:
            save_checkpoint(
                self.agent.state_dict(),
                Path(save_dir) / "ppo_agent_final.pt",
            )

        logger.info("RL training complete — best val_acc=%.4f", best_val_acc)
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fusion_epoch(
        self,
        eeg_feat: torch.Tensor,
        eeg_lbl: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_lbl: torch.Tensor,
        syn_feat: torch.Tensor,
        syn_lbl: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """One fusion training epoch with real + synthetic EEG."""
        self.fusion.train()

        # Combine real + synthetic EEG
        all_eeg = torch.cat([eeg_feat.to(self.device), syn_feat], dim=0)
        all_eeg_lbl = torch.cat([eeg_lbl.to(self.device), syn_lbl], dim=0)

        # Encode
        with torch.no_grad():
            eeg_emb = self.eeg_encoder(all_eeg)
            # Match speech samples to EEG (simple repeat/truncate)
            sp = speech_feat.to(self.device)
            speech_emb = self.speech_encoder(sp)

        # Align lengths: take min(len(eeg_emb), len(speech_emb))
        n = min(len(eeg_emb), len(speech_emb))
        eeg_emb_batch = eeg_emb[:n]
        speech_emb_batch = speech_emb[:n]
        labels_batch = all_eeg_lbl[:n]

        # Forward
        logits = self.fusion(eeg_emb_batch, speech_emb_batch)
        loss = criterion(logits, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _evaluate(
        self,
        eeg_val: torch.Tensor,
        eeg_val_lbl: torch.Tensor,
        speech_val: torch.Tensor,
        speech_val_lbl: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> tuple[float, float, list[float]]:
        """Evaluate fusion on validation data.

        Returns:
            ``(accuracy, loss, per_class_f1)``
        """
        self.fusion.eval()
        eeg_emb = self.eeg_encoder(eeg_val.to(self.device))
        speech_emb = self.speech_encoder(speech_val.to(self.device))

        n = min(len(eeg_emb), len(speech_emb))
        logits = self.fusion(eeg_emb[:n], speech_emb[:n])
        labels = eeg_val_lbl[:n].to(self.device)

        loss = criterion(logits, labels).item()
        preds = logits.argmax(1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        acc = float((preds == labels_np).mean())

        # Per-class F1 (simple)
        num_classes = self.cfg.model.num_classes
        f1s: list[float] = []
        for c in range(num_classes):
            tp = ((preds == c) & (labels_np == c)).sum()
            fp = ((preds == c) & (labels_np != c)).sum()
            fn = ((preds != c) & (labels_np == c)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            f1s.append(float(f1))

        return acc, loss, f1s
