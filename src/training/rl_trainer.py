"""RL trainer — PPO agent with label-aligned fusion training.

Key improvements:
1. **Label-aligned pairing** — EEG and speech are matched by emotion label
2. **Smoothed reward** — exponential moving average of val_acc
3. **Bounded augmentation** — ratio clamped to [0.2, 0.6]
4. **Focal loss** — handles class imbalance in fusion training
5. **Policy updates every N epochs** — reduces noise
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.rl.environment import AugmentationEnv
from src.rl.ppo_agent import PPOAgent
from src.rl.reward import shaped_reward
from src.models.gan import ConditionalGAN
from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.fusion import FusionClassifier
from src.training.fusion_trainer import LabelAlignedDataset, FocalLoss
from src.utils.device import get_device, log_gpu_memory
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class RLTrainer:
    """Train the PPO augmentation agent with label-aligned fusion."""

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

        self.ppo_update_every = rcfg.get("ppo_update_every", 3)
        self.fusion_lr = cfg.model.fusion.get("lr", 1e-3) * 0.5  # lower LR for RL fine-tuning
        self.num_classes = cfg.model.num_classes

        # Bounded augmentation ratio
        self.min_ratio = rcfg.get("min_ratio", 0.2)
        self.max_ratio_bound = rcfg.get("max_ratio_bound", 0.6)

        # Smoothed reward (EMA)
        self.reward_ema = 0.0
        self.reward_alpha = 0.3  # EMA smoothing factor

    # ------------------------------------------------------------------
    # Batched encoding helper
    # ------------------------------------------------------------------
    def _encode_batched(
        self, encoder: torch.nn.Module, data: torch.Tensor, batch_size: int = 512
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size].to(self.device)
            parts.append(encoder(chunk).cpu())
        return torch.cat(parts, dim=0)

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
        """Run RL-augmented fusion training with label-aligned pairing."""
        history: dict[str, list] = {
            "val_acc": [], "val_loss": [],
            "aug_ratio": [], "reward": [],
            "policy_loss": [], "value_loss": [],
        }

        # Freeze GAN, EEG & speech encoders
        self.gan.eval()
        for p in self.gan.generator.parameters():
            p.requires_grad = False
        for p in self.gan.discriminator.parameters():
            p.requires_grad = False
        self.eeg_encoder.eval()
        for p in self.eeg_encoder.parameters():
            p.requires_grad = False
        self.speech_encoder.eval()
        for p in self.speech_encoder.parameters():
            p.requires_grad = False

        # Pre-encode speech (doesn't change) — encode in batches
        with torch.no_grad():
            sp_emb_train = self._encode_batched(self.speech_encoder, speech_train)
            sp_emb_val = self._encode_batched(self.speech_encoder, speech_val)

        # Pre-encode EEG validation (doesn't change)
        with torch.no_grad():
            eeg_emb_val = self._encode_batched(self.eeg_encoder, eeg_val)

        # Focal loss with class weights
        all_labels = torch.cat([eeg_labels, speech_labels])
        from collections import Counter
        counts = Counter(all_labels.numpy().tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            weights[cls] = total / (n_cls * cnt)
        criterion = FocalLoss(gamma=2.0, weight=weights.to(self.device), label_smoothing=0.05)

        # Fusion optimizer
        fusion_opt = torch.optim.AdamW(
            self.fusion.parameters(), lr=self.fusion_lr, weight_decay=1e-4,
        )

        obs, _ = self.env.reset()
        prev_val_acc = 0.0
        stale_count = 0
        best_val_acc = 0.0
        reward_history = deque(maxlen=10)

        for step in range(1, self.env.max_steps + 1):
            # 1. Agent selects augmentation ratio (bounded)
            action, log_prob, value = self.agent.select_action(obs)
            ratio = float(np.clip(action, self.min_ratio, self.max_ratio_bound))

            # 2. Generate synthetic EEG data
            n_synthetic = max(1, int(ratio * len(eeg_train)))
            syn_labels = eeg_labels[
                torch.randint(0, len(eeg_labels), (n_synthetic,))
            ].to(self.device)
            syn_features = self.gan.generate_from_labels(syn_labels).detach()

            # 3. Train fusion for one epoch (label-aligned)
            train_loss = self._fusion_epoch(
                eeg_train, eeg_labels,
                sp_emb_train, speech_labels,
                syn_features, syn_labels,
                fusion_opt, criterion,
            )

            # 4. Evaluate (label-aligned)
            val_acc, val_loss, class_f1s = self._evaluate(
                eeg_emb_val, eeg_val_labels,
                sp_emb_val, speech_val_labels,
                criterion,
            )

            # 5. Smoothed reward
            raw_reward = shaped_reward(
                val_acc, prev_val_acc, class_f1s,
                augmentation_ratio=ratio,
                stale_count=stale_count,
            )
            self.reward_ema = self.reward_alpha * raw_reward + (1 - self.reward_alpha) * self.reward_ema
            reward = self.reward_ema

            reward_history.append(val_acc)

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

            # 7. PPO update (every N steps for stability)
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
                "RL step %d/%d — ratio=%.3f  val_acc=%.3f  reward=%.4f  "
                "smoothed=%.4f  best=%.3f",
                step, self.env.max_steps, ratio, val_acc, raw_reward,
                reward, best_val_acc,
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
        speech_emb: torch.Tensor,
        speech_lbl: torch.Tensor,
        syn_feat: torch.Tensor,
        syn_lbl: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> float:
        """One fusion training epoch with label-aligned pairing."""
        self.fusion.train()

        # Combine real + synthetic EEG
        all_eeg = torch.cat([eeg_feat, syn_feat.cpu()], dim=0)
        all_eeg_lbl = torch.cat([eeg_lbl, syn_lbl.cpu()], dim=0)

        # Encode EEG in batches
        with torch.no_grad():
            eeg_emb = self._encode_batched(self.eeg_encoder, all_eeg)

        # Create label-aligned dataset
        ds = LabelAlignedDataset(
            eeg_emb, all_eeg_lbl, speech_emb, speech_lbl,
            num_classes=self.num_classes, balance_classes=True,
        )
        loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

        total_loss = 0.0
        n_batches = 0
        for eeg_b, sp_b, lbl_b in loader:
            eeg_b = eeg_b.to(self.device)
            sp_b = sp_b.to(self.device)
            lbl_b = lbl_b.to(self.device)

            logits = self.fusion(eeg_b, sp_b)
            loss = criterion(logits, lbl_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        eeg_emb: torch.Tensor,
        eeg_lbl: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_lbl: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> tuple[float, float, list[float]]:
        """Evaluate fusion on label-aligned validation data."""
        self.fusion.eval()

        # Create label-aligned val dataset
        ds = LabelAlignedDataset(
            eeg_emb, eeg_lbl, speech_emb, speech_lbl,
            num_classes=self.num_classes, balance_classes=False,
        )
        loader = DataLoader(ds, batch_size=512)

        all_preds = []
        all_labels = []
        total_loss = 0.0
        total = 0

        for eeg_b, sp_b, lbl_b in loader:
            eeg_b = eeg_b.to(self.device)
            sp_b = sp_b.to(self.device)
            lbl_b = lbl_b.to(self.device)

            logits = self.fusion(eeg_b, sp_b)
            total_loss += criterion(logits, lbl_b).item() * eeg_b.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(lbl_b.cpu())
            total += eeg_b.size(0)

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        acc = float((preds == labels).mean())
        loss = total_loss / max(total, 1)

        # Per-class F1
        f1s: list[float] = []
        for c in range(self.num_classes):
            tp = ((preds == c) & (labels == c)).sum()
            fp = ((preds == c) & (labels != c)).sum()
            fn = ((preds != c) & (labels == c)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            f1s.append(float(f1))

        return acc, loss, f1s
