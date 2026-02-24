"""RL v2 Trainer — Improved PPO agent for v3 pipeline.

Key improvements over v1 RL trainer:
1. **Composite reward**: val_acc gain + macro F1 + class balance - overfit penalty
2. **Advantage normalization**: stabilises PPO updates
3. **Entropy regularization**: better exploration
4. **Multi-step warm-up**: trains fusion for 5 epochs before RL starts
5. **Cosine annealing LR for fusion**: aids convergence
6. **Works with TransformerFusionClassifier**: replaces FusionClassifier
7. **Label smoothing in focal loss**: reduces overconfidence
8. **Better reward smoothing**: Welford running mean + std normalization
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.rl.environment import AugmentationEnv
from src.rl.ppo_agent import PPOAgent
from src.models.gan import ConditionalGAN
from src.models.eeg_encoder import EEGEncoder
from src.models.speech_encoder import SpeechEncoder
from src.models.transformer_fusion import TransformerFusionClassifier
from src.training.fusion_trainer import LabelAlignedDataset, FocalLoss
from src.utils.device import get_device, log_gpu_memory
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


# ======================================================================
# Improved Reward Function
# ======================================================================

def composite_reward(
    val_acc: float,
    prev_val_acc: float,
    train_acc: float,
    class_f1s: list[float] | np.ndarray,
    augmentation_ratio: float,
    stale_count: int = 0,
) -> float:
    """Composite reward with multiple terms.

    Components:
    1. Accuracy delta (improvement signal)
    2. Macro F1 bonus (overall quality)
    3. Class balance: reward min F1, penalize std(F1)
    4. Overfitting penalty: train_acc - val_acc gap
    5. Ratio cost: discourage extreme augmentation
    6. Staleness penalty: no improvement → negative
    """
    class_f1s = np.asarray(class_f1s, dtype=np.float64)

    # 1. Accuracy improvement
    delta_acc = val_acc - prev_val_acc

    # 2. Macro F1 bonus (reward absolute quality, not just delta)
    macro_f1 = float(class_f1s.mean()) if len(class_f1s) > 0 else 0.0
    f1_bonus = 0.3 * macro_f1

    # 3. Class balance
    min_f1 = float(class_f1s.min()) if len(class_f1s) > 0 else 0.0
    std_f1 = float(class_f1s.std()) if len(class_f1s) > 0 else 0.0
    balance = 0.5 * (min_f1 - std_f1)

    # 4. Overfitting penalty
    overfit_gap = max(0.0, train_acc - val_acc - 0.05)  # Allow 5% gap
    overfit_penalty = 0.3 * overfit_gap

    # 5. Ratio cost
    ratio_cost = 0.03 * augmentation_ratio

    # 6. Staleness penalty
    stale_penalty = 0.05 * stale_count

    reward = delta_acc + f1_bonus + balance - overfit_penalty - ratio_cost - stale_penalty
    return float(reward)


# ======================================================================
# Running Stats for Reward Normalization
# ======================================================================

class RunningStats:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(float(np.sqrt(self.M2 / (self.n - 1))), 1e-8)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


# ======================================================================
# RL v2 Trainer
# ======================================================================

class RLv2Trainer:
    """Improved RL-augmented fusion training for v3 pipeline.

    Uses TransformerFusionClassifier instead of FusionClassifier.
    """

    def __init__(
        self,
        cfg: DictConfig,
        gan: ConditionalGAN,
        eeg_encoder: EEGEncoder,
        speech_encoder: SpeechEncoder,
        fusion: TransformerFusionClassifier,
    ) -> None:
        self.cfg = cfg
        self.device = get_device()

        self.gan = gan
        self.gan.device = self.device
        self.gan.generator.to(self.device)
        self.gan.discriminator.to(self.device)
        self.eeg_encoder = eeg_encoder.to(self.device)
        self.speech_encoder = speech_encoder.to(self.device)
        self.fusion = fusion.to(self.device)

        rcfg = cfg.rl
        self.env = AugmentationEnv(
            obs_dim=rcfg.obs_dim,
            max_ratio=rcfg.max_ratio,
            max_steps=rcfg.get("max_steps_v2", rcfg.max_steps),
        )

        # v3 RL hyperparameters
        v3 = getattr(cfg, "v3", {})
        rl_v2 = v3.get("rl_v2", {}) if isinstance(v3, dict) else getattr(v3, "rl_v2", {})

        lr_actor = rl_v2.get("lr_actor", 3e-5) if isinstance(rl_v2, dict) else getattr(rl_v2, "lr_actor", 3e-5)
        entropy_coeff = rl_v2.get("entropy_coeff", 0.02) if isinstance(rl_v2, dict) else getattr(rl_v2, "entropy_coeff", 0.02)

        self.agent = PPOAgent(
            obs_dim=rcfg.obs_dim,
            hidden_dim=rcfg.hidden_dim,
            lr_actor=lr_actor,
            lr_critic=rcfg.lr_critic,
            gamma=rcfg.gamma,
            gae_lambda=rcfg.gae_lambda,
            clip_eps=rcfg.clip_eps,
            entropy_coeff=entropy_coeff,
            update_epochs=rcfg.update_epochs,
            device=str(self.device),
        )

        self.ppo_update_every = rcfg.get("ppo_update_every", 3)
        self.fusion_lr = rl_v2.get("fusion_lr", 5e-4) if isinstance(rl_v2, dict) else getattr(rl_v2, "fusion_lr", 5e-4)
        self.num_classes = cfg.model.num_classes
        self.warmup_epochs = rl_v2.get("warmup_epochs", 5) if isinstance(rl_v2, dict) else getattr(rl_v2, "warmup_epochs", 5)

        # Bounded augmentation ratio
        self.min_ratio = rcfg.get("min_ratio", 0.2)
        self.max_ratio_bound = rcfg.get("max_ratio_bound", 0.6)

        # Reward normalization
        self.reward_stats = RunningStats()

    def _encode_batched(
        self, encoder: nn.Module, data: torch.Tensor, batch_size: int = 512
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size].to(self.device)
            parts.append(encoder(chunk).cpu())
        return torch.cat(parts, dim=0)

    def _build_criterion(
        self,
        eeg_labels: torch.Tensor,
        speech_labels: torch.Tensor,
    ) -> FocalLoss:
        """Build FocalLoss with dynamic class weights."""
        from collections import Counter
        all_labels = torch.cat([eeg_labels, speech_labels])
        counts = Counter(all_labels.numpy().tolist())
        total = sum(counts.values())
        n_cls = max(counts.keys()) + 1
        weights = torch.zeros(n_cls)
        for cls, cnt in counts.items():
            weights[cls] = total / (n_cls * cnt)
        return FocalLoss(gamma=2.0, weight=weights.to(self.device), label_smoothing=0.1)

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
        """Run RL-augmented transformer fusion training."""
        history: dict[str, list] = {
            "val_acc": [], "val_loss": [], "train_acc": [],
            "aug_ratio": [], "reward": [], "macro_f1": [],
            "policy_loss": [], "value_loss": [],
        }

        # Freeze GAN & encoders
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

        # Pre-encode speech (fixed)
        with torch.no_grad():
            sp_emb_train = self._encode_batched(self.speech_encoder, speech_train)
            sp_emb_val = self._encode_batched(self.speech_encoder, speech_val)
            eeg_emb_val = self._encode_batched(self.eeg_encoder, eeg_val)

        criterion = self._build_criterion(eeg_labels, speech_labels)

        fusion_opt = torch.optim.AdamW(
            self.fusion.parameters(), lr=self.fusion_lr, weight_decay=1e-4,
        )
        fusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            fusion_opt, T_max=self.env.max_steps, eta_min=1e-5,
        )

        # Warm-up: train fusion without RL for a few epochs
        if self.warmup_epochs > 0:
            logger.info("RL v2: Warming up fusion for %d epochs...", self.warmup_epochs)
            for wu_ep in range(self.warmup_epochs):
                self._fusion_epoch(
                    eeg_train, eeg_labels,
                    sp_emb_train, speech_labels,
                    None, None,  # no synthetic data
                    fusion_opt, criterion,
                )
            logger.info("RL v2: Warm-up complete.")

        obs, _ = self.env.reset()
        prev_val_acc = 0.0
        stale_count = 0
        best_val_acc = 0.0
        best_macro_f1 = 0.0

        for step in range(1, self.env.max_steps + 1):
            action, log_prob, value = self.agent.select_action(obs)
            ratio = float(np.clip(action, self.min_ratio, self.max_ratio_bound))

            # Generate synthetic EEG
            n_synthetic = max(1, int(ratio * len(eeg_train)))
            syn_labels = eeg_labels[
                torch.randint(0, len(eeg_labels), (n_synthetic,))
            ].to(self.device)
            syn_features = self.gan.generate_from_labels(syn_labels).detach()

            # Train fusion one epoch
            train_loss = self._fusion_epoch(
                eeg_train, eeg_labels,
                sp_emb_train, speech_labels,
                syn_features, syn_labels,
                fusion_opt, criterion,
            )

            # Compute train accuracy for overfit detection
            train_acc = self._quick_accuracy(
                eeg_train[:1000], eeg_labels[:1000],
                sp_emb_train[:1000], speech_labels[:1000],
            )

            # Evaluate
            val_acc, val_loss, class_f1s = self._evaluate(
                eeg_emb_val, eeg_val_labels,
                sp_emb_val, speech_val_labels,
                criterion,
            )
            macro_f1 = float(np.mean(class_f1s))

            # Composite reward
            raw_reward = composite_reward(
                val_acc, prev_val_acc, train_acc,
                class_f1s, ratio, stale_count,
            )
            self.reward_stats.update(raw_reward)
            normalized_reward = self.reward_stats.normalize(raw_reward)

            # Staleness
            if val_acc <= prev_val_acc:
                stale_count += 1
            else:
                stale_count = 0

            # RL bookkeeping
            done = step >= self.env.max_steps
            self.env.set_observation(
                np.array(
                    [val_acc, val_loss] + list(class_f1s) + [ratio, step / self.env.max_steps],
                    dtype=np.float32,
                )
            )
            self.env.set_reward(normalized_reward)
            self.env.set_done(done)

            next_obs, _, terminated, truncated, _ = self.env.step(np.array([ratio]))
            self.agent.buffer.add(obs, action, log_prob, normalized_reward, value, done)
            obs = next_obs

            # PPO update
            ppo_info = {"policy_loss": 0.0, "value_loss": 0.0}
            if step % self.ppo_update_every == 0 or terminated or truncated:
                ppo_info = self.agent.update()

            fusion_scheduler.step()

            # Record
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["aug_ratio"].append(ratio)
            history["reward"].append(normalized_reward)
            history["macro_f1"].append(macro_f1)
            history["policy_loss"].append(ppo_info["policy_loss"])
            history["value_loss"].append(ppo_info["value_loss"])

            # Save best by macro F1 (more robust than accuracy)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_val_acc = val_acc
                if save_dir:
                    save_checkpoint(
                        {
                            "fusion": self.fusion.state_dict(),
                            "step": step,
                            "val_acc": val_acc,
                            "macro_f1": macro_f1,
                        },
                        Path(save_dir) / "best_fusion_v3.pt",
                    )

            prev_val_acc = val_acc
            logger.info(
                "RL-v2 %d/%d  ratio=%.3f  val_acc=%.3f  F1=%.3f  "
                "train_acc=%.3f  reward=%.3f  best_F1=%.3f",
                step, self.env.max_steps, ratio, val_acc, macro_f1,
                train_acc, normalized_reward, best_macro_f1,
            )

            if terminated or truncated:
                break

        # Save agent
        if save_dir:
            save_checkpoint(
                self.agent.state_dict(),
                Path(save_dir) / "ppo_agent_v2_final.pt",
            )

        logger.info(
            "RL v2 complete — best val_acc=%.4f, best macro_F1=%.4f",
            best_val_acc, best_macro_f1,
        )
        return history

    def _fusion_epoch(
        self,
        eeg_feat: torch.Tensor,
        eeg_lbl: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_lbl: torch.Tensor,
        syn_feat: torch.Tensor | None,
        syn_lbl: torch.Tensor | None,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """One fusion training epoch with label-aligned pairing."""
        self.fusion.train()

        # Combine real + synthetic EEG
        if syn_feat is not None and syn_lbl is not None:
            all_eeg = torch.cat([eeg_feat, syn_feat.cpu()], dim=0)
            all_eeg_lbl = torch.cat([eeg_lbl, syn_lbl.cpu()], dim=0)
        else:
            all_eeg = eeg_feat
            all_eeg_lbl = eeg_lbl

        # Encode EEG in batches
        with torch.no_grad():
            eeg_emb = self._encode_batched(self.eeg_encoder, all_eeg)

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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _quick_accuracy(
        self,
        eeg_feat: torch.Tensor,
        eeg_lbl: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_lbl: torch.Tensor,
    ) -> float:
        """Fast train accuracy on a subset for overfitting detection."""
        self.fusion.eval()
        eeg_emb = self._encode_batched(self.eeg_encoder, eeg_feat)
        ds = LabelAlignedDataset(
            eeg_emb, eeg_lbl, speech_emb, speech_lbl,
            num_classes=self.num_classes, balance_classes=False,
        )
        loader = DataLoader(ds, batch_size=512)
        correct, total = 0, 0
        for eeg_b, sp_b, lbl_b in loader:
            logits = self.fusion(
                eeg_b.to(self.device), sp_b.to(self.device)
            )
            correct += (logits.argmax(1).cpu() == lbl_b).sum().item()
            total += len(lbl_b)
        self.fusion.train()
        return correct / max(total, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        eeg_emb: torch.Tensor,
        eeg_lbl: torch.Tensor,
        speech_emb: torch.Tensor,
        speech_lbl: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[float, float, list[float]]:
        """Evaluate fusion on label-aligned validation data."""
        self.fusion.eval()
        ds = LabelAlignedDataset(
            eeg_emb, eeg_lbl, speech_emb, speech_lbl,
            num_classes=self.num_classes, balance_classes=False,
        )
        loader = DataLoader(ds, batch_size=512)

        all_preds, all_labels = [], []
        total_loss, total = 0.0, 0

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

        f1s: list[float] = []
        for c in range(self.num_classes):
            tp = ((preds == c) & (labels == c)).sum()
            fp = ((preds == c) & (labels != c)).sum()
            fn = ((preds != c) & (labels == c)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1s.append(float(f1))

        return acc, loss, f1s
