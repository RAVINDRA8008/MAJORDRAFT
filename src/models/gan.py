"""Conditional GAN for EEG data augmentation.

Generator produces synthetic differential-entropy feature vectors
conditioned on an emotion class label.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Generator
# ======================================================================
class Generator(nn.Module):
    """cGAN generator: noise *z* + class label → synthetic DE feature."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 4,
        feature_dim: int = 160,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        layers: list[nn.Module] = []
        in_dim = latent_dim + num_classes
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, feature_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noise vector ``(batch, latent_dim)``
            labels: Class labels ``(batch,)`` — integer values

        Returns:
            Synthetic features ``(batch, feature_dim)``
        """
        label_emb = self.label_embedding(labels)  # (B, num_classes)
        x = torch.cat([z, label_emb], dim=1)
        return self.model(x)


# ======================================================================
# Discriminator
# ======================================================================
class Discriminator(nn.Module):
    """cGAN discriminator: feature + class label → real/fake score."""

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 160,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        layers: list[nn.Module] = []
        in_dim = feature_dim + num_classes
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Real or synthetic features ``(batch, feature_dim)``
            labels: Class labels ``(batch,)``

        Returns:
            Probability of being real ``(batch, 1)``
        """
        label_emb = self.label_embedding(labels)
        x = torch.cat([features, label_emb], dim=1)
        return self.model(x)


# ======================================================================
# Wrapper
# ======================================================================
class ConditionalGAN:
    """High-level wrapper managing Generator + Discriminator training."""

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device = device
        self.latent_dim: int = config.get("latent_dim", 100)
        feature_dim: int = config.get("feature_dim", 160)
        num_classes: int = config.get("num_classes", 4)

        g_hidden = config.get("generator_hidden_dims", [256, 512, 256])
        d_hidden = config.get("discriminator_hidden_dims", [256, 512, 256])

        self.generator = Generator(
            latent_dim=self.latent_dim,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dims=g_hidden,
        ).to(device)

        self.discriminator = Discriminator(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dims=d_hidden,
        ).to(device)

        lr = config.get("lr", 0.0002)
        beta1 = config.get("beta1", 0.5)
        beta2 = config.get("beta2", 0.999)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )

        self.criterion = nn.BCELoss()
        self.label_smooth: float = config.get("label_smooth", 0.9)
        self.d_updates_per_g: int = config.get("d_updates_per_g", 1)

        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        logger.info("Generator: %s params | Discriminator: %s params", f"{g_params:,}", f"{d_params:,}")

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------
    def train_step(
        self, real_features: torch.Tensor, real_labels: torch.Tensor
    ) -> dict[str, float]:
        """One training step (optionally multiple D updates per G update).

        Returns:
            Dictionary with ``d_loss``, ``g_loss``, ``d_real_acc``.
        """
        batch_size = real_features.size(0)
        real_target = torch.full((batch_size, 1), self.label_smooth, device=self.device)
        fake_target = torch.zeros(batch_size, 1, device=self.device)

        # --- Discriminator update(s) ---
        d_loss_total = 0.0
        for _ in range(self.d_updates_per_g):
            self.opt_d.zero_grad()

            # Real
            d_real = self.discriminator(real_features, real_labels)
            loss_real = self.criterion(d_real, real_target)

            # Fake
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z, real_labels).detach()
            d_fake = self.discriminator(fake, real_labels)
            loss_fake = self.criterion(d_fake, fake_target)

            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            self.opt_d.step()
            d_loss_total += d_loss.item()

        # --- Generator update ---
        self.opt_g.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake = self.generator(z, real_labels)
        d_decision = self.discriminator(fake, real_labels)
        g_loss = self.criterion(d_decision, real_target)  # fool D
        g_loss.backward()
        self.opt_g.step()

        # Accuracy of D on real samples
        d_real_acc = (d_real > 0.5).float().mean().item()

        return {
            "d_loss": d_loss_total / self.d_updates_per_g,
            "g_loss": g_loss.item(),
            "d_real_acc": d_real_acc,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, num_samples: int, class_label: int) -> torch.Tensor:
        """Generate *num_samples* synthetic features for a given class.

        Returns:
            ``(num_samples, feature_dim)`` tensor on ``self.device``.
        """
        self.generator.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        labels = torch.full((num_samples,), class_label, dtype=torch.long, device=self.device)
        fake = self.generator(z, labels)
        self.generator.train()
        return fake

    @torch.no_grad()
    def generate_augmentation(
        self,
        real_features: torch.Tensor,
        real_labels: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate augmented data and concatenate with real data.

        Args:
            real_features: ``(N, feature_dim)``
            real_labels: ``(N,)``
            ratio: Augmentation ratio (1.0 = add N synthetic samples).

        Returns:
            ``(combined_features, combined_labels)``
        """
        if ratio <= 0.0:
            return real_features, real_labels

        n_synthetic = int(len(real_features) * ratio)
        unique_classes = real_labels.unique()
        n_per_class = max(1, n_synthetic // len(unique_classes))

        syn_features_list = []
        syn_labels_list = []
        for cls in unique_classes:
            syn = self.generate(n_per_class, cls.item())
            syn_features_list.append(syn)
            syn_labels_list.append(
                torch.full((n_per_class,), cls.item(), dtype=torch.long, device=self.device)
            )

        syn_features = torch.cat(syn_features_list, dim=0)
        syn_labels = torch.cat(syn_labels_list, dim=0)

        combined_features = torch.cat([real_features, syn_features], dim=0)
        combined_labels = torch.cat([real_labels, syn_labels], dim=0)

        return combined_features, combined_labels
