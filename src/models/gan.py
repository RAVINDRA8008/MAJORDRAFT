"""Conditional WGAN-GP for EEG data augmentation (v4).

Wasserstein GAN with gradient penalty replaces standard BCE-based cGAN
for more stable training and higher quality synthetic data.

Key changes from v3:
- Wasserstein loss (no BCE, no Sigmoid in D)
- Gradient penalty (λ=10) for Lipschitz constraint
- Spectral normalisation on D for additional stability
- n_critic=5 (more D updates per G update)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

logger = logging.getLogger(__name__)


# ======================================================================
# Generator
# ======================================================================
class Generator(nn.Module):
    """WGAN-GP generator: noise *z* + class label → synthetic DE feature."""

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
        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)
        return self.model(x)


# ======================================================================
# Critic (Discriminator without Sigmoid for WGAN)
# ======================================================================
class Discriminator(nn.Module):
    """WGAN-GP critic: feature + class label → real/fake score (no Sigmoid)."""

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
                nn.utils.spectral_norm(nn.Linear(in_dim, h_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
            ])
            in_dim = h_dim

        # No Sigmoid — WGAN uses raw critic scores
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        x = torch.cat([features, label_emb], dim=1)
        return self.model(x)


# ======================================================================
# Wrapper
# ======================================================================
class ConditionalGAN:
    """High-level WGAN-GP wrapper for Generator + Critic training."""

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

        # WGAN-GP hyperparameters
        self.gp_lambda: float = config.get("gp_lambda", 10.0)
        self.n_critic: int = config.get("n_critic", 5)
        self.d_updates_per_g: int = config.get("d_updates_per_g", self.n_critic)
        self.label_smooth: float = config.get("label_smooth", 1.0)  # not used in WGAN-GP

        # Keep criterion for backward compat but we use Wasserstein loss
        self.criterion = None

        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        logger.info("Generator: %s params | Critic: %s params", f"{g_params:,}", f"{d_params:,}")

    def _gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        B = real.size(0)
        alpha = torch.rand(B, 1, device=self.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_inter = self.discriminator(interpolated, labels)
        gradients = autograd.grad(
            outputs=d_inter,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_inter),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    # ------------------------------------------------------------------
    # Single training step — WGAN-GP
    # ------------------------------------------------------------------
    def train_step(
        self, real_features: torch.Tensor, real_labels: torch.Tensor
    ) -> dict[str, float]:
        """One WGAN-GP training step.

        Returns:
            Dictionary with ``d_loss``, ``g_loss``, ``d_real_acc``.
        """
        batch_size = real_features.size(0)

        # --- Critic update(s) ---
        d_loss_total = 0.0
        for _ in range(self.d_updates_per_g):
            self.opt_d.zero_grad()

            # Real
            d_real = self.discriminator(real_features, real_labels)

            # Fake
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z, real_labels).detach()
            d_fake = self.discriminator(fake, real_labels)

            # Gradient penalty
            gp = self._gradient_penalty(real_features, fake, real_labels)

            # Wasserstein loss: maximize E[D(real)] - E[D(fake)]
            # => minimize -E[D(real)] + E[D(fake)] + λ * GP
            d_loss = -d_real.mean() + d_fake.mean() + self.gp_lambda * gp
            d_loss.backward()
            self.opt_d.step()
            d_loss_total += d_loss.item()

        # --- Generator update ---
        self.opt_g.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake = self.generator(z, real_labels)
        d_decision = self.discriminator(fake, real_labels)
        g_loss = -d_decision.mean()  # maximize E[D(fake)]
        g_loss.backward()
        self.opt_g.step()

        # Pseudo-accuracy: what fraction of real scores > fake scores
        with torch.no_grad():
            d_real_score = self.discriminator(real_features, real_labels).mean().item()
            d_real_acc = 1.0 if d_real_score > 0 else 0.0

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

    # ------------------------------------------------------------------
    # Serialisation helpers (mimic nn.Module interface)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Return combined state dict for generator, discriminator & optimisers."""
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load a state dict produced by :meth:`state_dict`.

        Also accepts a *flat* generator-only state dict for backward
        compatibility (e.g. if only the generator was saved).
        """
        if "generator" in state:
            self.generator.load_state_dict(state["generator"])
            self.discriminator.load_state_dict(state["discriminator"])
            if "opt_g" in state:
                self.opt_g.load_state_dict(state["opt_g"])
            if "opt_d" in state:
                self.opt_d.load_state_dict(state["opt_d"])
        else:
            # Legacy: flat state dict is generator-only
            self.generator.load_state_dict(state)

    # ------------------------------------------------------------------
    # nn.Module-like interface (needed by RLTrainer and other callers)
    # ------------------------------------------------------------------
    def to(self, device: torch.device | str) -> "ConditionalGAN":
        """Move generator and discriminator to *device*."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        return self

    def train(self, mode: bool = True) -> "ConditionalGAN":
        """Set generator and discriminator to training mode."""
        self.generator.train(mode)
        self.discriminator.train(mode)
        return self

    def eval(self) -> "ConditionalGAN":
        """Set generator and discriminator to evaluation mode."""
        return self.train(False)

    def parameters(self):
        """Yield all parameters from generator and discriminator."""
        yield from self.generator.parameters()
        yield from self.discriminator.parameters()

    def generate_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Generate one synthetic sample per label in *labels*.

        Args:
            labels: ``(N,)`` integer class-label tensor.

        Returns:
            ``(N, feature_dim)`` synthetic features on the same device.
        """
        self.generator.eval()
        z = torch.randn(len(labels), self.latent_dim, device=labels.device)
        fake = self.generator(z, labels)
        self.generator.train()
        return fake
