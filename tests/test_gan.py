"""Tests for conditional GAN."""

import torch
import pytest

from src.models.gan import ConditionalGAN


@pytest.fixture
def gan():
    return ConditionalGAN(
        feature_dim=160, noise_dim=64, hidden_dim=128,
        num_classes=4, lr_g=1e-4, lr_d=1e-4,
    )


def test_generate_shape(gan):
    labels = torch.tensor([0, 1, 2, 3])
    out = gan.generate(labels)
    assert out.shape == (4, 160)


def test_train_step_returns_losses(gan):
    real = torch.randn(16, 160)
    labels = torch.randint(0, 4, (16,))
    losses = gan.train_step(real, labels)
    assert "g_loss" in losses
    assert "d_loss" in losses
    assert isinstance(losses["g_loss"], float)


def test_generate_augmentation(gan):
    existing_labels = torch.randint(0, 4, (100,))
    feats, lbls = gan.generate_augmentation(existing_labels, ratio=0.5)
    assert len(feats) == 50
    assert len(lbls) == 50
    assert feats.shape[1] == 160
