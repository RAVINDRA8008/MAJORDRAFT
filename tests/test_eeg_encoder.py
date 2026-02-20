"""Tests for EEG encoder model."""

import torch
import pytest

from src.models.eeg_encoder import EEGEncoder


@pytest.fixture
def encoder():
    return EEGEncoder(input_dim=160, hidden_dims=[256, 128], embedding_dim=128, dropout=0.3)


def test_output_shape(encoder):
    x = torch.randn(8, 160)
    out = encoder(x)
    assert out.shape == (8, 128)


def test_single_sample(encoder):
    x = torch.randn(1, 160)
    out = encoder(x)
    assert out.shape == (1, 128)


def test_eval_mode(encoder):
    encoder.eval()
    x = torch.randn(4, 160)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (4, 128)


def test_gradients_flow(encoder):
    x = torch.randn(4, 160, requires_grad=False)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    for p in encoder.parameters():
        if p.requires_grad:
            assert p.grad is not None
