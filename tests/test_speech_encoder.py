"""Tests for speech encoder."""

import torch
import pytest

from src.models.speech_encoder import SpeechEncoder


@pytest.fixture
def encoder():
    return SpeechEncoder(
        n_mfcc=120, cnn_channels=[32, 64, 128],
        lstm_hidden=128, lstm_layers=2,
        embedding_dim=128, dropout=0.3,
    )


def test_output_shape(encoder):
    # (batch, time, features)
    x = torch.randn(4, 800, 120)
    out = encoder(x)
    assert out.shape == (4, 128)


def test_shorter_sequence(encoder):
    x = torch.randn(2, 200, 120)
    out = encoder(x)
    assert out.shape == (2, 128)


def test_eval_deterministic(encoder):
    encoder.eval()
    x = torch.randn(2, 400, 120)
    with torch.no_grad():
        out1 = encoder(x)
        out2 = encoder(x)
    assert torch.allclose(out1, out2)
