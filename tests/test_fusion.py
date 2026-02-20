"""Tests for fusion classifier."""

import torch
import pytest

from src.models.fusion import FusionClassifier


@pytest.fixture
def fusion():
    return FusionClassifier(
        eeg_dim=128, speech_dim=128,
        hidden_dims=[128, 64], num_classes=4,
        dropout=0.3, modality_dropout=0.2,
    )


def test_output_shape(fusion):
    eeg = torch.randn(8, 128)
    speech = torch.randn(8, 128)
    out = fusion(eeg, speech)
    assert out.shape == (8, 4)


def test_eval_no_modality_dropout(fusion):
    fusion.eval()
    eeg = torch.randn(4, 128)
    speech = torch.randn(4, 128)
    with torch.no_grad():
        out1 = fusion(eeg, speech)
        out2 = fusion(eeg, speech)
    assert torch.allclose(out1, out2)


def test_single_sample(fusion):
    eeg = torch.randn(1, 128)
    speech = torch.randn(1, 128)
    fusion.eval()
    out = fusion(eeg, speech)
    assert out.shape == (1, 4)
