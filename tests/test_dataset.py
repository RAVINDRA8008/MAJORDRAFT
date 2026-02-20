"""Tests for dataset classes."""

import torch
import pytest

from src.data.dataset import EEGDataset, SpeechDataset, FusionDataset


def test_eeg_dataset():
    feats = torch.randn(100, 160)
    labels = torch.randint(0, 4, (100,))
    ds = EEGDataset(feats, labels)
    assert len(ds) == 100
    x, y = ds[0]
    assert x.shape == (160,)


def test_speech_dataset():
    feats = torch.randn(50, 800, 120)
    labels = torch.randint(0, 4, (50,))
    ds = SpeechDataset(feats, labels)
    assert len(ds) == 50
    x, y = ds[0]
    assert x.shape == (800, 120)


def test_fusion_dataset():
    eeg = torch.randn(30, 128)
    sp = torch.randn(30, 128)
    labels = torch.randint(0, 4, (30,))
    ds = FusionDataset(eeg, sp, labels)
    assert len(ds) == 30
    e, s, y = ds[0]
    assert e.shape == (128,)
    assert s.shape == (128,)
