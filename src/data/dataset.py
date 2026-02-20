"""PyTorch Dataset wrappers for DEAP (EEG) and IEMOCAP (speech) data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """PyTorch Dataset for DEAP EEG differential-entropy features.

    Each sample is a ``(features, label)`` tuple where
    *features* is a float32 tensor of shape ``(32, 5)`` or ``(160,)``
    depending on ``flatten``.
    """

    def __init__(
        self, features: np.ndarray, labels: np.ndarray, flatten: bool = True
    ) -> None:
        """
        Args:
            features: ``(N, 32, 5)`` DE feature array.
            labels: ``(N,)`` integer class labels.
            flatten: If *True*, each sample is ``(160,)`` instead of ``(32, 5)``.
        """
        if flatten and features.ndim == 3:
            features = features.reshape(features.shape[0], -1)
        self.features = torch.as_tensor(features).float()
        self.labels = torch.as_tensor(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SpeechDataset(Dataset):
    """PyTorch Dataset for IEMOCAP MFCC feature sequences.

    Each sample is ``(features, label)`` where *features* is
    ``(max_frames, n_features)`` — e.g. ``(800, 120)``.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            features: ``(N, max_frames, n_features)`` MFCC array.
            labels: ``(N,)`` integer class labels.
        """
        self.features = torch.as_tensor(features).float()
        self.labels = torch.as_tensor(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class FusionDataset(Dataset):
    """Combined EEG + Speech dataset for late-fusion training.

    Requires aligned samples (same index → same trial).
    """

    def __init__(
        self,
        eeg_features: np.ndarray,
        speech_features: np.ndarray,
        labels: np.ndarray,
        flatten_eeg: bool = True,
    ) -> None:
        if flatten_eeg and eeg_features.ndim == 3:
            eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
        self.eeg = torch.as_tensor(eeg_features).float()
        self.speech = torch.as_tensor(speech_features).float()
        self.labels = torch.as_tensor(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eeg[idx], self.speech[idx], self.labels[idx]
