"""EEG preprocessing pipeline for the DEAP dataset.

Pipeline steps (per subject):
    1. Load ``.dat`` pickle → extract 32 EEG channels
    2. Remove 3-second pre-trial baseline
    3. Band-pass filter (0.5–50 Hz)
    4. Segment into 1-second non-overlapping epochs
    5. Compute Differential Entropy (DE) in 5 frequency bands
    6. Z-score normalisation
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Preprocess raw DEAP EEG data into differential-entropy features."""

    # Default frequency bands (Hz)
    DEFAULT_BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 14),
        "beta": (14, 31),
        "gamma": (31, 50),
    }

    def __init__(self, config: dict) -> None:
        self.sampling_rate: int = config.get("sampling_rate", 128)
        self.epoch_length_sec: float = config.get("epoch_length_sec", 1.0)
        self.filter_order: int = config.get("filter_order", 5)
        self.n_eeg_channels: int = 32  # DEAP has 32 EEG + 8 peripheral
        self.baseline_samples: int = int(3 * self.sampling_rate)  # 3 s baseline

        # Frequency bands
        bands_cfg = config.get("frequency_bands", {})
        self.frequency_bands: dict[str, tuple[float, float]] = {}
        for name, default in self.DEFAULT_BANDS.items():
            val = bands_cfg.get(name, list(default))
            self.frequency_bands[name] = (float(val[0]), float(val[1]))

        self.samples_per_epoch = int(self.epoch_length_sec * self.sampling_rate)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def bandpass_filter(
        self, signal: np.ndarray, low: float, high: float
    ) -> np.ndarray:
        """Zero-phase Butterworth band-pass filter (single channel).

        Args:
            signal: 1-D array of samples.
            low: Lower cut-off frequency (Hz).
            high: Upper cut-off frequency (Hz).

        Returns:
            Filtered signal (same length).
        """
        nyq = 0.5 * self.sampling_rate
        b, a = butter(self.filter_order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, signal).astype(np.float32)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def segment_epochs(
        self, data: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment continuous trial data into fixed-length epochs.

        Args:
            data: ``(n_trials, n_channels, n_samples)`` — raw trial data
                  **after** baseline removal.
            labels: ``(n_trials,)`` — integer class labels per trial.

        Returns:
            ``(epochs, epoch_labels)`` where
            *epochs* has shape ``(n_epochs, n_channels, samples_per_epoch)``
            and *epoch_labels* has shape ``(n_epochs,)``.
        """
        n_trials, n_channels, n_samples = data.shape
        n_epochs_per_trial = n_samples // self.samples_per_epoch

        all_epochs = []
        all_labels = []
        for t in range(n_trials):
            for e in range(n_epochs_per_trial):
                start = e * self.samples_per_epoch
                end = start + self.samples_per_epoch
                all_epochs.append(data[t, :, start:end])
                all_labels.append(labels[t])

        return np.array(all_epochs, dtype=np.float32), np.array(all_labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # Differential Entropy
    # ------------------------------------------------------------------
    def compute_differential_entropy(self, epoch: np.ndarray) -> np.ndarray:
        """Compute DE features for one epoch.

        Args:
            epoch: ``(n_channels, samples_per_epoch)``

        Returns:
            ``(n_channels, n_bands)`` — e.g. ``(32, 5)``
        """
        n_channels = epoch.shape[0]
        n_bands = len(self.frequency_bands)
        de_features = np.zeros((n_channels, n_bands), dtype=np.float32)

        for ch in range(n_channels):
            for b_idx, (_, (low, high)) in enumerate(self.frequency_bands.items()):
                filtered = self.bandpass_filter(epoch[ch], low, high)
                variance = np.var(filtered) + 1e-10  # avoid log(0)
                de_features[ch, b_idx] = 0.5 * np.log(2 * np.pi * np.e * variance)

        return de_features

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(features: np.ndarray) -> np.ndarray:
        """Z-score normalisation across the feature array.

        Args:
            features: ``(n_samples, n_channels, n_bands)`` or ``(n_samples, feature_dim)``

        Returns:
            Normalised array (same shape).
        """
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return ((features - mean) / std).astype(np.float32)

    # ------------------------------------------------------------------
    # End-to-end per-subject processing
    # ------------------------------------------------------------------
    def process_subject(
        self, filepath: str, label_mapper=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full preprocessing pipeline for one DEAP subject file.

        Args:
            filepath: Path to a DEAP ``.dat`` file (pickle).
            label_mapper: Optional :class:`LabelMapper` instance.  If *None*,
                a default 4-class quadrant mapping is used internally.

        Returns:
            ``(features, labels)`` — features ``(n_epochs, 32, 5)``,
            labels ``(n_epochs,)``.
        """
        logger.info("Loading %s ...", filepath)

        with open(filepath, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        data = raw["data"]  # (40, 40, 8064)
        raw_labels = raw["labels"]  # (40, 4)

        # Keep only 32 EEG channels (drop 8 peripheral)
        eeg_data = data[:, : self.n_eeg_channels, :]  # (40, 32, 8064)

        # Remove 3-s baseline at the start of each trial
        eeg_data = eeg_data[:, :, self.baseline_samples :]  # (40, 32, 7680)
        logger.info("Baseline removed. Signal length: %d samples", eeg_data.shape[2])

        # Map labels
        if label_mapper is not None:
            int_labels = np.array(
                [
                    label_mapper.map_deap_labels(
                        raw_labels[t, 0], raw_labels[t, 1]
                    )
                    for t in range(raw_labels.shape[0])
                ],
                dtype=np.int64,
            )
        else:
            # Default 4-class quadrant mapping
            int_labels = self._default_label_map(raw_labels)

        # Segment into 1-second epochs
        epochs, epoch_labels = self.segment_epochs(eeg_data, int_labels)
        logger.info(
            "Segmented: %d epochs of %d samples each", len(epochs), self.samples_per_epoch
        )

        # Compute DE for every epoch
        de_features = np.array(
            [self.compute_differential_entropy(ep) for ep in epochs],
            dtype=np.float32,
        )
        logger.info("DE features shape: %s", de_features.shape)

        # Normalise
        de_features = self.normalize(de_features)

        return de_features, epoch_labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_label_map(
        raw_labels: np.ndarray,
        v_thresh: float = 5.0,
        a_thresh: float = 5.0,
    ) -> np.ndarray:
        """Quadrant mapping: (valence, arousal) → 4-class label.

        0=happy (HV+HA), 1=sad (LV+LA), 2=angry (LV+HA), 3=neutral (HV+LA)
        """
        labels = []
        for t in range(raw_labels.shape[0]):
            v, a = raw_labels[t, 0], raw_labels[t, 1]
            if v >= v_thresh and a >= a_thresh:
                labels.append(0)  # happy
            elif v < v_thresh and a < a_thresh:
                labels.append(1)  # sad
            elif v < v_thresh and a >= a_thresh:
                labels.append(2)  # angry
            else:
                labels.append(3)  # neutral
        return np.array(labels, dtype=np.int64)
