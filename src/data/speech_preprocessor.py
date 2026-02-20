"""Speech preprocessing pipeline for the IEMOCAP dataset.

Pipeline steps (per utterance):
    1. Load WAV → resample to 16 kHz
    2. Pre-emphasis filter
    3. Trim leading / trailing silence
    4. Extract 40-dim MFCCs + delta + delta-delta → 120 features
    5. Pad or truncate to a fixed number of frames (800)
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class SpeechPreprocessor:
    """Preprocess IEMOCAP audio into fixed-length MFCC feature sequences."""

    def __init__(self, config: dict) -> None:
        self.target_sr: int = config.get("target_sr", 16000)
        self.n_mfcc: int = config.get("n_mfcc", 40)
        self.frame_length_ms: int = config.get("frame_length_ms", 25)
        self.frame_shift_ms: int = config.get("frame_shift_ms", 10)
        self.n_mels: int = config.get("n_mels", 128)
        self.include_delta: bool = config.get("include_delta", True)
        self.include_delta_delta: bool = config.get("include_delta_delta", True)
        self.max_utterance_length_sec: float = config.get("max_utterance_length_sec", 8.0)
        self.pre_emphasis_coeff: float = config.get("pre_emphasis_coeff", 0.97)

        # Derived
        self.n_fft: int = int(self.frame_length_ms * self.target_sr / 1000)
        self.hop_length: int = int(self.frame_shift_ms * self.target_sr / 1000)
        self.max_frames: int = int(
            self.max_utterance_length_sec * 1000 / self.frame_shift_ms
        )
        self.n_features: int = self.n_mfcc * (
            1 + int(self.include_delta) + int(self.include_delta_delta)
        )

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------
    def load_audio(self, filepath: str) -> np.ndarray:
        """Load and resample audio to ``target_sr``.

        Returns:
            1-D float32 numpy array.
        """
        signal, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        return signal.astype(np.float32)

    def pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        """Apply first-order pre-emphasis: y[n] = x[n] - coeff * x[n-1]."""
        return np.append(signal[0], signal[1:] - self.pre_emphasis_coeff * signal[:-1]).astype(
            np.float32
        )

    @staticmethod
    def trim_silence(signal: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim leading and trailing silence from the waveform."""
        trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
        return trimmed

    def extract_mfcc(self, signal: np.ndarray) -> np.ndarray:
        """Extract MFCC features (+ optional deltas).

        Returns:
            ``(n_frames, n_features)`` — e.g. ``(T, 120)``
        """
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )  # (n_mfcc, T)

        features = [mfcc]
        if self.include_delta:
            features.append(librosa.feature.delta(mfcc))
        if self.include_delta_delta:
            features.append(librosa.feature.delta(mfcc, order=2))

        # Stack along feature axis → (n_features, T) → transpose → (T, n_features)
        return np.vstack(features).T.astype(np.float32)

    def pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """Pad (zero) or truncate to ``self.max_frames`` frames.

        Args:
            features: ``(T, n_features)``

        Returns:
            ``(max_frames, n_features)``
        """
        t, d = features.shape
        if t >= self.max_frames:
            return features[: self.max_frames]
        padding = np.zeros((self.max_frames - t, d), dtype=np.float32)
        return np.vstack([features, padding])

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------
    def process_utterance(self, filepath: str) -> np.ndarray:
        """Full pipeline for a single utterance WAV file.

        Returns:
            ``(max_frames, n_features)`` — e.g. ``(800, 120)``
        """
        signal = self.load_audio(filepath)
        signal = self.pre_emphasis(signal)
        signal = self.trim_silence(signal)
        mfcc_features = self.extract_mfcc(signal)
        return self.pad_or_truncate(mfcc_features)
