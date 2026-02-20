"""DEAP dataset loader — read preprocessed .npy feature files from Drive."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DEAPLoader:
    """Load preprocessed DEAP features from ``.npy`` files."""

    def __init__(self, processed_dir: str, subjects: list[int] | None = None) -> None:
        """
        Args:
            processed_dir: Path to the ``deap/processed/features/`` folder.
            subjects: List of subject IDs (1–32) to load.
                      ``None`` loads all 32.
        """
        self.processed_dir = Path(processed_dir) / "features"
        self.subjects = subjects or list(range(1, 33))

    def load_subject(self, subject_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load features and labels for one subject.

        Returns:
            ``(features, labels)`` — features ``(n_epochs, 32, 5)``,
            labels ``(n_epochs,)``.
        """
        feat_path = self.processed_dir / f"s{subject_id:02d}_features.npy"
        label_path = self.processed_dir / f"s{subject_id:02d}_labels.npy"

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        features = np.load(str(feat_path))
        labels = np.load(str(label_path))
        return features, labels

    def load_all(
        self, flatten: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load all requested subjects and concatenate.

        Args:
            flatten: If *True*, reshape features to ``(N, 160)`` for FC input.

        Returns:
            ``(features, labels, subject_ids)``
        """
        all_features, all_labels, all_sids = [], [], []
        for sid in self.subjects:
            try:
                feat, lab = self.load_subject(sid)
                all_features.append(feat)
                all_labels.append(lab)
                all_sids.append(np.full(len(lab), sid, dtype=np.int64))
                logger.info("Loaded subject %02d: %d samples", sid, len(lab))
            except FileNotFoundError:
                logger.warning("Subject %02d not found — skipping", sid)

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        subject_ids = np.concatenate(all_sids, axis=0)

        if flatten:
            features = features.reshape(features.shape[0], -1)  # (N, 160)

        logger.info(
            "DEAP loaded: %d total samples, features shape %s",
            len(labels),
            features.shape,
        )
        return features, labels, subject_ids
