"""IEMOCAP dataset loader — read preprocessed .npy feature files from Drive."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class IEMOCAPLoader:
    """Load preprocessed IEMOCAP features from ``.npy`` files."""

    def __init__(
        self,
        processed_dir: str,
        sessions: list[int] | None = None,
    ) -> None:
        """
        Args:
            processed_dir: Path to ``iemocap/processed/features/``.
            sessions: Session IDs (1–5) to load.  ``None`` loads all.
        """
        self.processed_dir = Path(processed_dir) / "features"
        self.sessions = sessions or list(range(1, 6))

    def load_session(self, session_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load features and labels for one session.

        Returns:
            ``(features, labels)`` — features ``(n_utterances, max_frames, 120)``,
            labels ``(n_utterances,)``.
        """
        feat_path = self.processed_dir / f"session{session_id}_features.npy"
        label_path = self.processed_dir / f"session{session_id}_labels.npy"

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        features = np.load(str(feat_path))
        labels = np.load(str(label_path))
        return features, labels

    def load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load all requested sessions and concatenate.

        Returns:
            ``(features, labels, session_ids)``
        """
        all_features, all_labels, all_sids = [], [], []
        for sid in self.sessions:
            try:
                feat, lab = self.load_session(sid)
                all_features.append(feat)
                all_labels.append(lab)
                all_sids.append(np.full(len(lab), sid, dtype=np.int64))
                logger.info("Loaded session %d: %d utterances", sid, len(lab))
            except FileNotFoundError:
                logger.warning("Session %d not found — skipping", sid)

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        session_ids = np.concatenate(all_sids, axis=0)

        logger.info(
            "IEMOCAP loaded: %d total utterances, features shape %s",
            len(labels),
            features.shape,
        )
        return features, labels, session_ids
