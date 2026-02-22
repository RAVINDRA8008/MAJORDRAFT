"""DEAP dataset loader — read preprocessed .npy feature files from Drive.

Includes data validation, integrity checking, and label-distribution
reporting so that downstream training scripts get clean data.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_FEATURE_DIM = 160  # 32 channels × 5 frequency bands


class DEAPLoader:
    """Load preprocessed DEAP features from ``.npy`` files.

    Features:
      * Automatic discovery of available subjects
      * Corruption detection (NaN / Inf / wrong shape)
      * Label-distribution summary
      * Never crashes if some subjects are missing
    """

    def __init__(self, processed_dir: str, subjects: list[int] | None = None) -> None:
        """
        Args:
            processed_dir: Path to the ``deap/processed/`` folder.
            subjects: List of subject IDs (1–32) to load.
                      ``None`` loads all 32.
        """
        self.processed_dir = Path(processed_dir)
        self.subjects = subjects or list(range(1, 33))

    # ------------------------------------------------------------------
    # Discovery & validation
    # ------------------------------------------------------------------
    def discover_subjects(self) -> tuple[list[int], list[int]]:
        """Scan the directory and classify subjects as found or missing.

        Returns:
            ``(found_ids, missing_ids)``
        """
        found, missing = [], []
        for sid in self.subjects:
            feat_path = self.processed_dir / f"s{sid:02d}_features.npy"
            if feat_path.exists():
                found.append(sid)
            else:
                missing.append(sid)
        return found, missing

    @staticmethod
    def _validate_array(arr: np.ndarray, name: str) -> list[str]:
        """Return a list of issues found in *arr* (empty = OK)."""
        issues: list[str] = []
        if arr.size == 0:
            issues.append(f"{name}: empty array")
        if np.isnan(arr).any():
            issues.append(f"{name}: contains NaN ({np.isnan(arr).sum()} values)")
        if np.isinf(arr).any():
            issues.append(f"{name}: contains Inf ({np.isinf(arr).sum()} values)")
        return issues

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_subject(self, subject_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load features and labels for one subject.

        Returns:
            ``(features, labels)`` — features ``(n_epochs, 32, 5)`` or
            ``(n_epochs, 160)``, labels ``(n_epochs,)``.
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
        """Load all requested subjects, validate, and concatenate.

        Args:
            flatten: If *True*, reshape features to ``(N, 160)`` for FC input.

        Returns:
            ``(features, labels, subject_ids)``

        Raises:
            RuntimeError: If **no** subject files could be loaded.
        """
        found, missing = self.discover_subjects()

        if missing:
            logger.warning(
                "Missing %d/%d subjects: %s",
                len(missing), len(self.subjects),
                ", ".join(f"s{s:02d}" for s in missing),
            )
        if not found:
            raise RuntimeError(
                f"No DEAP subject files found in {self.processed_dir}. "
                "Run preprocess_deap.py first."
            )

        all_features, all_labels, all_sids = [], [], []
        corrupt_subjects: list[int] = []

        for sid in found:
            feat, lab = self.load_subject(sid)

            # ── Integrity checks ──
            issues = self._validate_array(feat, f"s{sid:02d}_features")
            issues += self._validate_array(lab, f"s{sid:02d}_labels")

            flat_dim = feat.reshape(feat.shape[0], -1).shape[1]
            if flat_dim != EXPECTED_FEATURE_DIM:
                issues.append(
                    f"s{sid:02d}: expected flat dim {EXPECTED_FEATURE_DIM}, "
                    f"got {flat_dim}"
                )

            if issues:
                for issue in issues:
                    logger.warning("DATA ISSUE — %s", issue)
                corrupt_subjects.append(sid)
                continue

            all_features.append(feat)
            all_labels.append(lab)
            all_sids.append(np.full(len(lab), sid, dtype=np.int64))
            logger.info("Loaded subject %02d: %d samples", sid, len(lab))

        if not all_features:
            raise RuntimeError(
                "All loaded subject files were corrupt. "
                "Re-run preprocess_deap.py to regenerate."
            )

        if corrupt_subjects:
            logger.warning(
                "Skipped %d corrupt subjects: %s",
                len(corrupt_subjects),
                ", ".join(f"s{s:02d}" for s in corrupt_subjects),
            )

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        subject_ids = np.concatenate(all_sids, axis=0)

        if flatten:
            features = features.reshape(features.shape[0], -1)  # (N, 160)

        # ── Summary ──
        label_dist = Counter(labels.tolist())
        dist_str = "  ".join(
            f"class {k}: {v}" for k, v in sorted(label_dist.items())
        )
        logger.info(
            "DEAP loaded: %d samples from %d/%d subjects, shape %s",
            len(labels), len(found) - len(corrupt_subjects),
            len(self.subjects), features.shape,
        )
        logger.info("Label distribution — %s", dist_str)

        return features, labels, subject_ids
