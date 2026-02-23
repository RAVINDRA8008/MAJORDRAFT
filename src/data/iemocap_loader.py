"""IEMOCAP dataset loader — read preprocessed .npy feature files from Drive.

Robust loader with automatic path discovery, integrity validation, and
label-distribution reporting.  If the processed directory does not contain
``session*_features.npy`` directly it will also search a ``features/``
sub-folder for backward compatibility.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Human-readable label map (matches preprocess_iemocap.py)
_LABEL_NAMES = {0: "angry", 1: "happy", 2: "sad", 3: "neutral"}


class IEMOCAPLoader:
    """Load preprocessed IEMOCAP features from ``.npy`` files.

    The loader performs the following steps on construction:

    1. **Path discovery** — accepts ``iemocap/processed`` (or ``.../features``)
       and resolves whichever directory actually contains the ``.npy`` files.
    2. **File-system scan** — discovers all ``session*_features.npy`` files.
    3. **Integrity validation** — checks for NaN / Inf and matching
       feature/label counts.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        processed_dir: str,
        sessions: list[int] | None = None,
    ) -> None:
        self.processed_dir = self._resolve_dir(Path(processed_dir))
        self.sessions = sessions or list(range(1, 6))
        logger.info("IEMOCAPLoader — resolved dir: %s", self.processed_dir)

    @staticmethod
    def _resolve_dir(base: Path) -> Path:
        """Find the directory that actually has the .npy files."""
        candidates = [
            base,
            base / "features",
        ]
        for c in candidates:
            if c.is_dir() and any(c.glob("session*_features.npy")):
                return c
        # Fall back to base even if empty — later calls will give clear errors.
        logger.warning(
            "No session*_features.npy found in %s or %s/features — "
            "load_all() will fail.",
            base,
            base,
        )
        return base

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_sessions(self) -> list[int]:
        """Return sorted list of session IDs found on disk."""
        found: list[int] = []
        for p in sorted(self.processed_dir.glob("session*_features.npy")):
            try:
                sid = int(p.stem.replace("session", "").replace("_features", ""))
                found.append(sid)
            except ValueError:
                continue
        return found

    # ------------------------------------------------------------------
    # Single-session load with validation
    # ------------------------------------------------------------------
    def load_session(self, session_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load and validate features + labels for one session."""
        feat_path = self.processed_dir / f"session{session_id}_features.npy"
        label_path = self.processed_dir / f"session{session_id}_labels.npy"

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        features = np.load(str(feat_path))
        labels = np.load(str(label_path))

        # --- integrity checks ---
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Session {session_id}: feature rows ({features.shape[0]}) "
                f"!= label rows ({labels.shape[0]})"
            )
        if np.isnan(features).any() or np.isinf(features).any():
            logger.warning(
                "Session %d features contain NaN/Inf — replacing with 0.",
                session_id,
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, labels

    # ------------------------------------------------------------------
    # Load all sessions
    # ------------------------------------------------------------------
    def load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load all requested sessions, concatenate, and report stats."""
        available = set(self.discover_sessions())
        if not available:
            raise FileNotFoundError(
                f"No session .npy files found in {self.processed_dir}. "
                "Did preprocessing run successfully?"
            )

        all_features, all_labels, all_sids = [], [], []
        loaded, skipped = 0, 0

        for sid in self.sessions:
            if sid not in available:
                logger.warning("Session %d not on disk — skipping.", sid)
                skipped += 1
                continue
            try:
                feat, lab = self.load_session(sid)
                all_features.append(feat)
                all_labels.append(lab)
                all_sids.append(np.full(len(lab), sid, dtype=np.int64))
                logger.info(
                    "Session %d: %d utterances, shape %s",
                    sid,
                    len(lab),
                    feat.shape,
                )
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Session %d failed: %s — skipping.", sid, exc)
                skipped += 1

        if not all_features:
            raise RuntimeError(
                f"No sessions could be loaded (requested {self.sessions}, "
                f"available on disk {sorted(available)}). "
                "Check preprocessing output."
            )

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        session_ids = np.concatenate(all_sids, axis=0)

        # --- label distribution ---
        dist = Counter(int(l) for l in labels)
        dist_str = ", ".join(
            f"{_LABEL_NAMES.get(k, k)}: {v}" for k, v in sorted(dist.items())
        )
        logger.info(
            "IEMOCAP — %d sessions loaded, %d skipped | "
            "%d total utterances | features %s | labels {%s}",
            loaded,
            skipped,
            len(labels),
            features.shape,
            dist_str,
        )

        return features, labels, session_ids
