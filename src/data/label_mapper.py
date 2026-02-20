"""Unified emotion label mapping for DEAP and IEMOCAP.

4-class unified set:
    0 — Happy
    1 — Sad
    2 — Angry
    3 — Neutral
"""

from __future__ import annotations


class LabelMapper:
    """Map dataset-specific labels to a unified emotion label set."""

    EMOTIONS_4CLASS: dict[int, str] = {
        0: "happy",
        1: "sad",
        2: "angry",
        3: "neutral",
    }
    NUM_CLASSES = 4

    # IEMOCAP label string → unified int
    _IEMOCAP_MAP: dict[str, int] = {
        "hap": 0,
        "happy": 0,
        "exc": 0,
        "excited": 0,
        "sad": 1,
        "ang": 2,
        "angry": 2,
        "neu": 3,
        "neutral": 3,
    }

    # ------------------------------------------------------------------
    # DEAP
    # ------------------------------------------------------------------
    @staticmethod
    def map_deap_labels(
        valence: float,
        arousal: float,
        label_type: str = "4class",
        valence_threshold: float = 5.0,
        arousal_threshold: float = 5.0,
    ) -> int:
        """Map DEAP valence/arousal ratings to the 4-class quadrant model.

        Returns:
            0 (happy)   — High V, High A
            1 (sad)     — Low V, Low A
            2 (angry)   — Low V, High A
            3 (neutral) — High V, Low A
        """
        hv = valence >= valence_threshold
        ha = arousal >= arousal_threshold

        if hv and ha:
            return 0  # happy
        if not hv and not ha:
            return 1  # sad
        if not hv and ha:
            return 2  # angry
        return 3  # neutral

    # ------------------------------------------------------------------
    # IEMOCAP
    # ------------------------------------------------------------------
    @classmethod
    def map_iemocap_labels(cls, label_str: str) -> int:
        """Map an IEMOCAP categorical label string to the unified integer.

        Returns:
            Unified class id (0–3) or **-1** if the label is not in the
            target emotion set (e.g. ``"fru"``, ``"oth"``).
        """
        return cls._IEMOCAP_MAP.get(label_str.strip().lower(), -1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def label_name(cls, label_id: int) -> str:
        """Return the human-readable name for a class integer."""
        return cls.EMOTIONS_4CLASS.get(label_id, "unknown")
