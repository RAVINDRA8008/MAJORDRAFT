"""Centralized path definitions for Colab + Google Drive.

All project paths are resolved relative to the Google Drive mount point.
Supports an optional local cache on the Colab VM for faster data I/O.
"""

import os


# ---------------------------------------------------------------------------
# Default base paths
# ---------------------------------------------------------------------------
_DRIVE_BASE = "/content/drive/MyDrive/AMERS"
_LOCAL_CACHE = "/content/local_data"


def get_paths(use_local_cache: bool = True) -> dict[str, str]:
    """Return all project paths, resolved for Colab + Google Drive.

    Args:
        use_local_cache: If True *and* the local cache directory exists,
            data paths point to ``/content/local_data/`` for faster training
            I/O.  Otherwise data paths point directly to Drive.

    Returns:
        Dictionary with the following string keys:
        ``drive_base``, ``code_dir``, ``config_dir``,
        ``deap_raw``, ``deap_processed``,
        ``iemocap_raw``, ``iemocap_processed``,
        ``checkpoints``, ``tensorboard``, ``evaluation``, ``export``.
    """
    if use_local_cache and os.path.exists(_LOCAL_CACHE):
        data_base = _LOCAL_CACHE
    else:
        data_base = f"{_DRIVE_BASE}/data"

    return {
        # Drive paths (always persistent)
        "drive_base": _DRIVE_BASE,
        "code_dir": f"{_DRIVE_BASE}/code",
        "config_dir": f"{_DRIVE_BASE}/config",

        # Data paths (local cache or Drive)
        "deap_raw": f"{_DRIVE_BASE}/data/deap/raw",
        "deap_processed": f"{data_base}/deap/processed",
        "iemocap_raw": f"{_DRIVE_BASE}/data/iemocap/raw",
        "iemocap_processed": f"{data_base}/iemocap/processed",

        # Output paths (always on Drive for persistence)
        "checkpoints": f"{_DRIVE_BASE}/outputs/checkpoints",
        "tensorboard": f"{_DRIVE_BASE}/outputs/tensorboard",
        "evaluation": f"{_DRIVE_BASE}/outputs/evaluation",
        "export": f"{_DRIVE_BASE}/outputs/export",
    }


def ensure_dirs(paths: dict[str, str] | None = None) -> None:
    """Create every directory listed in *paths* if it doesn't already exist."""
    if paths is None:
        paths = get_paths(use_local_cache=False)
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
