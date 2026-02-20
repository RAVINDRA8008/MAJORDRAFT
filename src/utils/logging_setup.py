"""Logging configuration for the AMERS project."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Logging level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers[0] = handler
