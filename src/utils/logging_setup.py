"""Logging configuration for the AMERS project."""

from __future__ import annotations

import logging
import sys


def setup_logging(level_or_cfg="INFO") -> None:
    """Configure the root logger with a consistent format.

    Args:
        level_or_cfg: A logging-level string (``"INFO"``, etc.) **or** an
            OmegaConf / dict config object that has ``logging.level``.
    """
    # Accept full cfg objects — extract the level string
    if hasattr(level_or_cfg, "logging"):
        try:
            level = str(level_or_cfg.logging.level)
        except Exception:
            level = "INFO"
    elif isinstance(level_or_cfg, dict) and "logging" in level_or_cfg:
        level = str(level_or_cfg["logging"].get("level", "INFO"))
    else:
        level = str(level_or_cfg)

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
