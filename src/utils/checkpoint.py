"""Model checkpoint save / load utilities.

Checkpoints are always saved to **Google Drive** so they persist across
Colab session resets.  Loading supports ``map_location`` so checkpoints
saved on GPU can be loaded on CPU and vice-versa.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: dict[str, Any],
    filepath: str | Path,
    *,
    tag: str = "",
) -> str:
    """Save a checkpoint dictionary to *filepath*.

    Args:
        state: Typically includes ``model_state_dict``, ``optimizer_state_dict``,
            ``epoch``, ``best_metric``, and any other metadata.
        filepath: Absolute path on Google Drive (e.g.
            ``/content/drive/MyDrive/AMERS/outputs/checkpoints/gan_epoch300.pt``).
        tag: Optional human-readable label for the log message.

    Returns:
        The actual path the file was written to (as a string).
    """
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    label = f" [{tag}]" if tag else ""
    logger.info("Checkpoint saved%s: %s", label, filepath)
    return filepath


def load_checkpoint(
    filepath: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint dictionary from *filepath*.

    Args:
        filepath: Path to the ``.pt`` file.
        device: Target device (handles GPU → CPU mapping transparently).

    Returns:
        The loaded state dict.
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    state = torch.load(filepath, map_location=device, weights_only=False)
    logger.info("Checkpoint loaded: %s", filepath)
    return state


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metric: float,
    filepath: str | Path,
    *,
    tag: str = "",
    extra: dict[str, Any] | None = None,
) -> str:
    """Convenience wrapper: build a state dict and call :func:`save_checkpoint`.

    Args:
        model: The PyTorch module.
        optimizer: Optimizer (can be *None* for inference-only checkpoints).
        epoch: Current epoch number.
        metric: The metric value that triggered saving (e.g., best val acc).
        filepath: Destination on Drive.
        tag: Human-readable label.
        extra: Any additional metadata to include in the checkpoint.

    Returns:
        Path the checkpoint was written to.
    """
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metric": metric,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    return save_checkpoint(state, filepath, tag=tag)


def load_model(
    model: torch.nn.Module,
    filepath: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights from a checkpoint file.

    Args:
        model: The model instance to load weights into.
        filepath: Path to the ``.pt`` checkpoint.
        device: Target device.
        strict: Whether to enforce an exact match of parameter keys.

    Returns:
        The full checkpoint dict (so callers can access ``epoch``, etc.).
    """
    ckpt = load_checkpoint(filepath, device=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return ckpt
