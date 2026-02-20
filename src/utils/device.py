"""GPU / CPU device management helpers."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "cuda") -> torch.device:
    """Return a :class:`torch.device`, falling back to CPU if CUDA unavailable.

    Args:
        preferred: ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved torch device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info("Using GPU: %s (%.1f GB)", name, mem)
    else:
        dev = torch.device("cpu")
        if preferred == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU")
        else:
            logger.info("Using CPU")
    return dev


def log_gpu_memory() -> None:
    """Log current CUDA memory allocation (no-op on CPU)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logger.info("GPU memory — allocated: %.2f GB, cached: %.2f GB", alloc, cached)


def free_gpu_memory() -> None:
    """Empty the CUDA cache and run Python garbage collection."""
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
