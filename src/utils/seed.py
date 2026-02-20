"""Global random-seed setting for full reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Also enables deterministic CuDNN behaviour.  Call this **once** at the
    very beginning of every script / notebook session.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic ops (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
