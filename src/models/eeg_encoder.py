"""EEG Fully-Connected Encoder.

Maps flattened differential-entropy features ``(batch, 160)``
to a fixed-dimensional emotion embedding ``(batch, 128)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """FC encoder for DEAP EEG differential-entropy features."""

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final projection to embedding space (no activation)
        layers.append(nn.Linear(in_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, 32, 5)`` or ``(batch, 160)``

        Returns:
            Embedding ``(batch, embedding_dim)``
        """
        if x.ndim == 3:
            x = x.view(x.size(0), -1)  # flatten (B, 32, 5) → (B, 160)
        return self.encoder(x)
