"""EEG Encoder with self-attention.

Maps flattened differential-entropy features ``(batch, 160)``
to a fixed-dimensional emotion embedding ``(batch, 128)``.

Upgraded architecture:
- Reshape 160 → (32, 5): 32 channels × 5 frequency bands
- Multi-head self-attention across channels
- FC projection to embedding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Multi-head self-attention across EEG channels."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        return x


class EEGEncoder(nn.Module):
    """Attention-augmented FC encoder for DEAP EEG features.

    Architecture:
    1. Reshape (B, 160) → (B, 32, 5)
    2. Linear projection: 5 → hidden_dim per channel
    3. N self-attention layers across 32 channels
    4. Mean-pool across channels → (B, hidden_dim)
    5. FC → embedding
    """

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        n_channels: int = 32,
        n_bands: int = 5,
        n_attn_layers: int = 2,
        n_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.n_channels = n_channels
        self.n_bands = n_bands
        attn_dim = hidden_dims[0] if hidden_dims else 128

        # Channel feature projection: 5 bands → attn_dim
        self.channel_proj = nn.Sequential(
            nn.Linear(n_bands, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Self-attention layers
        self.attn_layers = nn.ModuleList([
            ChannelAttention(attn_dim, num_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_attn_layers)
        ])

        # FC projection head
        fc_layers: list[nn.Module] = []
        in_dim = attn_dim
        for h_dim in hidden_dims[1:]:
            fc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        fc_layers.append(nn.Linear(in_dim, embedding_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, 160)`` or ``(batch, 32, 5)``

        Returns:
            Embedding ``(batch, embedding_dim)``
        """
        if x.ndim == 2:
            x = x.view(x.size(0), self.n_channels, self.n_bands)  # (B, 32, 5)

        # Project each channel's band-features
        x = self.channel_proj(x)  # (B, 32, attn_dim)

        # Self-attention across channels
        for layer in self.attn_layers:
            x = layer(x)

        # Mean-pool over channels
        x = x.mean(dim=1)  # (B, attn_dim)

        # FC head
        return self.fc(x)
