"""EEG Encoder with multi-scale attention (v4).

Maps flattened differential-entropy features ``(batch, 160)``
to a fixed-dimensional emotion embedding ``(batch, 128)``.

v4 architecture upgrades:
- Reshape 160 → (32, 5): 32 channels × 5 frequency bands
- Deeper channel projection with residual connections
- Multi-head self-attention across channels (configurable layers)
- Learnable CLS token for aggregation (replaces mean pooling)
- Positional encoding for channel ordering
- FC projection to embedding with skip connection
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Multi-head self-attention across EEG channels with pre-norm."""

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
        # Pre-norm self-attention with residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        # Pre-norm feed-forward with residual
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class EEGEncoder(nn.Module):
    """Attention-augmented encoder for DEAP EEG features (v4).

    Architecture:
    1. Reshape (B, 160) → (B, 32, 5)
    2. Linear projection: 5 → hidden_dim per channel
    3. Add learnable positional encoding
    4. Prepend CLS token
    5. N self-attention layers across 33 tokens (CLS + 32 channels)
    6. Extract CLS token → (B, hidden_dim)
    7. FC → embedding with skip connection
    """

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        n_channels: int = 32,
        n_bands: int = 5,
        n_attn_layers: int = 3,
        n_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.n_channels = n_channels
        self.n_bands = n_bands
        attn_dim = hidden_dims[0] if hidden_dims else 128

        # Channel feature projection: 5 bands → attn_dim (deeper)
        self.channel_proj = nn.Sequential(
            nn.Linear(n_bands, attn_dim // 2),
            nn.LayerNorm(attn_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(attn_dim // 2, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Learnable CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, attn_dim) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_channels + 1, attn_dim) * 0.02)

        # Self-attention layers (deeper)
        self.attn_layers = nn.ModuleList([
            ChannelAttention(attn_dim, num_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_attn_layers)
        ])
        self.final_norm = nn.LayerNorm(attn_dim)

        # FC projection head with skip
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

        # Skip connection from attn_dim to embedding_dim
        self.skip = nn.Linear(attn_dim, embedding_dim) if attn_dim != embedding_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, 160)`` or ``(batch, 32, 5)``

        Returns:
            Embedding ``(batch, embedding_dim)``
        """
        if x.ndim == 2:
            x = x.view(x.size(0), self.n_channels, self.n_bands)  # (B, 32, 5)

        B = x.size(0)

        # Project each channel's band-features
        x = self.channel_proj(x)  # (B, 32, attn_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, attn_dim)
        x = torch.cat([cls, x], dim=1)  # (B, 33, attn_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, : x.size(1), :]

        # Self-attention across channels
        for layer in self.attn_layers:
            x = layer(x)

        x = self.final_norm(x)

        # Extract CLS token
        cls_out = x[:, 0, :]  # (B, attn_dim)

        # FC head with skip connection
        return self.fc(cls_out) + self.skip(cls_out)
