"""Gated fusion classifier with embedding normalization.

Architecture:
1. Normalize each modality's embeddings (LayerNorm)
2. Gating mechanism: gate = σ(W[eeg || speech])
3. Fused = gate * eeg_proj + (1-gate) * speech_proj
4. MLP classifier

Replaces the previous cross-modal attention approach, which was
unstable on single-vector (B, D) embeddings.  The gating mechanism
is simpler, more robust, and naturally handles feature-scale differences
via LayerNorm + learned gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Learnable gating mechanism for multimodal fusion.

    gate = σ(MLP([eeg || speech]))
    fused = gate * eeg_proj + (1 - gate) * speech_proj
    """

    def __init__(
        self,
        eeg_dim: int,
        speech_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.eeg_norm = nn.LayerNorm(eeg_dim)
        self.speech_norm = nn.LayerNorm(speech_dim)

        # Project both modalities to the same dimension
        self.fuse_dim = max(eeg_dim, speech_dim)
        self.eeg_proj = (
            nn.Linear(eeg_dim, self.fuse_dim)
            if eeg_dim != self.fuse_dim
            else nn.Identity()
        )
        self.speech_proj = (
            nn.Linear(speech_dim, self.fuse_dim)
            if speech_dim != self.fuse_dim
            else nn.Identity()
        )

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(self.fuse_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.fuse_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, eeg_emb: torch.Tensor, speech_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            eeg_emb: ``(B, eeg_dim)``
            speech_emb: ``(B, speech_dim)``

        Returns:
            Fused representation ``(B, fuse_dim)``
        """
        eeg = self.eeg_norm(eeg_emb)
        speech = self.speech_norm(speech_emb)

        eeg_p = self.eeg_proj(eeg)
        speech_p = self.speech_proj(speech)

        combined = torch.cat([eeg_p, speech_p], dim=1)
        gate = self.gate_net(combined)

        fused = gate * eeg_p + (1 - gate) * speech_p
        return fused


class FusionClassifier(nn.Module):
    """Gated fusion classifier: EEG + speech → class logits.

    Architecture:
    1. Normalize + project each modality embedding
    2. Gated fusion (learnable gate)
    3. MLP classifier with BatchNorm + Dropout
    """

    def __init__(
        self,
        eeg_embed_dim: int = 128,
        speech_embed_dim: int = 128,
        hidden_dims: list[int] | None = None,
        num_classes: int = 4,
        dropout: list[float] | float | None = None,
        modality_dropout_prob: float = 0.1,
        n_attn_heads: int = 4,  # kept for config backward-compat, unused
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if dropout is None:
            dropout = [0.4, 0.3]
        if isinstance(dropout, (int, float)):
            dropout = [float(dropout)] * len(hidden_dims)

        self.modality_dropout_prob = modality_dropout_prob
        self.eeg_embed_dim = eeg_embed_dim
        self.speech_embed_dim = speech_embed_dim

        # Gated fusion
        self.gated_fusion = GatedFusion(
            eeg_embed_dim,
            speech_embed_dim,
            hidden_dim=hidden_dims[0] if hidden_dims else 128,
            dropout=dropout[0] if dropout else 0.3,
        )

        # MLP classifier
        fuse_dim = self.gated_fusion.fuse_dim
        layers: list[nn.Module] = []
        in_dim = fuse_dim
        for h_dim, drop in zip(hidden_dims, dropout):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(drop),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        eeg_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            eeg_embedding: ``(batch, eeg_embed_dim)``
            speech_embedding: ``(batch, speech_embed_dim)``

        Returns:
            Raw logits ``(batch, num_classes)``
        """
        # Modality dropout during training
        if self.training and self.modality_dropout_prob > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout_prob / 2:
                eeg_embedding = torch.zeros_like(eeg_embedding)
            elif r < self.modality_dropout_prob:
                speech_embedding = torch.zeros_like(speech_embedding)

        fused = self.gated_fusion(eeg_embedding, speech_embedding)
        return self.classifier(fused)
