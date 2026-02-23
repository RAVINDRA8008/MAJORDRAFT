"""Cross-modal attention fusion classifier.

Replaces simple concatenation with a cross-attention mechanism:
- EEG attends to speech features
- Speech attends to EEG features
- Fused representations are combined and classified

Also supports modality dropout for robustness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention between two modality embeddings."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # EEG attends to speech
        self.eeg_to_speech = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Speech attends to EEG
        self.speech_to_eeg = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_eeg = nn.LayerNorm(embed_dim)
        self.norm_speech = nn.LayerNorm(embed_dim)

    def forward(
        self, eeg_emb: torch.Tensor, speech_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg_emb: ``(B, D)`` or ``(B, 1, D)``
            speech_emb: ``(B, D)`` or ``(B, 1, D)``

        Returns:
            Attended (eeg, speech) both ``(B, D)``
        """
        # Add sequence dim if needed
        if eeg_emb.ndim == 2:
            eeg_emb = eeg_emb.unsqueeze(1)
        if speech_emb.ndim == 2:
            speech_emb = speech_emb.unsqueeze(1)

        # Cross-attention
        eeg_attended, _ = self.eeg_to_speech(eeg_emb, speech_emb, speech_emb)
        speech_attended, _ = self.speech_to_eeg(speech_emb, eeg_emb, eeg_emb)

        # Residual + norm
        eeg_out = self.norm_eeg(eeg_emb + eeg_attended).squeeze(1)
        speech_out = self.norm_speech(speech_emb + speech_attended).squeeze(1)

        return eeg_out, speech_out


class FusionClassifier(nn.Module):
    """Cross-modal attention fusion: EEG + speech → class logits.

    Architecture:
    1. Project each modality to shared dim (if different)
    2. Cross-modal attention (EEG ↔ speech)
    3. Concatenate attended features
    4. MLP classifier
    """

    def __init__(
        self,
        eeg_embed_dim: int = 128,
        speech_embed_dim: int = 128,
        hidden_dims: list[int] | None = None,
        num_classes: int = 4,
        dropout: list[float] | None = None,
        modality_dropout_prob: float = 0.2,
        n_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if dropout is None:
            dropout = [0.5, 0.3]

        self.modality_dropout_prob = modality_dropout_prob
        self.eeg_embed_dim = eeg_embed_dim
        self.speech_embed_dim = speech_embed_dim

        # Project to common dim for cross-attention
        attn_dim = max(eeg_embed_dim, speech_embed_dim)
        self.eeg_proj = (
            nn.Linear(eeg_embed_dim, attn_dim) if eeg_embed_dim != attn_dim
            else nn.Identity()
        )
        self.speech_proj = (
            nn.Linear(speech_embed_dim, attn_dim) if speech_embed_dim != attn_dim
            else nn.Identity()
        )

        # Cross-modal attention
        self.cross_attn = CrossModalAttention(
            attn_dim, num_heads=n_attn_heads, dropout=dropout[0] if dropout else 0.3
        )

        # MLP classifier on concatenated attended features
        input_dim = attn_dim * 2
        layers: list[nn.Module] = []
        in_dim = input_dim
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

        # Project to shared dim
        eeg_proj = self.eeg_proj(eeg_embedding)
        speech_proj = self.speech_proj(speech_embedding)

        # Cross-modal attention
        eeg_attended, speech_attended = self.cross_attn(eeg_proj, speech_proj)

        # Fuse and classify
        fused = torch.cat([eeg_attended, speech_attended], dim=1)
        return self.classifier(fused)
