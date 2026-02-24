"""Cross-Modal Transformer Fusion (v3).

Research rationale
──────────────────
GatedFusion (v2) uses a scalar gate to blend EEG and speech embeddings.
While stable, it cannot model fine-grained interactions between modalities.

The Cross-Modal Transformer (inspired by ViLBERT, Lu et al. 2019) uses
multi-head cross-attention to let each modality attend to the other,
learning richer inter-modal relationships.

Architecture
────────────
1. Modality-specific projection + learnable CLS tokens
2. Positional encoding (learned)
3. N layers of:
   a. Self-attention within each modality
   b. Cross-attention between modalities
   c. Feed-forward network with residual
4. CLS token pooling → MLP classifier

Since each modality produces a single (B, 128) vector, we tokenize by
splitting the embedding into K sub-vectors (e.g., 128/16 = 8 tokens of
dim 16) to give the transformer meaningful sequences to attend over.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Token Embedding: split (B, D) → (B, K, d_model)
# ======================================================================

class ModalityTokenizer(nn.Module):
    """Split a flat embedding into K tokens and project to d_model.

    (B, D) → (B, K+1, d_model)  where +1 is a learnable CLS token.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_tokens: int = 8,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = embed_dim // n_tokens
        assert embed_dim % n_tokens == 0, \
            f"embed_dim={embed_dim} must be divisible by n_tokens={n_tokens}"

        self.proj = nn.Linear(self.token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, embed_dim)``

        Returns:
            ``(B, K+1, d_model)`` — K content tokens + CLS token.
        """
        B = x.size(0)
        # Split into tokens: (B, embed_dim) → (B, K, token_dim)
        tokens = x.view(B, self.n_tokens, self.token_dim)
        tokens = self.proj(tokens)  # (B, K, d_model)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, K+1, d_model)
        tokens = self.norm(tokens + self.pos_embed)
        return tokens


# ======================================================================
# Cross-Attention Layer
# ======================================================================

class CrossAttentionLayer(nn.Module):
    """One layer of cross-modal attention.

    Query comes from modality A, Key/Value from modality B.
    Followed by feed-forward network with residual + layer norm.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        # Cross-attention: Q from A, KV from B
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention within modality
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, S, d_model)`` — query modality tokens.
            context: ``(B, T, d_model)`` — key/value modality tokens.

        Returns:
            Updated ``x`` with same shape.
        """
        # Self-attention
        sa_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + sa_out)

        # Cross-attention: x attends to context
        ca_out, _ = self.cross_attn(x, context, context)
        x = self.norm2(x + ca_out)

        # Feed-forward
        x = self.norm3(x + self.ff(x))
        return x


# ======================================================================
# Cross-Modal Transformer Fusion
# ======================================================================

class CrossModalTransformerFusion(nn.Module):
    """Transformer-based cross-modal fusion.

    Architecture:
    1. Tokenize each modality embedding: (B, 128) → (B, K+1, d_model)
    2. N layers of cross-attention (EEG attends to speech, speech attends to EEG)
    3. Extract CLS tokens from both
    4. Concatenate CLS tokens → MLP classifier

    Unlike GatedFusion which is limited to scalar gating, this allows
    fine-grained feature-level interactions between modalities.
    """

    def __init__(
        self,
        eeg_embed_dim: int = 128,
        speech_embed_dim: int = 128,
        n_tokens: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.15,
        modality_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob

        # Tokenizers
        self.eeg_tokenizer = ModalityTokenizer(eeg_embed_dim, n_tokens, d_model)
        self.speech_tokenizer = ModalityTokenizer(speech_embed_dim, n_tokens, d_model)

        # Cross-attention layers (interleaved)
        self.eeg_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.speech_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output dimension
        self.output_dim = d_model * 2  # CLS from both modalities

    def forward(
        self,
        eeg_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            eeg_embedding: ``(B, eeg_embed_dim)``
            speech_embedding: ``(B, speech_embed_dim)``

        Returns:
            Fused representation ``(B, d_model*2)``
        """
        # Modality dropout during training
        if self.training and self.modality_dropout_prob > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout_prob / 2:
                eeg_embedding = torch.zeros_like(eeg_embedding)
            elif r < self.modality_dropout_prob:
                speech_embedding = torch.zeros_like(speech_embedding)

        # Tokenize
        eeg_tokens = self.eeg_tokenizer(eeg_embedding)      # (B, K+1, d_model)
        speech_tokens = self.speech_tokenizer(speech_embedding)  # (B, K+1, d_model)

        # Cross-attention layers
        for eeg_layer, speech_layer in zip(self.eeg_layers, self.speech_layers):
            eeg_tokens = eeg_layer(eeg_tokens, speech_tokens)
            speech_tokens = speech_layer(speech_tokens, eeg_tokens)

        # Extract CLS tokens (index 0)
        eeg_cls = eeg_tokens[:, 0, :]    # (B, d_model)
        speech_cls = speech_tokens[:, 0, :]  # (B, d_model)

        # Concatenate
        fused = torch.cat([eeg_cls, speech_cls], dim=1)  # (B, d_model*2)
        return fused


# ======================================================================
# Transformer Fusion Classifier
# ======================================================================

class TransformerFusionClassifier(nn.Module):
    """Full cross-modal transformer classifier.

    Wraps CrossModalTransformerFusion + MLP classification head.
    Drop-in replacement for FusionClassifier with same interface.
    """

    def __init__(
        self,
        eeg_embed_dim: int = 128,
        speech_embed_dim: int = 128,
        n_tokens: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.15,
        modality_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.fusion = CrossModalTransformerFusion(
            eeg_embed_dim=eeg_embed_dim,
            speech_embed_dim=speech_embed_dim,
            n_tokens=n_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            modality_dropout_prob=modality_dropout_prob,
        )

        fused_dim = self.fusion.output_dim  # d_model * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(
        self,
        eeg_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            eeg_embedding: ``(B, eeg_embed_dim)``
            speech_embedding: ``(B, speech_embed_dim)``

        Returns:
            Raw logits ``(B, num_classes)``
        """
        fused = self.fusion(eeg_embedding, speech_embedding)
        return self.classifier(fused)
