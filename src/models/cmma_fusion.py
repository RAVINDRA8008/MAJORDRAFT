"""Cross-Modal Mutual Attention (CMMA) Fusion — v5.

Novel architecture contributions
─────────────────────────────────
1. **Cross-Modal Mutual Attention (CMMA)**: Bidirectional cross-attention
   where EEG attends to speech AND speech attends to EEG simultaneously,
   enabling each modality to discover complementary features from the other
   *before* fusion.  This goes beyond v4's approach where encoders were
   completely independent.

2. **Emotion-Aware Gating (EAG)**: A learned per-class modality weighting
   mechanism.  Different emotions rely differently on each modality (e.g.,
   "Angry" may be more salient in speech prosody while "Sad" may show
   stronger EEG signatures).  EAG learns a (num_classes x 2) gate matrix
   that dynamically re-weights the modality contributions.

3. **End-to-End Joint Training**: Unlike v4 where encoders were frozen
   and only the fusion head was trained, v5 fine-tunes encoders jointly
   with the CMMA layers and classification head.  A discriminative learning
   rate schedule (lower LR for encoders, higher for new layers) prevents
   catastrophic forgetting of pretrained representations.

Architecture
────────────
    Raw EEG (B, 160)
      │
      ▼
    [EEG Encoder]  ──→  eeg_emb (B, 128)
      │                       │
      │    ┌──────────────────┤
      │    │  Cross-Modal     │
      │    │  Mutual Attn     │  (N layers, bidirectional)
      │    │  (CMMA)          │
      │    └──────────────────┤
      │                       │
    [Speech Encoder] ──→ sp_emb (B, 128)
      │
    Raw Speech (B, T, 120)

    eeg_enhanced (B, d_model) ──┐
                                 ├─→ [Emotion-Aware Gate] ─→ [Classifier] ─→ logits
    sp_enhanced  (B, d_model) ──┘

Reference
─────────
Inspired by ViLBERT (Lu et al., 2019) and Perceiver (Jaegle et al., 2021)
but adapted for EEG+Speech emotion recognition with novel gating.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Cross-Modal Mutual Attention Block
# ======================================================================

class CMMABlock(nn.Module):
    """One layer of Cross-Modal Mutual Attention.

    Both modalities perform:
    1. Self-attention (intra-modal context)
    2. Cross-attention (inter-modal context) — Q from self, KV from other
    3. Gated residual fusion — learns how much cross-modal info to inject
    4. Feed-forward network with pre-norm

    The gated residual is novel: instead of simply adding the cross-attention
    output, a sigmoid gate controls the mixing ratio, allowing the model to
    learn *how much* to trust the other modality per token.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # --- Self-attention (intra-modal) ---
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_sa = nn.LayerNorm(d_model)

        # --- Cross-attention (inter-modal) ---
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ca = nn.LayerNorm(d_model)

        # --- Gated residual for cross-attention ---
        self.cross_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # --- Feed-forward ---
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, d_model) — this modality's tokens
            context: (B, T, d_model) — other modality's tokens

        Returns:
            Updated x: (B, S, d_model)
        """
        # 1. Self-attention with pre-norm + residual
        h = self.norm_sa(x)
        sa_out, _ = self.self_attn(h, h, h)
        x = x + sa_out

        # 2. Cross-attention with gated residual
        h = self.norm_ca(x)
        ca_out, _ = self.cross_attn(h, context, context)

        # Gate: learn how much cross-modal info to inject
        gate = self.cross_gate(torch.cat([x, ca_out], dim=-1))  # (B, S, d_model)
        x = x + gate * ca_out

        # 3. Feed-forward with pre-norm + residual
        h = self.norm_ff(x)
        x = x + self.ff(h)

        return x


# ======================================================================
# Emotion-Aware Gating
# ======================================================================

class EmotionAwareGating(nn.Module):
    """Learns per-class modality importance weights.

    Instead of treating EEG and speech equally, this module estimates
    a probability distribution over emotions, then uses that to look up
    learned modality weights for each class.  The final representation
    is a weighted combination of EEG and speech features, where the
    weights are emotion-dependent.

    This is novel because existing fusion methods use static or
    input-dependent gates, but not *class-dependent* gates that learn
    which modality is more informative for each specific emotion.

    Forward:
        1. Estimate preliminary class probs from fused features
        2. Look up per-class modality weights
        3. Compute weighted sum: out = w_eeg * eeg + w_sp * speech
    """

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        # Preliminary emotion estimator (lightweight)
        self.emotion_probe = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

        # Per-class modality weights: (num_classes, 2) — [eeg_weight, speech_weight]
        # Initialized to 0.5/0.5 (equal) so training starts from balanced fusion
        self.class_modality_weights = nn.Parameter(
            torch.full((num_classes, 2), 0.5)
        )

        self.d_model = d_model
        self.num_classes = num_classes

    def forward(
        self,
        eeg_feat: torch.Tensor,
        speech_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            eeg_feat: (B, d_model) — enhanced EEG representation
            speech_feat: (B, d_model) — enhanced speech representation

        Returns:
            fused: (B, d_model) — emotion-aware fused representation
        """
        # Preliminary emotion estimate
        combined = torch.cat([eeg_feat, speech_feat], dim=-1)  # (B, 2*d_model)
        emotion_logits = self.emotion_probe(combined)  # (B, num_classes)
        emotion_probs = F.softmax(emotion_logits, dim=-1)  # (B, num_classes)

        # Normalize class modality weights to sum to 1 per class
        weights = F.softmax(self.class_modality_weights, dim=-1)  # (num_classes, 2)

        # Weighted combination: (B, num_classes) @ (num_classes, 2) → (B, 2)
        modality_weights = emotion_probs @ weights  # (B, 2)

        w_eeg = modality_weights[:, 0:1]    # (B, 1)
        w_speech = modality_weights[:, 1:2]  # (B, 1)

        # Emotion-aware fusion
        fused = w_eeg * eeg_feat + w_speech * speech_feat  # (B, d_model)

        return fused


# ======================================================================
# CMMA Fusion Classifier (v5 main module)
# ======================================================================

class CMMAFusionClassifier(nn.Module):
    """Cross-Modal Mutual Attention Fusion Classifier (v5).

    Complete classification pipeline:
    1. Tokenize encoder embeddings → sequences
    2. N layers of bidirectional CMMA (EEG ↔ Speech)
    3. Pool to single vectors per modality
    4. Emotion-Aware Gating for dynamic modality weighting
    5. Classification head

    This is the drop-in replacement for TransformerFusionClassifier,
    with the same input interface: (eeg_embedding, speech_embedding) → logits.
    """

    def __init__(
        self,
        eeg_embed_dim: int = 128,
        speech_embed_dim: int = 128,
        n_tokens: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_cmma_layers: int = 3,
        ff_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.1,
        modality_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
        self.d_model = d_model
        self.n_tokens = n_tokens

        # --- Tokenizers: (B, embed_dim) → (B, K+1, d_model) ---
        self.eeg_token_dim = eeg_embed_dim // n_tokens
        self.sp_token_dim = speech_embed_dim // n_tokens

        self.eeg_proj = nn.Linear(self.eeg_token_dim, d_model)
        self.sp_proj = nn.Linear(self.sp_token_dim, d_model)

        # Learnable CLS tokens
        self.eeg_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sp_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encodings
        self.eeg_pos = nn.Parameter(torch.randn(1, n_tokens + 1, d_model) * 0.02)
        self.sp_pos = nn.Parameter(torch.randn(1, n_tokens + 1, d_model) * 0.02)

        # Input norms
        self.eeg_norm = nn.LayerNorm(d_model)
        self.sp_norm = nn.LayerNorm(d_model)

        # --- CMMA layers (bidirectional) ---
        self.eeg_cmma_layers = nn.ModuleList([
            CMMABlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_cmma_layers)
        ])
        self.sp_cmma_layers = nn.ModuleList([
            CMMABlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_cmma_layers)
        ])

        # Output norms
        self.eeg_out_norm = nn.LayerNorm(d_model)
        self.sp_out_norm = nn.LayerNorm(d_model)

        # --- Emotion-Aware Gating ---
        self.emotion_gate = EmotionAwareGating(d_model, num_classes)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _tokenize(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        cls_token: nn.Parameter,
        pos_embed: nn.Parameter,
        norm: nn.LayerNorm,
        n_tokens: int,
        token_dim: int,
    ) -> torch.Tensor:
        """(B, embed_dim) → (B, K+1, d_model)"""
        B = x.size(0)
        tokens = x.view(B, n_tokens, token_dim)  # (B, K, token_dim)
        tokens = proj(tokens)  # (B, K, d_model)
        cls = cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, K+1, d_model)
        tokens = norm(tokens + pos_embed)
        return tokens

    def forward(
        self,
        eeg_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            eeg_embedding: (B, eeg_embed_dim) — from EEG encoder
            speech_embedding: (B, speech_embed_dim) — from Speech encoder

        Returns:
            logits: (B, num_classes)
        """
        # --- Modality dropout (training only) ---
        if self.training and self.modality_dropout_prob > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout_prob / 2:
                eeg_embedding = torch.zeros_like(eeg_embedding)
            elif r < self.modality_dropout_prob:
                speech_embedding = torch.zeros_like(speech_embedding)

        # --- Tokenize ---
        eeg_tokens = self._tokenize(
            eeg_embedding, self.eeg_proj, self.eeg_cls,
            self.eeg_pos, self.eeg_norm, self.n_tokens, self.eeg_token_dim,
        )
        sp_tokens = self._tokenize(
            speech_embedding, self.sp_proj, self.sp_cls,
            self.sp_pos, self.sp_norm, self.n_tokens, self.sp_token_dim,
        )

        # --- CMMA: bidirectional cross-attention ---
        for eeg_layer, sp_layer in zip(self.eeg_cmma_layers, self.sp_cmma_layers):
            # EEG attends to speech, speech attends to EEG (simultaneously)
            eeg_tokens_new = eeg_layer(eeg_tokens, sp_tokens)
            sp_tokens_new = sp_layer(sp_tokens, eeg_tokens)
            eeg_tokens = eeg_tokens_new
            sp_tokens = sp_tokens_new

        # --- Pool: extract CLS tokens ---
        eeg_cls = self.eeg_out_norm(eeg_tokens[:, 0, :])  # (B, d_model)
        sp_cls = self.sp_out_norm(sp_tokens[:, 0, :])      # (B, d_model)

        # --- Emotion-Aware Gating ---
        fused = self.emotion_gate(eeg_cls, sp_cls)  # (B, d_model)

        # --- Classify ---
        logits = self.classifier(fused)  # (B, num_classes)

        return logits

    def get_modality_weights(self) -> torch.Tensor:
        """Return the learned per-class modality weights for interpretability.

        Returns:
            (num_classes, 2) tensor — [eeg_weight, speech_weight] per class
        """
        return F.softmax(self.emotion_gate.class_modality_weights, dim=-1).detach()

    def get_attention_maps(
        self,
        eeg_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
    ) -> dict:
        """Extract cross-attention maps for interpretability.

        Returns dict with 'eeg_to_speech' and 'speech_to_eeg' attention
        weights from the last CMMA layer.
        """
        # Tokenize
        eeg_tokens = self._tokenize(
            eeg_embedding, self.eeg_proj, self.eeg_cls,
            self.eeg_pos, self.eeg_norm, self.n_tokens, self.eeg_token_dim,
        )
        sp_tokens = self._tokenize(
            speech_embedding, self.sp_proj, self.sp_cls,
            self.sp_pos, self.sp_norm, self.n_tokens, self.sp_token_dim,
        )

        # Run through all but last layer
        for eeg_layer, sp_layer in zip(
            self.eeg_cmma_layers[:-1], self.sp_cmma_layers[:-1]
        ):
            eeg_new = eeg_layer(eeg_tokens, sp_tokens)
            sp_new = sp_layer(sp_tokens, eeg_tokens)
            eeg_tokens, sp_tokens = eeg_new, sp_new

        # Last layer — capture attention weights
        last_eeg = self.eeg_cmma_layers[-1]
        last_sp = self.sp_cmma_layers[-1]

        # Self-attn
        h = last_eeg.norm_sa(eeg_tokens)
        _, eeg_sa_weights = last_eeg.self_attn(h, h, h)

        h = last_sp.norm_sa(sp_tokens)
        _, sp_sa_weights = last_sp.self_attn(h, h, h)

        # Cross-attn
        h_eeg = last_eeg.norm_ca(eeg_tokens)
        _, eeg_to_sp_weights = last_eeg.cross_attn(h_eeg, sp_tokens, sp_tokens)

        h_sp = last_sp.norm_ca(sp_tokens)
        _, sp_to_eeg_weights = last_sp.cross_attn(h_sp, eeg_tokens, eeg_tokens)

        return {
            'eeg_self_attn': eeg_sa_weights,
            'speech_self_attn': sp_sa_weights,
            'eeg_to_speech': eeg_to_sp_weights,
            'speech_to_eeg': sp_to_eeg_weights,
        }
