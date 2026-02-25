"""Cross-Modal Mutual Attention (CMMA) Fusion — v5.7.

Novel architecture contributions
─────────────────────────────────
1. **Cross-Modal Mutual Attention (CMMA)**: Bidirectional cross-attention
   where EEG attends to speech AND speech attends to EEG simultaneously,
   enabling each modality to discover complementary features from the other
   *before* fusion.

2. **Emotion-Aware Gating (EAG)**: Sigmoid-based per-class modality gates.
   Each class has a learnable gate logit → sigmoid → determines EEG/speech
   balance.  Strong gradient flow (no softmax dead zone).
   Additionally, input-dependent adjustment via a lightweight gate network
   modulates the per-class weights based on the actual input features.

3. **Auxiliary Unimodal Losses**: Each modality gets a small classification
   head that provides gradient signal directly to each encoder, ensuring
   both produce class-discriminative features independently.

4. **End-to-End Joint Training**: Encoders fine-tuned with very conservative
   LR (0.05x), with frozen warmup phase to stabilize CMMA before encoder
   gradients are enabled.

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
                                 ├─→ [Aux EEG Head]   ─→ eeg_logits  (auxiliary)
                                 └─→ [Aux Speech Head] ─→ sp_logits   (auxiliary)

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
        # Bias initialized to -2.0 so gate starts near 0.12 (conservative:
        # mostly self-attention initially, learns to open as needed)
        self.cross_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        # Initialize gate bias to -2.0 for conservative start
        nn.init.constant_(self.cross_gate[0].bias, -2.0)

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
    """Learns per-class modality importance weights via sigmoid gating.

    v5.3: Annealed teacher forcing + gate diversity regularization.

    Each class has ONE learnable gate logit.  During training, the model
    blends teacher-forced gating (using ground-truth labels) with
    probe-based gating.  The blend ratio (tf_ratio) is annealed from
    1.0 → 0.0 over training, so:
      - Early epochs: strong per-class gradient (TF-dominant)
      - Late epochs:  probe-based (matches inference distribution)

    A gate diversity loss (-std(gate_logits)) prevents logits from
    collapsing back to equal values.

    Only ONE sigmoid is applied (on the blended logit + input adjustment).
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
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

        # Per-class gate logits: ONE scalar per class
        self.class_gate_logits = nn.Parameter(torch.zeros(num_classes))

        # Input-dependent gate adjustment (adds to per-class gate)
        self.input_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.d_model = d_model
        self.num_classes = num_classes

    def forward(
        self,
        eeg_feat: torch.Tensor,
        speech_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        tf_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            eeg_feat: (B, d_model) — enhanced EEG representation
            speech_feat: (B, d_model) — enhanced speech representation
            labels: (B,) int — ground-truth class indices (training only)
            tf_ratio: float in [0, 1] — teacher forcing ratio.
                      1.0 = fully teacher-forced, 0.0 = fully probe-based.
                      Annealed during training from 1.0 → 0.0.

        Returns:
            fused: (B, d_model) — emotion-aware fused representation
        """
        combined = torch.cat([eeg_feat, speech_feat], dim=-1)  # (B, 2*d_model)

        # Input-dependent adjustment (shared for both modes)
        input_adj = self.input_gate(combined)  # (B, 1)

        # ── Always compute probe-based logit (needed for inference) ──
        emotion_logits = self.emotion_probe(combined)  # (B, C)
        emotion_probs = F.softmax(emotion_logits, dim=-1)  # (B, C)
        probe_logit = (emotion_probs * self.class_gate_logits.unsqueeze(0)).sum(
            dim=-1, keepdim=True
        )  # (B, 1)

        if labels is not None and tf_ratio > 0 and self.training:
            # ── Blend teacher-forced + probe-based ──
            tf_logit = self.class_gate_logits[labels].unsqueeze(-1)  # (B, 1)
            base_logit = tf_ratio * tf_logit + (1 - tf_ratio) * probe_logit
        else:
            # ── Inference or tf_ratio=0: fully probe-based ──
            base_logit = probe_logit

        # Single sigmoid: blended logit + adjustment → [0, 1]
        alpha = torch.sigmoid(base_logit + input_adj)  # (B, 1)

        # Emotion-aware fusion
        fused = alpha * eeg_feat + (1 - alpha) * speech_feat  # (B, d_model)

        return fused

    def gate_diversity_loss(self) -> torch.Tensor:
        """Regularizer that encourages per-class gate logits to diverge.

        Returns -std(gate_logits).  Minimizing this loss maximizes the
        standard deviation of the gate logits, preventing them from
        collapsing back to uniform 0.50/0.50.
        """
        return -torch.std(self.class_gate_logits)

    def get_emotion_logits(
        self,
        eeg_feat: torch.Tensor,
        speech_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return emotion probe logits for auxiliary loss."""
        combined = torch.cat([eeg_feat, speech_feat], dim=-1)
        return self.emotion_probe(combined)


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

        # --- Auxiliary unimodal classification heads ---
        # These provide direct gradient signal to each encoder,
        # ensuring both produce class-discriminative features
        self.aux_eeg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self.aux_speech_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

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
        return_aux: bool = False,
        labels: Optional[torch.Tensor] = None,
        tf_ratio: float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            eeg_embedding: (B, eeg_embed_dim) — from EEG encoder
            speech_embedding: (B, speech_embed_dim) — from Speech encoder
            return_aux: if True, also return auxiliary logits dict
            labels: (B,) int — ground-truth labels for teacher-forced gating
            tf_ratio: float in [0, 1] — teacher forcing blend ratio

        Returns:
            logits: (B, num_classes)
            [optional] aux_dict: {'eeg_logits', 'speech_logits', 'probe_logits'}
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
        fused = self.emotion_gate(
            eeg_cls, sp_cls, labels=labels, tf_ratio=tf_ratio,
        )  # (B, d_model)

        # --- Classify ---
        logits = self.classifier(fused)  # (B, num_classes)

        if return_aux:
            aux = {
                'eeg_logits': self.aux_eeg_head(eeg_cls),
                'speech_logits': self.aux_speech_head(sp_cls),
                'probe_logits': self.emotion_gate.get_emotion_logits(eeg_cls, sp_cls),
            }
            return logits, aux

        return logits

    def get_modality_weights(self) -> torch.Tensor:
        """Return the learned per-class modality weights for interpretability.

        Returns:
            (num_classes, 2) tensor — [eeg_weight, speech_weight] per class
        """
        alpha = torch.sigmoid(self.emotion_gate.class_gate_logits).detach()
        return torch.stack([alpha, 1 - alpha], dim=-1)  # (num_classes, 2)

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
