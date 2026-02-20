"""Late Fusion classifier — combines EEG and speech embeddings.

Concatenates modality embeddings → MLP → class logits.
Supports modality dropout for robustness training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionClassifier(nn.Module):
    """Late fusion MLP: EEG embedding + speech embedding → class logits."""

    def __init__(
        self,
        eeg_embed_dim: int = 128,
        speech_embed_dim: int = 128,
        hidden_dims: list[int] | None = None,
        num_classes: int = 4,
        dropout: list[float] | None = None,
        modality_dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if dropout is None:
            dropout = [0.5, 0.3]

        self.modality_dropout_prob = modality_dropout_prob
        self.eeg_embed_dim = eeg_embed_dim
        self.speech_embed_dim = speech_embed_dim

        input_dim = eeg_embed_dim + speech_embed_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim, drop in zip(hidden_dims, dropout):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
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
            Raw logits ``(batch, num_classes)`` — apply softmax externally
            or use ``nn.CrossEntropyLoss`` which handles it.
        """
        # Modality dropout during training
        if self.training and self.modality_dropout_prob > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout_prob / 2:
                eeg_embedding = torch.zeros_like(eeg_embedding)
            elif r < self.modality_dropout_prob:
                speech_embedding = torch.zeros_like(speech_embedding)

        fused = torch.cat([eeg_embedding, speech_embedding], dim=1)
        return self.classifier(fused)
