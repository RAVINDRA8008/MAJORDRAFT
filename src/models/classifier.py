"""Softmax classification wrapper.

Thin wrapper combining an encoder with a linear classification head.
Primarily used for standalone EEG or speech classification baselines.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Linear classification head on top of an embedding.

    This is used when training a single modality in isolation
    (e.g., EEG-only baseline).
    """

    def __init__(self, embedding_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: ``(batch, embedding_dim)``

        Returns:
            Raw logits ``(batch, num_classes)``
        """
        return self.head(embedding)


class EncoderClassifier(nn.Module):
    """Convenience module: encoder + classification head.

    Forward pass goes ``input → encoder → linear head → logits``.
    """

    def __init__(self, encoder: nn.Module, embedding_dim: int = 128, num_classes: int = 4) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return self.classifier(embedding)
