"""CNN-LSTM Speech Encoder for IEMOCAP MFCC sequences.

Architecture:
    MFCC input (B, T, 120) → Conv2D layers → BiLSTM → mean pooling → FC → embedding (128)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpeechEncoder(nn.Module):
    """CNN-LSTM encoder: MFCC sequences → emotion embedding."""

    def __init__(
        self,
        n_features: int = 120,
        cnn_channels: list[int] | None = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        embedding_dim: int = 128,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.pooling = pooling

        # ----- CNN layers -----
        cnn_layers: list[nn.Module] = []
        in_ch = 1  # single-channel "image" of (T, n_features)
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Compute the feature dimension after CNN
        # After 3 MaxPool2d(2,2): frequency dim → n_features / 8
        # Each pool halves both time and frequency dims
        self._freq_after_cnn = n_features // (2 ** len(cnn_channels))
        self._lstm_input_dim = cnn_channels[-1] * self._freq_after_cnn

        # ----- LSTM -----
        self.lstm = nn.LSTM(
            input_size=self._lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_size * (2 if lstm_bidirectional else 1)

        # ----- FC projection -----
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, T_max, n_features)`` — e.g. ``(B, 800, 120)``

        Returns:
            Embedding ``(batch, embedding_dim)``
        """
        # (B, T, F) → (B, 1, T, F) for Conv2d
        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(x)  # (B, C_last, T', F')

        # Reshape for LSTM: merge channel and frequency dims
        B, C, T_new, F_new = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T', C, F')
        x = x.view(B, T_new, C * F_new)  # (B, T', C*F')

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T', 2*H)

        # Temporal pooling
        if self.pooling == "mean":
            x = lstm_out.mean(dim=1)  # (B, 2*H)
        elif self.pooling == "last":
            x = lstm_out[:, -1, :]
        else:
            x = lstm_out.mean(dim=1)

        # FC projection
        return self.fc(x)  # (B, embedding_dim)
