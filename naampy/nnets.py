"""Char-level bidirectional LSTM gender model for naampy (first name -> P(female)).

Self-contained (torch only). Mirrors the instate CharBiLSTM; a single output unit + sigmoid
gives P(female). Trained/served with a 26-letter vocab (a-z, ``<PAD>`` = 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Character vocabulary: a-z -> 1..26, <PAD> = 0.
CHAR_TO_IDX: dict[str, int] = {"<PAD>": 0}
CHAR_TO_IDX.update({c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})
VOCAB_SIZE = len(CHAR_TO_IDX)  # 27

# Model configuration.
LSTM_EMB = 64
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2


def encode_name(name: str) -> list[int]:
    """Map a (cleaned, lowercase) name to ``CHAR_TO_IDX`` indices, dropping out-of-vocab chars."""
    return [CHAR_TO_IDX[c] for c in name if c in CHAR_TO_IDX]


def pad_encoded(encoded: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of index-lists into ``(LongTensor [B, T], lengths LongTensor [B])`` (PAD=0)."""
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    maxlen = int(lengths.max()) if len(encoded) else 0
    x = torch.zeros(len(encoded), maxlen, dtype=torch.long)
    for i, e in enumerate(encoded):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return x, lengths


class CharBiLSTM(nn.Module):
    """Embedding -> packed BiLSTM -> Linear over ``num_classes`` raw logits."""

    def __init__(
        self,
        num_chars: int,
        num_classes: int = 1,
        embedding_dim: int = LSTM_EMB,
        hidden_dim: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)
