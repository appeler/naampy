"""Character-level bidirectional LSTM for first-name pattern scores."""

from __future__ import annotations

import torch
import torch.nn as nn

CHARACTER_TO_INDEX: dict[str, int] = {"<PAD>": 0}
CHARACTER_TO_INDEX.update(
    {
        character: position + 1
        for position, character in enumerate("abcdefghijklmnopqrstuvwxyz")
    }
)
CHARACTER_VOCABULARY_SIZE = len(CHARACTER_TO_INDEX)

LSTM_EMBEDDING_DIMENSION = 64
LSTM_HIDDEN_DIMENSION = 256
LSTM_LAYER_COUNT = 2
LSTM_DROPOUT_PROBABILITY = 0.2


def encode_normalized_name(normalized_name: str) -> list[int]:
    """Encode one validated lowercase ASCII name without dropping characters."""
    return [CHARACTER_TO_INDEX[character] for character in normalized_name]


def pad_encoded_names(
    encoded_names: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad encoded names and return their unpadded lengths."""
    name_lengths = torch.tensor(
        [len(encoded_name) for encoded_name in encoded_names], dtype=torch.long
    )
    maximum_name_length = int(name_lengths.max()) if len(encoded_names) else 0
    padded_names = torch.zeros(
        len(encoded_names), maximum_name_length, dtype=torch.long
    )
    for position, encoded_name in enumerate(encoded_names):
        padded_names[position, : len(encoded_name)] = torch.tensor(
            encoded_name, dtype=torch.long
        )
    return padded_names, name_lengths


class CharacterBiLSTM(nn.Module):
    """Map encoded names to raw logits with a bidirectional LSTM."""

    def __init__(
        self,
        vocabulary_size: int,
        output_dimension: int = 1,
        embedding_dimension: int = LSTM_EMBEDDING_DIMENSION,
        hidden_dimension: int = LSTM_HIDDEN_DIMENSION,
        layer_count: int = LSTM_LAYER_COUNT,
        dropout_probability: float = LSTM_DROPOUT_PROBABILITY,
    ) -> None:
        """Build the embedding, BiLSTM, and output projection layers.

        Args:
            vocabulary_size: Number of distinct character indices.
            output_dimension: Number of output logits.
            embedding_dimension: Character embedding dimension.
            hidden_dimension: LSTM hidden state size per direction.
            layer_count: Number of stacked LSTM layers.
            dropout_probability: Dropout between recurrent layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dimension,
            hidden_dimension,
            num_layers=layer_count,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_probability if layer_count > 1 else 0.0,
        )
        self.fc = nn.Linear(2 * hidden_dimension, output_dimension)

    def forward(
        self, encoded_names: torch.Tensor, name_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return raw logits for one padded batch.

        Args:
            encoded_names: Padded character indices with shape ``[B, T]``.
            name_lengths: Unpadded sequence length for each row.

        Returns:
            Raw logits with shape ``[B, output_dimension]``.
        """
        embedded_names = self.embedding(encoded_names)
        packed_names = nn.utils.rnn.pack_padded_sequence(
            embedded_names,
            name_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_states, _) = self.lstm(packed_names)
        final_bidirectional_state = torch.cat(
            [hidden_states[-2], hidden_states[-1]], dim=1
        )
        logits: torch.Tensor = self.fc(final_bidirectional_state)
        return logits
