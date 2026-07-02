from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.register_buffer(
            "position_encodings",
            self._build_position_encodings(),
            persistent=False,
        )

    def _build_position_encodings(self) -> torch.Tensor:
        position = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(self.max_seq_len, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position_encodings = cast(torch.Tensor, self.position_encodings)
        if position_encodings.size(1) < seq_len:
            raise ValueError("Sequence length exceeds configured maximum")
        return position_encodings[:, :seq_len, :]
