from __future__ import annotations

import torch
import torch.nn as nn

from attention.causal_attention import CausalAttention
from layers.layer_norm import LayerNorm
from model.feed_forward import FeedForwardBlock


class DecoderBlock(nn.Module):
    """A manually implemented decoder block for a GPT-style model."""

    def __init__(
        self, d_model: int, n_heads: int, ff_hidden_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attention = CausalAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForwardBlock(d_model, ff_hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        return x
