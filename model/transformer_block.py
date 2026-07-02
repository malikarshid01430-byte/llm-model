from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardBlock


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ff_hidden_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = FeedForwardBlock(d_model, ff_hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, key_values = self.attn(
            self.attn_norm(x), mask=mask, past_key_values=past_key_values
        )
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ff_norm(x))
        x = x + self.dropout(ff_out)
        return x, key_values
