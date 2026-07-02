from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _apply_projection(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        projected = proj(x)
        projected = projected.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return projected.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        q = self._apply_projection(x, self.q_proj)
        k = self._apply_projection(x, self.k_proj)
        v = self._apply_projection(x, self.v_proj)

        if past_key_values is None:
            key_states = k
            value_states = v
        else:
            past_key, past_value = past_key_values
            key_states = torch.cat([past_key, k], dim=2)
            value_states = torch.cat([past_value, v], dim=2)

        scores = torch.matmul(q, key_states.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value_states)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(context), (key_states, value_states)
