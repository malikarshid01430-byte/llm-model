from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim)
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))
