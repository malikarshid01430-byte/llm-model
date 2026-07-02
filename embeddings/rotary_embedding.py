from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """A simple rotary position embedding scaffold."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(
            seq_len, dtype=torch.float32, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device)
            * -(math.log(10000.0) / self.d_model)
        )
        angles = position * div_term
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return torch.stack([cos, sin], dim=-1)
