from __future__ import annotations

import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3))
                )
            )
        )


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
