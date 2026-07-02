from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 2000
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_hidden_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    tie_weights: bool = True
    eps: float = 1e-5
