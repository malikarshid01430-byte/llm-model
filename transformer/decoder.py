from __future__ import annotations

import torch
import torch.nn as nn

from model.embedding import PositionalEmbedding, TokenEmbedding
from transformer.decoder_block import DecoderBlock


class DecoderTransformer(nn.Module):
    """A stack of decoder blocks that forms the core of the GPT-style model."""

    def __init__(self, config) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = PositionalEmbedding(
            config.max_seq_len, config.d_model
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config.d_model, config.n_heads, config.ff_hidden_dim, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_weights:
            self.output_head.weight = self.token_embedding.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(token_ids) + self.position_embedding(
            token_ids
        )
        x = embeddings
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_head(x)
