from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .embedding import PositionalEmbedding, TokenEmbedding
from .kv_cache import KVCache
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = PositionalEmbedding(
            config.max_seq_len, config.d_model
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.ff_hidden_dim, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._init_weights)
        if getattr(config, "tie_weights", True):
            self.output_head.weight = self.token_embedding.embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def _get_position_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token_ids) + self.position_embedding(token_ids)

    def forward(
        self,
        token_ids: torch.Tensor,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache | None]:
        embeddings = self._get_position_embeddings(token_ids)
        seq_len = token_ids.size(1)
        mask = self._build_causal_mask(seq_len, token_ids.device)
        if past_key_values is None:
            cache = KVCache() if use_cache else None
        else:
            cache = past_key_values

        x = embeddings
        layer_cache: list[tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else []
        for block in self.blocks:
            x, key_values = block(x, mask=mask, past_key_values=None)
            if use_cache and key_values is not None:
                layer_cache.append(key_values)
        x = self.norm(x)
        logits = self.output_head(x)
        if use_cache and cache is not None:
            cache.layers = layer_cache
        if use_cache:
            return logits, cache
        return logits

    def compute_loss(
        self, input_ids: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = self(input_ids, use_cache=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                result = self(generated, use_cache=False)
                if isinstance(result, tuple):
                    logits, _ = result
                else:
                    logits = result
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-9)
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    values, indices = torch.topk(next_token_logits, top_k)
                    filtered = torch.full_like(next_token_logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    next_token_logits = filtered
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = cumulative_probs.cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., 1:
                    ].clone()
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, -float("inf")
                    )
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
            return generated

    def save_pretrained(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.state_dict(), "config": self.config}, path)

    def load_pretrained(self, path: str | os.PathLike[str]) -> None:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
