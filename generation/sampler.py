from __future__ import annotations

import torch


def top_k_top_p_filter(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """Apply top-k and top-p filtering to logits."""
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, indices, values)
        logits = filtered

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., 1:].clone()
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    return logits
