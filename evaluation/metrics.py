from __future__ import annotations

import math

import torch
from torch import nn


def calculate_perplexity(
    model: nn.Module, inputs: torch.Tensor, device: str = "cpu"
) -> float:
    """Compute perplexity for a model on a batch of tokens."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device))
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), inputs.to(device).view(-1)
        )
    return float(math.exp(loss.item()))


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute simple accuracy for classification-style predictions."""
    correct = (predictions.argmax(dim=-1) == targets).sum().item()
    total = targets.numel()
    return correct / total if total else 0.0
