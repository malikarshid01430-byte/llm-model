from __future__ import annotations

import math

import torch


class MetricsTracker:
    def __init__(self) -> None:
        self.losses: list[float] = []

    def update(self, loss: torch.Tensor) -> None:
        self.losses.append(float(loss.detach().cpu()))

    @property
    def mean(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else float("inf")

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean) if self.mean != float("inf") else float("inf")
