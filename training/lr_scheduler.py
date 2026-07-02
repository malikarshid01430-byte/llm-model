from __future__ import annotations

import math


class CosineWarmupScheduler:
    """A simple cosine warmup learning rate scheduler."""

    def __init__(self, initial_lr: float, warmup_steps: int, total_steps: int) -> None:
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.initial_lr * (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return self.initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
