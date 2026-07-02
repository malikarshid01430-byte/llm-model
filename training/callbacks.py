from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best_score: float | None = None
    wait: int = 0
    stopped: bool = False

    def update(self, value: float) -> bool:
        if self.best_score is None or value < self.best_score - self.min_delta:
            self.best_score = value
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped = True
            return True
        return False
