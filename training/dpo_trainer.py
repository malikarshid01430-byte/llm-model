from __future__ import annotations

from typing import List, Tuple


class DPOTrainer:
    """A minimal DPO-style trainer scaffold for preference optimization."""

    def __init__(self) -> None:
        self.examples: List[Tuple[str, str]] = []

    def add_example(self, chosen: str, rejected: str) -> None:
        self.examples.append((chosen, rejected))

    def train(self) -> int:
        return len(self.examples)
