from __future__ import annotations

from typing import List


class PPOTrainer:
    """A minimal PPO-style trainer scaffold for RLHF experimentation."""

    def __init__(self) -> None:
        self.rewards: List[float] = []

    def update(self, reward: float) -> None:
        self.rewards.append(reward)

    def get_average_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
