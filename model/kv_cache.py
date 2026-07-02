from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class KVCache:
    layers: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.layers.append((key, value))

    def clear(self) -> None:
        self.layers.clear()

    def __len__(self) -> int:
        return len(self.layers)

    def get_layer(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if index < 0 or index >= len(self.layers):
            return None
        return self.layers[index]
