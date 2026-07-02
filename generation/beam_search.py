from __future__ import annotations

from typing import List, Tuple

import torch


class BeamSearchDecoder:
    """A lightweight beam-search decoder for sequence generation."""

    def __init__(self, beam_width: int = 4) -> None:
        self.beam_width = beam_width

    def decode(self, logits: torch.Tensor) -> List[Tuple[float, List[int]]]:
        probs = torch.softmax(logits, dim=-1)
        ranked = sorted(
            ((float(prob), [idx]) for idx, prob in enumerate(probs[0])), reverse=True
        )
        return ranked[: self.beam_width]
