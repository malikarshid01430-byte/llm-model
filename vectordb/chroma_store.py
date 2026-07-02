from __future__ import annotations

from typing import List, Tuple


class ChromaStore:
    """A lightweight Chroma-like vector store scaffold."""

    def __init__(self) -> None:
        self._items: List[Tuple[str, List[float]]] = []

    def add(self, text: str, vector: List[float]) -> None:
        self._items.append((text, vector))

    def search(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        scored = []
        for text, vector in self._items:
            distance = sum(abs(q - v) for q, v in zip(query_vector, vector))
            scored.append((distance, text))
        return [text for _, text in sorted(scored)[:top_k]]
