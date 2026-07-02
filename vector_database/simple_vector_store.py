from __future__ import annotations

from typing import List, Tuple


class SimpleVectorStore:
    """A minimal vector store implementation for RAG-style experimentation."""

    def __init__(self) -> None:
        self._items: List[Tuple[str, List[float]]] = []

    def add(self, text: str, vector: List[float]) -> None:
        self._items.append((text, vector))

    def search(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        return [
            text
            for text, _ in sorted(
                self._items,
                key=lambda item: sum(abs(q - v) for q, v in zip(query_vector, item[1])),
            )[:top_k]
        ]
