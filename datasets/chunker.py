from __future__ import annotations

from typing import List


class DocumentChunker:
    """Split text documents into chunk-sized segments."""

    def __init__(self, chunk_size: int = 512, overlap: int = 32) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            step = max(1, self.chunk_size - self.overlap)
            start = min(len(words) - 1, start + step)
        return chunks
