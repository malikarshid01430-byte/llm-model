from __future__ import annotations

from typing import List

from datasets.chunker import DocumentChunker


class Retriever:
    """A simple semantic-style retriever scaffold for RAG workflows."""

    def __init__(self, documents: List[str] | None = None) -> None:
        self.documents = documents or []
        self.chunker = DocumentChunker()

    def index(self, documents: List[str]) -> None:
        self.documents.extend(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_terms = set(query.lower().split())
        scored = []
        for document in self.documents:
            overlap = len(query_terms.intersection(set(document.lower().split())))
            scored.append((overlap, document))
        return [doc for _, doc in sorted(scored, reverse=True)[:top_k]]
