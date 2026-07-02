from typing import List

import chromadb
from chromadb.config import Settings


class VectorStore:
    """Chromadb-powered vector store for embeddings and semantic search."""

    def __init__(self, collection_name: str = "eduai_documents") -> None:
        self.client = chromadb.Client(
            Settings(
                chroma_api_impl="chromadb.db.sqlite3",
                persist_directory="./vector_store",
            )
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        ids = [f"doc-{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[metadata] * len(texts),
        )

    def similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        docs = []
        for idx, text in enumerate(
            results["documents"][0] if results["documents"] else []
        ):
            docs.append(
                {
                    "id": results["ids"][0][idx],
                    "text": text,
                    "distance": results["distances"][0][idx],
                }
            )
        return docs
