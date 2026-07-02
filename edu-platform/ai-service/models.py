from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Sentence Transformers embeddings wrapper for document and query encoding."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
