from .models import EmbeddingModel
from .vector_store import VectorStore


class RAGPipeline:
    """Core Retrieval Augmented Generation service implementation."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = EmbeddingModel()

    def ingest_document(self, content: str, metadata: dict | None = None):
        embeddings = self.embedder.embed_texts([content])
        self.vector_store.add_documents([content], embeddings, metadata=metadata)

    def answer_question(
        self, question: str, language: str = "en", level: str = "medium"
    ) -> str:
        query_embedding = self.embedder.embed_texts([question])[0]
        hits = self.vector_store.similarity_search(query_embedding, top_k=5)
        if not hits:
            return "I could not find relevant educational material to answer that question."

        combined_context = "\n".join(item["text"] for item in hits)
        return f"[RAG answer based on retrieved educational sources] {combined_context[:800]}"
