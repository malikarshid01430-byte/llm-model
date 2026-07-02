from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datasets.chunker import DocumentChunker
from rag.retriever import Retriever


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    chunk_size: int = 512
    chunk_overlap: int = 32
    embedding_dim: int = 384
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_hybrid_search: bool = True
    use_reranking: bool = True
    max_context_length: int = 2048


@dataclass
class Document:
    """Document with metadata."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class SearchResult:
    """Search result with relevance score."""

    document: Document
    score: float
    rank: int


class EmbeddingGenerator:
    """Generate embeddings for text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None

    def _load_model(self) -> None:
        """Lazy load embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Install sentence-transformers for embedding support")

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self._load_model()
        assert self.model is not None
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        self._load_model()
        assert self.model is not None
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )
        return embeddings


class FAISSVectorStore:
    """FAISS-based vector store."""

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self.documents: list[Document] = []
        self.index = None

    def _init_index(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss

            self.index = faiss.IndexFlatL2(self.embedding_dim)
        except ImportError:
            raise ImportError("Install faiss-cpu or faiss-gpu for FAISS support")

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to vector store."""
        if self.index is None:
            self._init_index()

        embeddings = np.array(
            [doc.embedding for doc in documents if doc.embedding is not None]
        )
        if len(embeddings) > 0 and self.index is not None:
            self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Search for similar documents."""
        if self.index is None:
            return []

        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= 0:
                results.append((idx, float(dist)))
        return results

    def save(self, path: str | Path) -> None:
        """Save vector store to disk."""
        try:
            import faiss

            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(path / "index.faiss"))

            # Save documents
            docs_data = []
            for doc in self.documents:
                docs_data.append(
                    {
                        "id": doc.id,
                        "text": doc.text,
                        "metadata": doc.metadata,
                        "embedding": (
                            doc.embedding.tolist()
                            if doc.embedding is not None
                            else None
                        ),
                    }
                )
            with open(path / "documents.json", "w") as f:
                json.dump(docs_data, f)
        except ImportError:
            raise ImportError("Install faiss-cpu or faiss-gpu for FAISS support")

    def load(self, path: str | Path) -> None:
        """Load vector store from disk."""
        try:
            import faiss

            path = Path(path)

            # Load FAISS index
            index_path = path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))

            # Load documents
            docs_path = path / "documents.json"
            if docs_path.exists():
                with open(docs_path, "r") as f:
                    docs_data = json.load(f)
                self.documents = [
                    Document(
                        id=d["id"],
                        text=d["text"],
                        metadata=d["metadata"],
                        embedding=(
                            np.array(d["embedding"])
                            if d["embedding"] is not None
                            else None
                        ),
                    )
                    for d in docs_data
                ]
        except ImportError:
            raise ImportError("Install faiss-cpu or faiss-gpu for FAISS support")


class ChromaVectorStore:
    """ChromaDB-based vector store."""

    def __init__(
        self, collection_name: str = "documents", persist_directory: str | None = None
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

    def _init_client(self) -> None:
        """Initialize ChromaDB client."""
        if self.client is None:
            try:
                import chromadb

                if self.persist_directory:
                    self.client = chromadb.PersistentClient(path=self.persist_directory)
                else:
                    self.client = chromadb.EphemeralClient()
                self.collection = self.client.get_or_create_collection(
                    self.collection_name
                )
            except ImportError:
                raise ImportError("Install chromadb for ChromaDB support")

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to vector store."""
        self._init_client()

        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        embeddings = [
            doc.embedding.tolist() for doc in documents if doc.embedding is not None
        ]
        metadatas = [doc.metadata for doc in documents]

        if len(embeddings) > 0 and self.collection is not None:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Search for similar documents."""
        self._init_client()

        assert self.collection is not None
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        output = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, (doc_id, distance) in enumerate(
                zip(results["ids"][0], results["distances"][0])
            ):
                output.append((i, float(distance)))
        return output

    def get_documents(self, indices: list[int]) -> list[Document]:
        """Get documents by indices."""
        self._init_client()
        # ChromaDB returns documents in order
        assert self.collection is not None
        results = self.collection.get()
        documents = []
        for idx in indices:
            if idx < len(results["ids"]):
                doc = Document(
                    id=results["ids"][idx],
                    text=results["documents"][idx],
                    metadata=results["metadatas"][idx] if results["metadatas"] else {},
                )
                documents.append(doc)
        return documents


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search."""

    def __init__(self, vector_store, embedding_generator: EmbeddingGenerator) -> None:
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.keyword_retriever = Retriever()

    def index(self, documents: list[str]) -> None:
        """Index documents for both semantic and keyword search."""
        # Index for keyword search
        self.keyword_retriever.index(documents)

        # Index for semantic search
        doc_objects = []
        for i, text in enumerate(documents):
            embedding = self.embedding_generator.embed(text)
            doc = Document(
                id=str(i),
                text=text,
                embedding=embedding,
            )
            doc_objects.append(doc)

        self.vector_store.add_documents(doc_objects)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve documents using hybrid search."""
        # Semantic search
        query_embedding = self.embedding_generator.embed(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=top_k)

        # Keyword search
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k)

        # Combine and deduplicate
        seen_ids = set()
        combined = []

        for idx, score in semantic_results:
            if idx not in seen_ids:
                seen_ids.add(idx)
                combined.append((idx, score, "semantic"))

        for kw_doc in keyword_results:
            doc_id = hash(kw_doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined.append((doc_id, 0.5, "keyword"))

        # Sort by score and return top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []
        for rank, (idx, score, source) in enumerate(combined[:top_k]):
            if source == "semantic":
                result_doc = self.vector_store.documents[idx]
            else:
                result_doc = Document(id=str(idx), text=keyword_results[0])
            results.append(SearchResult(document=result_doc, score=score, rank=rank))

        return results


class ContextRanker:
    """Rank retrieved contexts by relevance."""

    def __init__(self) -> None:
        pass

    def rank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Rank results by relevance to query."""
        query_terms = set(query.lower().split())

        scored_results = []
        for result in results:
            doc_terms = set(result.document.text.lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            length_score = min(len(result.document.text) / 100, 1.0)
            combined_score = (
                (overlap * 0.3) + (result.score * 0.5) + (length_score * 0.2)
            )
            scored_results.append((combined_score, result))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        ranked = []
        for rank, (score, result) in enumerate(scored_results):
            result.rank = rank
            result.score = score
            ranked.append(result)

        return ranked


class PromptBuilder:
    """Build prompts with retrieved context."""

    def __init__(self, max_context_length: int = 2048) -> None:
        self.max_context_length = max_context_length

    def build(
        self,
        query: str,
        contexts: list[SearchResult],
        system_prompt: str = "You are a helpful assistant. Use the following context to answer the question.",
    ) -> str:
        """Build prompt with context."""
        context_parts = []
        total_length = len(system_prompt) + len(query)

        for i, result in enumerate(contexts):
            context_text = f"[{i+1}] {result.document.text}"
            if total_length + len(context_text) < self.max_context_length:
                context_parts.append(context_text)
                total_length += len(context_text)
            else:
                break

        context_str = "\n\n".join(context_parts)
        prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        return prompt

    def build_with_citations(
        self,
        query: str,
        contexts: list[SearchResult],
        system_prompt: str = "You are a helpful assistant. Use the following context to answer the question. Cite sources using [1], [2], etc.",
    ) -> str:
        """Build prompt with citation markers."""
        context_parts = []
        total_length = len(system_prompt) + len(query)

        for i, result in enumerate(contexts):
            context_text = f"[{i+1}] {result.document.text}"
            if total_length + len(context_text) < self.max_context_length:
                context_parts.append(context_text)
                total_length += len(context_text)
            else:
                break

        context_str = "\n\n".join(context_parts)
        prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer (with citations):"
        return prompt


class SourceCitation:
    """Manage source citations."""

    def __init__(self) -> None:
        self.sources: dict[int, Document] = {}

    def add_source(self, index: int, document: Document) -> None:
        """Add a source document."""
        self.sources[index] = document

    def get_citation(self, index: int) -> str:
        """Get citation text for a source."""
        if index in self.sources:
            doc = self.sources[index]
            return f"[{index}] {doc.metadata.get('source', 'Unknown')}"
        return f"[{index}] Unknown source"

    def format_citations(self, indices: list[int]) -> str:
        """Format multiple citations."""
        citations = []
        for idx in indices:
            citations.append(self.get_citation(idx))
        return "\n".join(citations)


class LongTermMemory:
    """Long-term memory for RAG system."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.memory: list[dict[str, Any]] = []

    def add(self, query: str, response: str, contexts: list[SearchResult]) -> None:
        """Add interaction to memory."""
        entry = {
            "query": query,
            "response": response,
            "contexts": [
                {"text": ctx.document.text, "score": ctx.score} for ctx in contexts
            ],
            "timestamp": hash(f"{query}{response}"),
        }
        self.memory.append(entry)

        # Trim memory if needed
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size :]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant past interactions."""
        query_terms = set(query.lower().split())
        scored = []

        for entry in self.memory:
            entry_terms = set(entry["query"].lower().split())
            overlap = len(query_terms.intersection(entry_terms))
            scored.append((overlap, entry))

        return [entry for _, entry in sorted(scored, reverse=True)[:top_k]]

    def clear(self) -> None:
        """Clear memory."""
        self.memory = []


class RAGSystem:
    """Complete RAG system."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore(embedding_dim=self.config.embedding_dim)
        self.chunker = DocumentChunker(
            self.config.chunk_size, self.config.chunk_overlap
        )
        self.ranker = ContextRanker()
        self.prompt_builder = PromptBuilder(self.config.max_context_length)
        self.citation_manager = SourceCitation()
        self.long_term_memory = LongTermMemory()
        self.hybrid_retriever = None

        if self.config.use_hybrid_search:
            self.hybrid_retriever = HybridRetriever(
                self.vector_store, self.embedding_generator
            )

    def index_documents(self, documents: list[str]) -> None:
        """Index documents for retrieval."""
        # Chunk documents
        chunks: list[str] = []
        for raw_doc in documents:
            chunks.extend(self.chunker.chunk(raw_doc))

        # Generate embeddings
        doc_objects: list[Document] = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_generator.embed(chunk)
            doc_obj = Document(
                id=str(i),
                text=chunk,
                embedding=embedding,
            )
            doc_objects.append(doc_obj)

        # Add to vector store
        self.vector_store.add_documents(doc_objects)

        # Index for hybrid search
        if self.hybrid_retriever:
            self.hybrid_retriever.index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve relevant contexts."""
        top_k = top_k or self.config.top_k

        if self.hybrid_retriever:
            results = self.hybrid_retriever.retrieve(query, top_k=top_k)
        else:
            # Semantic search only
            query_embedding = self.embedding_generator.embed(query)
            search_results = self.vector_store.search(query_embedding, top_k=top_k)

            results = []
            for rank, (idx, score) in enumerate(search_results):
                doc = self.vector_store.documents[idx]
                results.append(SearchResult(document=doc, score=score, rank=rank))

        # Rerank if enabled
        if self.config.use_reranking:
            results = self.ranker.rank(query, results)

        # Add to citation manager
        for result in results:
            self.citation_manager.add_source(result.rank + 1, result.document)

        return results

    def generate_prompt(
        self, query: str, contexts: list[SearchResult] | None = None
    ) -> str:
        """Generate prompt with context."""
        if contexts is None:
            contexts = self.retrieve(query)

        return self.prompt_builder.build_with_citations(query, contexts)

    def query(
        self, query: str, top_k: int | None = None
    ) -> tuple[str, list[SearchResult]]:
        """Query the RAG system."""
        # Retrieve contexts
        contexts = self.retrieve(query, top_k=top_k)

        # Generate prompt
        prompt = self.generate_prompt(query, contexts)

        # Add to long-term memory
        self.long_term_memory.add(query, prompt, contexts)

        return prompt, contexts

    def save(self, path: str | Path) -> None:
        """Save RAG system to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(path / "vector_store")

        # Save config
        import dataclasses

        config_dict = dataclasses.asdict(self.config)
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f)

    def load(self, path: str | Path) -> None:
        """Load RAG system from disk."""
        path = Path(path)
        self.vector_store.load(path / "vector_store")
