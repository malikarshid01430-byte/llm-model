from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag.rag_system import (
    ChromaVectorStore,
    ContextRanker,
    Document,
    EmbeddingGenerator,
    FAISSVectorStore,
    HybridRetriever,
    LongTermMemory,
    PromptBuilder,
    RAGConfig,
    RAGSystem,
    SearchResult,
    SourceCitation,
)


class TestEmbeddingGenerator:
    """Test embedding generator."""

    def test_embed_single_text(self) -> None:
        """Test embedding single text."""
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

        # This will fail if sentence-transformers not installed
        try:
            embedding = generator.embed("Hello world")
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape[0] > 0
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_embed_batch(self) -> None:
        """Test embedding batch of texts."""
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

        try:
            embeddings = generator.embed_batch(["Hello world", "Test text"])
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == 2
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestFAISSVectorStore:
    """Test FAISS vector store."""

    def test_add_documents(self) -> None:
        """Test adding documents."""
        try:
            store = FAISSVectorStore(embedding_dim=384)

            docs = [
                Document(id="1", text="Hello world", embedding=np.random.rand(384)),
                Document(id="2", text="Test text", embedding=np.random.rand(384)),
            ]
            store.add_documents(docs)

            assert len(store.documents) == 2
        except ImportError:
            pytest.skip("faiss not installed")

    def test_search(self) -> None:
        """Test searching documents."""
        try:
            store = FAISSVectorStore(embedding_dim=384)

            docs = [
                Document(id="1", text="Hello world", embedding=np.random.rand(384)),
                Document(id="2", text="Test text", embedding=np.random.rand(384)),
            ]
            store.add_documents(docs)

            query_embedding = np.random.rand(384)
            results = store.search(query_embedding, top_k=2)

            assert len(results) <= 2
            assert all(isinstance(r, tuple) for r in results)
        except ImportError:
            pytest.skip("faiss not installed")

    def test_save_load(self, tmp_path: Path) -> None:
        """Test saving and loading."""
        try:
            store = FAISSVectorStore(embedding_dim=384)

            docs = [
                Document(id="1", text="Hello world", embedding=np.random.rand(384)),
            ]
            store.add_documents(docs)

            # Save
            store.save(tmp_path / "faiss_store")

            # Load
            store2 = FAISSVectorStore(embedding_dim=384)
            store2.load(tmp_path / "faiss_store")

            assert len(store2.documents) == 1
            assert store2.documents[0].text == "Hello world"
        except ImportError:
            pytest.skip("faiss not installed")


class TestChromaVectorStore:
    """Test ChromaDB vector store."""

    def test_add_documents(self, tmp_path: Path) -> None:
        """Test adding documents."""
        try:
            store = ChromaVectorStore(
                collection_name="test",
                persist_directory=str(tmp_path / "chroma"),
            )

            docs = [
                Document(id="1", text="Hello world", embedding=np.random.rand(384)),
                Document(id="2", text="Test text", embedding=np.random.rand(384)),
            ]
            store.add_documents(docs)

            assert len(store.documents) == 0  # Chroma doesn't store in memory
        except ImportError:
            pytest.skip("chromadb not installed")

    def test_search(self, tmp_path: Path) -> None:
        """Test searching documents."""
        try:
            store = ChromaVectorStore(
                collection_name="test_search",
                persist_directory=str(tmp_path / "chroma_search"),
            )

            docs = [
                Document(id="1", text="Hello world", embedding=np.random.rand(384)),
                Document(id="2", text="Test text", embedding=np.random.rand(384)),
            ]
            store.add_documents(docs)

            query_embedding = np.random.rand(384)
            results = store.search(query_embedding, top_k=2)

            assert len(results) <= 2
        except ImportError:
            pytest.skip("chromadb not installed")


class TestHybridRetriever:
    """Test hybrid retriever."""

    def test_index_and_retrieve(self) -> None:
        """Test indexing and retrieval."""
        try:
            embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
            vector_store = FAISSVectorStore(embedding_dim=384)
            retriever = HybridRetriever(vector_store, embedding_gen)

            documents = [
                "Hello world this is a test",
                "Another document with different content",
                "Testing the retriever functionality",
            ]

            retriever.index(documents)
            results = retriever.retrieve("test document", top_k=2)

            assert len(results) <= 2
            assert all(isinstance(r, SearchResult) for r in results)
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestContextRanker:
    """Test context ranker."""

    def test_rank_results(self) -> None:
        """Test ranking search results."""
        ranker = ContextRanker()

        results = [
            SearchResult(
                document=Document(id="1", text="Hello world test"),
                score=0.8,
                rank=0,
            ),
            SearchResult(
                document=Document(id="2", text="Completely different content here"),
                score=0.6,
                rank=1,
            ),
        ]

        ranked = ranker.rank("test document", results)

        assert len(ranked) == 2
        assert all(isinstance(r, SearchResult) for r in ranked)
        assert ranked[0].rank == 0


class TestPromptBuilder:
    """Test prompt builder."""

    def test_build_prompt(self) -> None:
        """Test building prompt with context."""
        builder = PromptBuilder(max_context_length=2048)

        contexts = [
            SearchResult(
                document=Document(id="1", text="Context one"),
                score=0.9,
                rank=0,
            ),
            SearchResult(
                document=Document(id="2", text="Context two"),
                score=0.8,
                rank=1,
            ),
        ]

        prompt = builder.build("What is the answer?", contexts)

        assert "What is the answer?" in prompt
        assert "Context one" in prompt
        assert "Context two" in prompt

    def test_build_with_citations(self) -> None:
        """Test building prompt with citations."""
        builder = PromptBuilder(max_context_length=2048)

        contexts = [
            SearchResult(
                document=Document(id="1", text="Context one"),
                score=0.9,
                rank=0,
            ),
        ]

        prompt = builder.build_with_citations("What is the answer?", contexts)

        assert "[1]" in prompt
        assert "Context one" in prompt


class TestSourceCitation:
    """Test source citation."""

    def test_add_and_get_citation(self) -> None:
        """Test adding and getting citations."""
        citation = SourceCitation()

        doc = Document(id="1", text="Test", metadata={"source": "test.pdf"})
        citation.add_source(1, doc)

        result = citation.get_citation(1)
        assert "test.pdf" in result

    def test_format_citations(self) -> None:
        """Test formatting multiple citations."""
        citation = SourceCitation()

        doc1 = Document(id="1", text="Test 1", metadata={"source": "doc1.pdf"})
        doc2 = Document(id="2", text="Test 2", metadata={"source": "doc2.pdf"})

        citation.add_source(1, doc1)
        citation.add_source(2, doc2)

        result = citation.format_citations([1, 2])
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result


class TestLongTermMemory:
    """Test long-term memory."""

    def test_add_and_retrieve(self) -> None:
        """Test adding and retrieving memories."""
        memory = LongTermMemory(max_size=100)

        contexts = [
            SearchResult(
                document=Document(id="1", text="Test context"),
                score=0.9,
                rank=0,
            ),
        ]

        memory.add("test query", "test response", contexts)
        results = memory.retrieve("test query", top_k=1)

        assert len(results) == 1
        assert results[0]["query"] == "test query"

    def test_clear_memory(self) -> None:
        """Test clearing memory."""
        memory = LongTermMemory()

        contexts = [
            SearchResult(
                document=Document(id="1", text="Test"),
                score=0.9,
                rank=0,
            ),
        ]

        memory.add("query", "response", contexts)
        assert len(memory.memory) == 1

        memory.clear()
        assert len(memory.memory) == 0


class TestRAGSystem:
    """Test complete RAG system."""

    def test_index_documents(self) -> None:
        """Test indexing documents."""
        try:
            rag = RAGSystem(config=RAGConfig(chunk_size=50, chunk_overlap=10))

            documents = [
                "Hello world this is a test document",
                "Another document with different content",
            ]

            rag.index_documents(documents)

            assert len(rag.vector_store.documents) > 0
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_retrieve(self) -> None:
        """Test retrieval."""
        try:
            rag = RAGSystem(config=RAGConfig(chunk_size=50, chunk_overlap=10))

            documents = [
                "Hello world this is a test document",
                "Another document with different content",
            ]

            rag.index_documents(documents)
            results = rag.retrieve("test document", top_k=2)

            assert len(results) <= 2
            assert all(isinstance(r, SearchResult) for r in results)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_generate_prompt(self) -> None:
        """Test prompt generation."""
        try:
            rag = RAGSystem(config=RAGConfig(chunk_size=50, chunk_overlap=10))

            documents = [
                "Hello world this is a test document",
            ]

            rag.index_documents(documents)
            prompt = rag.generate_prompt("What is this?")

            assert "What is this?" in prompt
            assert "Hello world" in prompt
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_query(self) -> None:
        """Test full query pipeline."""
        try:
            rag = RAGSystem(config=RAGConfig(chunk_size=50, chunk_overlap=10))

            documents = [
                "Hello world this is a test document",
                "Another document with different content",
            ]

            rag.index_documents(documents)
            prompt, contexts = rag.query("test document", top_k=2)

            assert len(prompt) > 0
            assert len(contexts) <= 2
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_save_load(self, tmp_path: Path) -> None:
        """Test saving and loading."""
        try:
            rag = RAGSystem(config=RAGConfig(chunk_size=50, chunk_overlap=10))

            documents = [
                "Hello world this is a test document",
            ]

            rag.index_documents(documents)
            rag.save(tmp_path / "rag_system")

            # Load
            rag2 = RAGSystem(config=RAGConfig(chunk_size=50, chunk_overlap=10))
            rag2.load(tmp_path / "rag_system")

            assert len(rag2.vector_store.documents) > 0
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestIntegration:
    """Test RAG integration."""

    def test_full_rag_pipeline(self) -> None:
        """Test complete RAG pipeline."""
        try:
            # Create RAG system
            rag = RAGSystem(
                config=RAGConfig(
                    chunk_size=100,
                    chunk_overlap=20,
                    top_k=3,
                )
            )

            # Index documents
            documents = [
                "Python is a programming language. It is widely used for web development.",
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with many layers.",
                "Natural language processing helps computers understand human language.",
            ]

            rag.index_documents(documents)

            # Query
            prompt, contexts = rag.query("What is Python?", top_k=2)

            assert len(prompt) > 0
            assert len(contexts) > 0
            assert any("Python" in ctx.document.text for ctx in contexts)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_hybrid_search(self) -> None:
        """Test hybrid search functionality."""
        try:
            rag = RAGSystem(
                config=RAGConfig(
                    chunk_size=100,
                    chunk_overlap=20,
                    use_hybrid_search=True,
                )
            )

            documents = [
                "Python programming language tutorial",
                "JavaScript for web development",
                "Machine learning with Python",
            ]

            rag.index_documents(documents)
            results = rag.retrieve("Python", top_k=2)

            assert len(results) <= 2
        except ImportError:
            pytest.skip("sentence-transformers not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
