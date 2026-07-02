from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag.rag_system import RAGSystem, RAGConfig

router = APIRouter(prefix="/rag", tags=["rag"])

# Global RAG system instance
_rag_system: RAGSystem | None = None


def get_rag_system() -> RAGSystem:
    """Get or create RAG system instance."""
    global _rag_system
    if _rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    return _rag_system


def initialize_rag_system(config: RAGConfig | None = None) -> None:
    """Initialize RAG system."""
    global _rag_system
    _rag_system = RAGSystem(config=config)


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    documents: List[str] = Field(..., description="List of documents to index")


class IndexResponse(BaseModel):
    """Response model for indexing."""
    status: str
    documents_indexed: int
    chunks_created: int


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="Query text")
    top_k: int | None = Field(None, description="Number of contexts to retrieve")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    prompt: str
    contexts: List[dict[str, Any]]
    num_contexts: int


class SearchRequest(BaseModel):
    """Request model for search."""
    query: str = Field(..., description="Search query")
    top_k: int | None = Field(None, description="Number of results")


class SearchResponse(BaseModel):
    """Response model for search."""
    results: List[dict[str, Any]]
    num_results: int


class SaveRequest(BaseModel):
    """Request model for saving RAG system."""
    path: str = Field(..., description="Path to save RAG system")


class SaveResponse(BaseModel):
    """Response model for saving."""
    status: str
    path: str


class MemoryClearResponse(BaseModel):
    """Response model for clearing memory."""
    status: str
    message: str


@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest) -> IndexResponse:
    """
    Index documents for retrieval.
    
    Documents are chunked and embedded for semantic search.
    """
    try:
        rag = get_rag_system()
        rag.index_documents(request.documents)
        
        return IndexResponse(
            status="success",
            documents_indexed=len(request.documents),
            chunks_created=len(rag.vector_store.documents),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system.
    
    Retrieves relevant contexts and generates a prompt with citations.
    """
    try:
        rag = get_rag_system()
        prompt, contexts = rag.query(request.query, top_k=request.top_k)
        
        context_data = []
        for ctx in contexts:
            context_data.append({
                "text": ctx.document.text,
                "score": ctx.score,
                "rank": ctx.rank,
                "metadata": ctx.document.metadata,
            })
        
        return QueryResponse(
            prompt=prompt,
            contexts=context_data,
            num_contexts=len(contexts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search for relevant documents.
    
    Returns ranked search results without generating prompts.
    """
    try:
        rag = get_rag_system()
        results = rag.retrieve(request.query, top_k=request.top_k)
        
        result_data = []
        for res in results:
            result_data.append({
                "text": res.document.text,
                "score": res.score,
                "rank": res.rank,
                "metadata": res.document.metadata,
            })
        
        return SearchResponse(
            results=result_data,
            num_results=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save", response_model=SaveResponse)
async def save_rag_system(request: SaveRequest) -> SaveResponse:
    """
    Save RAG system to disk.
    
    Saves vector store, embeddings, and configuration.
    """
    try:
        rag = get_rag_system()
        rag.save(request.path)
        
        return SaveResponse(
            status="success",
            path=request.path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_rag_system(request: SaveRequest) -> dict[str, str]:
    """
    Load RAG system from disk.
    """
    try:
        initialize_rag_system()
        rag = get_rag_system()
        rag.load(request.path)
        
        return {"status": "success", "message": f"RAG system loaded from {request.path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/clear", response_model=MemoryClearResponse)
async def clear_memory() -> MemoryClearResponse:
    """
    Clear long-term memory.
    """
    try:
        rag = get_rag_system()
        rag.long_term_memory.clear()
        
        return MemoryClearResponse(
            status="success",
            message="Long-term memory cleared",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/retrieve")
async def retrieve_memory(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Retrieve relevant memories from long-term memory.
    """
    try:
        rag = get_rag_system()
        memories = rag.long_term_memory.retrieve(query, top_k=top_k)
        
        return {
            "memories": memories,
            "num_memories": len(memories),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    """
    Check if RAG system is ready.
    """
    global _rag_system
    status = "ready" if _rag_system is not None else "not_initialized"
    
    return {
        "status": status,
        "message": "RAG system is ready" if _rag_system else "RAG system not initialized",
    }