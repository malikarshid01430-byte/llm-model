from __future__ import annotations

from typing import Iterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from inference.engine import ConversationEngine, InferenceEngine

router = APIRouter(prefix="/inference", tags=["inference"])

# Global inference engine instance
_engine: InferenceEngine | None = None
_conversation_engine: ConversationEngine | None = None


def get_engine() -> InferenceEngine:
    """Get or create inference engine instance."""
    global _engine
    if _engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    return _engine


def get_conversation_engine() -> ConversationEngine:
    """Get or create conversation engine instance."""
    global _conversation_engine
    if _conversation_engine is None:
        raise HTTPException(status_code=500, detail="Conversation engine not initialized")
    return _conversation_engine


def initialize_engine(model, tokenizer, config=None) -> None:
    """Initialize inference engines."""
    global _engine, _conversation_engine
    _engine = InferenceEngine(model, tokenizer, config)
    _conversation_engine = ConversationEngine(model, tokenizer, config)


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: int | None = Field(None, description="Top-k filtering", ge=1)
    top_p: float | None = Field(None, description="Top-p filtering", ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, description="Repetition penalty", ge=1.0)
    stop_tokens: list[int] = Field(default_factory=list, description="Stop token IDs")
    max_context_len: int | None = Field(None, description="Max context length")
    do_sample: bool = Field(True, description="Whether to use sampling")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")


class StreamRequest(BaseModel):
    """Request model for streaming generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: int | None = Field(None, description="Top-k filtering", ge=1)
    top_p: float | None = Field(None, description="Top-p filtering", ge=0.0, le=1.0)
    stop_tokens: list[int] = Field(default_factory=list, description="Stop token IDs")
    max_context_len: int | None = Field(None, description="Max context length")


class BeamSearchRequest(BaseModel):
    """Request model for beam search."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(100, description="Maximum tokens to generate")
    beam_width: int = Field(4, description="Number of beams", ge=1, le=10)
    stop_tokens: list[int] = Field(default_factory=list, description="Stop token IDs")
    max_context_len: int | None = Field(None, description="Max context length")


class BeamSearchResponse(BaseModel):
    """Response model for beam search."""
    results: list[tuple[float, str]] = Field(..., description="List of (score, text) tuples")


class ChatRequest(BaseModel):
    """Request model for chat."""
    message: str = Field(..., description="User message")
    max_new_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: int | None = Field(None, description="Top-k filtering")
    top_p: float | None = Field(None, description="Top-p filtering")
    stop_tokens: list[int] = Field(default_factory=list, description="Stop token IDs")


class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str = Field(..., description="Assistant's response")
    history_length: int = Field(..., description="Number of turns in history")


class StreamChatRequest(BaseModel):
    """Request model for streaming chat."""
    message: str = Field(..., description="User message")
    max_new_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0, le=2.0)
    top_k: int | None = Field(None, description="Top-k filtering")
    top_p: float | None = Field(None, description="Top-p filtering")
    stop_tokens: list[int] = Field(default_factory=list, description="Stop token IDs")


class MemoryClearResponse(BaseModel):
    """Response model for clearing memory."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")


class SystemPromptRequest(BaseModel):
    """Request model for setting system prompt."""
    system_prompt: str = Field(..., description="System prompt text")


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text from a prompt.
    
    Supports:
    - Greedy search (temperature=0)
    - Temperature sampling
    - Top-K sampling
    - Top-P (nucleus) sampling
    - Repetition penalty
    - Stop tokens
    - Max context window
    """
    try:
        engine = get_engine()
        
        generated_text = engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            stop_tokens=request.stop_tokens,
            max_context_len=request.max_context_len,
            do_sample=request.do_sample,
        )
        
        tokens_generated = len(engine.tokenizer.encode(generated_text))
        
        return GenerateResponse(text=generated_text, tokens_generated=tokens_generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def stream_generate(request: StreamRequest) -> Iterator[str]:
    """
    Stream generated text token by token.
    
    Returns a streaming response with generated text chunks.
    """
    try:
        engine = get_engine()
        
        def generate_stream():
            for chunk in engine.stream_generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop_tokens=request.stop_tokens,
                max_context_len=request.max_context_len,
            ):
                yield chunk
        
        return generate_stream()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/beam-search", response_model=BeamSearchResponse)
async def beam_search(request: BeamSearchRequest) -> BeamSearchResponse:
    """
    Generate text using beam search.
    
    Returns multiple candidates sorted by score.
    """
    try:
        engine = get_engine()
        
        results = engine.beam_search(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            beam_width=request.beam_width,
            stop_tokens=request.stop_tokens,
            max_context_len=request.max_context_len,
        )
        
        return BeamSearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with conversation memory.
    
    Maintains conversation history and generates contextual responses.
    """
    try:
        engine = get_conversation_engine()
        
        response = engine.chat(
            user_message=request.message,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            stop_tokens=request.stop_tokens,
        )
        
        return ChatResponse(
            response=response,
            history_length=len(engine.memory),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def stream_chat(request: StreamChatRequest) -> Iterator[str]:
    """
    Stream chat response token by token.
    """
    try:
        engine = get_conversation_engine()
        
        def generate_stream():
            for chunk in engine.stream_chat(
                user_message=request.message,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop_tokens=request.stop_tokens,
            ):
                yield chunk
        
        return generate_stream()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/clear", response_model=MemoryClearResponse)
async def clear_memory() -> MemoryClearResponse:
    """
    Clear conversation history.
    """
    try:
        engine = get_conversation_engine()
        engine.clear_history()
        
        return MemoryClearResponse(
            status="success",
            message="Conversation history cleared",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system-prompt")
async def set_system_prompt(request: SystemPromptRequest) -> dict[str, str]:
    """
    Set system prompt for conversation.
    """
    try:
        engine = get_conversation_engine()
        engine.set_system_prompt(request.system_prompt)
        
        return {"status": "success", "message": "System prompt set"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    """
    Check if inference engine is ready.
    """
    global _engine
    status = "ready" if _engine is not None else "not_initialized"
    
    return {
        "status": status,
        "message": "Inference engine is ready" if _engine else "Engine not initialized",
    }