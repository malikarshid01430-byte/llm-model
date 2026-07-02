from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag_pipeline import RAGPipeline

app = FastAPI(title="EduAI AI Service", version="0.1.0")

pipeline = RAGPipeline()


class DocumentUpload(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    user_id: str
    question: str
    language: str = "en"
    level: str = "medium"


@app.post("/upload")
def upload_document(payload: DocumentUpload):
    pipeline.ingest_document(payload.content, metadata=payload.metadata)
    return {"status": "uploaded"}


@app.post("/query")
def query(payload: QueryRequest):
    answer = pipeline.answer_question(
        payload.question, language=payload.language, level=payload.level
    )
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found")
    return {"answer": answer}
