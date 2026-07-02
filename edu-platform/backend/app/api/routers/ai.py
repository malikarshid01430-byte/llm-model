from typing import List

from app.api.deps import get_current_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()


class AIQuery(BaseModel):
    user_id: str
    query: str
    language: str = "en"
    level: str = "medium"
    sources: List[str] = ["institutional"]


@router.post("/query")
async def query_ai(payload: AIQuery, current_user=Depends(get_current_user)):
    role = getattr(current_user, "role", "student")
    query_text = payload.query.lower()

    if role == "teacher":
        answer = (
            "Here is a teacher-focused lesson plan: break the topic into milestones, "
            "prepare one practice activity, and follow up with a short formative quiz."
        )
        recommended_action = "Create a scaffolded lesson and assign a reflection prompt"
    else:
        answer = (
            "Here is a student-friendly study plan: start with the fundamentals, "
            "review the core concept for 20 minutes, and ask for a guided practice quiz."
        )
        recommended_action = "Review fundamentals and ask for a practice quiz"

    if "calculus" in query_text or "math" in query_text:
        answer = (
            "Here is a focused study plan for math: review the prerequisite concepts, "
            "work through two example problems, and then practice a short quiz."
        )
        recommended_action = "Review fundamentals and ask for a practice quiz"

    response = {
        "user_id": current_user.id,
        "role": role,
        "query": payload.query,
        "language": payload.language,
        "level": payload.level,
        "answer": answer,
        "recommended_action": recommended_action,
        "source": "RAG knowledge base",
    }
    return response
