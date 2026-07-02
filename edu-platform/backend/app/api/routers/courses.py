from app.api.deps import get_current_user
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/")
async def list_courses():
    return [
        {"id": "course-1", "title": "AI for Education", "level": "beginner"},
        {"id": "course-2", "title": "Adaptive Learning", "level": "intermediate"},
    ]


@router.post("/")
async def create_course(payload: dict, current_user=Depends(get_current_user)):
    return {"status": "created", "course": payload, "created_by": current_user.email}
