from app.api.deps import get_current_user
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/overview")
async def analytics_overview(current_user=Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "summary": {
            "students_active": 124,
            "courses_running": 8,
            "completion_rate": 89,
            "ai_interactions": 5421,
        },
    }
