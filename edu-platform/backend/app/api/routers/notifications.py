from app.api.deps import get_current_user
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/")
async def notification_center(current_user=Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "items": [
            {"title": "Assignment feedback ready", "channel": "in_app"},
            {"title": "Parent conference scheduled", "channel": "email"},
        ],
    }
