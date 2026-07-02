from app.api.deps import get_current_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    message: str


@router.post("/message")
async def send_message(payload: ChatMessage, current_user=Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "reply": f"AI assistant received: {payload.message}",
        "streaming": True,
    }
