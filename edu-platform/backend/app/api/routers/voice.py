from app.api.deps import get_current_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()


class VoiceRequest(BaseModel):
    text: str


@router.post("/synthesize")
async def synthesize_voice(
    payload: VoiceRequest, current_user=Depends(get_current_user)
):
    return {
        "user_id": current_user.id,
        "text": payload.text,
        "audio_url": "https://example.invalid/audio/voice.mp3",
    }
