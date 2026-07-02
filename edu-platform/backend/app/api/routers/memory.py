from app.api.deps import get_current_user
from app.services.memory_service import MemoryService
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()
service = MemoryService()


class MemoryPayload(BaseModel):
    content: str
    memory_type: str = "episodic"


@router.post("")
async def store_memory(payload: MemoryPayload, current_user=Depends(get_current_user)):
    return await service.store(current_user.id, payload.content, payload.memory_type)


@router.get("")
async def list_memory(current_user=Depends(get_current_user)):
    return await service.list_for_user(current_user.id)
