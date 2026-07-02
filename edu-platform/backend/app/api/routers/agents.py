from app.api.deps import get_current_user
from app.services.agent_service import AgentService
from fastapi import APIRouter, Depends
from pydantic import BaseModel

router = APIRouter()
service = AgentService()


class AgentRequest(BaseModel):
    agent: str
    task: str


@router.post("/run")
async def run_agent(
    payload: AgentRequest,
    current_user=Depends(get_current_user),
):
    return service.run(payload.agent, payload.task)
