from app.api.deps import get_current_user
from app.services.mcp_service import MCPService
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

router = APIRouter()
service = MCPService()


class MCPRequest(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)


@router.post("/execute")
async def execute_tool(
    payload: MCPRequest,
    current_user=Depends(get_current_user),
):
    return service.execute_tool(payload.tool_name, payload.arguments)
