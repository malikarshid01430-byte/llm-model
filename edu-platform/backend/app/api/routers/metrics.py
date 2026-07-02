from app.api.rbac import require_roles
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("")
async def metrics(
    current_user=Depends(require_roles("admin", "super_admin")),
):
    return {
        "status": "ok",
        "uptime_seconds": 0,
        "requests_total": 0,
        "active_users": 0,
    }
