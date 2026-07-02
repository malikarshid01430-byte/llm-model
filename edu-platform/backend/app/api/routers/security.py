from app.api.deps import get_current_user
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/status")
async def security_status(current_user=Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "security": {
            "https_enabled": True,
            "jwt_enabled": True,
            "rate_limiting": True,
            "audit_logging": True,
        },
    }
