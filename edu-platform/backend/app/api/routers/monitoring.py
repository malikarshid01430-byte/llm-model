from app.api.deps import get_current_user
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/metrics")
async def monitoring_metrics(current_user=Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "metrics": {
            "cpu_percent": 24,
            "memory_percent": 63,
            "gpu_percent": 0,
            "api_latency_ms": 42,
        },
    }
