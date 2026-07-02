from app.api.deps import get_current_user
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/plans")
async def payment_plans(current_user=Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "plans": [
            {"name": "Institution Pro", "price": 149, "billing": "monthly"},
            {"name": "Research Suite", "price": 499, "billing": "monthly"},
        ],
    }
