from app.api.deps import get_current_user
from app.services.teacher_service import TeacherService
from fastapi import APIRouter, Depends

router = APIRouter()
service = TeacherService()


@router.get("/dashboard")
async def teacher_dashboard(current_user=Depends(get_current_user)):
    return service.get_dashboard(current_user.id)


@router.get("/assignments")
async def teacher_assignments(current_user=Depends(get_current_user)):
    return service.list_assignments(current_user.id)
