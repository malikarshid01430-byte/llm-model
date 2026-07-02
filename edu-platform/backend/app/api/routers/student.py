from app.api.deps import get_current_user
from app.services.student_service import StudentService
from fastapi import APIRouter, Depends

router = APIRouter()
service = StudentService()


@router.get("/dashboard")
async def student_dashboard(current_user=Depends(get_current_user)):
    return service.get_dashboard(current_user.id)


@router.get("/courses")
async def student_courses(current_user=Depends(get_current_user)):
    return service.list_courses(current_user.id)
