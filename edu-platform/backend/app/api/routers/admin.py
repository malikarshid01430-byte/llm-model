from app.api.rbac import require_roles
from app.services.admin_service import AdminService
from fastapi import APIRouter, Depends

router = APIRouter()
service = AdminService()


@router.get("/dashboard")
async def admin_dashboard(
    current_user=Depends(require_roles("admin", "super_admin")),
):
    return service.get_dashboard(current_user.id)
