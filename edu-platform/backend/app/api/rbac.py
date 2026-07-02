from typing import Callable

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import ROLE_PERMISSIONS
from app.db.models.user import User
from fastapi import Depends, HTTPException, status


def require_roles(*roles: str) -> Callable:
    allowed = set(roles)

    async def checker(current_user: User = Depends(get_current_user)) -> User:
        role_name = str(getattr(current_user, "role", ""))
        if role_name not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this resource",
            )
        return current_user

    return checker


def require_permissions(*permissions: str) -> Callable:
    required = set(permissions)

    async def checker(current_user: User = Depends(get_current_user)) -> User:
        role_name = str(getattr(current_user, "role", ""))
        granted = set(ROLE_PERMISSIONS.get(role_name, []))
        if not required.issubset(granted):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this resource",
            )
        return current_user

    return checker


ALLOWED_REGISTRATION_ROLES = {"student", "teacher", "parent"}
if settings.debug:
    ALLOWED_REGISTRATION_ROLES = ALLOWED_REGISTRATION_ROLES | {
        "admin",
        "super_admin",
    }
