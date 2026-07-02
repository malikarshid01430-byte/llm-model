from uuid import uuid4

from app.api.rbac import ALLOWED_REGISTRATION_ROLES
from app.core.security import (create_access_token, hash_password,
                               verify_password)
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.token import Token
from app.schemas.user import UserCreate, UserRead
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        body = await request.json()
        username = body.get("username") or body.get("email")
        password = body.get("password")
    else:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credentials"
        )

    query = await db.execute(select(User).where(User.email == username))
    user = query.scalar_one_or_none()
    if not user or not verify_password(str(password), str(user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=str(user.id))
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(user_create: UserCreate, db: AsyncSession = Depends(get_db)):
    if user_create.role not in ALLOWED_REGISTRATION_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role must be one of: {', '.join(sorted(ALLOWED_REGISTRATION_ROLES))}",
        )

    existing = await db.execute(select(User).where(User.email == user_create.email))
    existing_user = existing.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    user = User(
        id=str(uuid4()),
        email=user_create.email,
        full_name=user_create.full_name,
        hashed_password=hash_password(user_create.password),
        role=user_create.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
