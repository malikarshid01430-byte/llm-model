from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import settings
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ROLE_PERMISSIONS = {
    "super_admin": ["user:read", "user:write", "course:write", "admin:write"],
    "admin": ["user:read", "user:write", "course:write"],
    "teacher": ["course:write", "student:read"],
    "student": ["content:read", "conversation:write"],
    "parent": ["student:read"],
}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": int(expire.timestamp())}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None
