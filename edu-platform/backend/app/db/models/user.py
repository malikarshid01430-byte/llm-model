from datetime import datetime, timezone

from app.db.base import Base
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, ForeignKey("roles.name"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utc_now)

    role_info = relationship("Role", backref="users")
