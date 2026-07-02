from datetime import datetime, timezone

from app.db.base import Base
from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRecord(Base):
    __tablename__ = "memory_records"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    content = Column(String, nullable=False)
    memory_type = Column(String, nullable=False, default="episodic")
    rank = Column(String, nullable=False, default="1.0")
    created_at = Column(DateTime, default=_utc_now)

    user = relationship("User", backref="memory_records")
