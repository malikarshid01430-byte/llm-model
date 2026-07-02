from app.db.base import Base
from sqlalchemy import Column, String


class Role(Base):
    __tablename__ = "roles"

    name = Column(String, primary_key=True, index=True)
    description = Column(String, nullable=True)
