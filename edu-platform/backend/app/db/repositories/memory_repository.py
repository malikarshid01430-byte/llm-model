from __future__ import annotations

from typing import List
from uuid import uuid4

from app.db.models.memory import MemoryRecord
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, user_id: str, content: str, memory_type: str = "episodic"
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid4()),
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            rank="1.0",
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def list_for_user(self, user_id: str) -> List[MemoryRecord]:
        result = await self.session.execute(
            select(MemoryRecord)
            .where(MemoryRecord.user_id == user_id)
            .order_by(MemoryRecord.created_at.desc())
        )
        return list(result.scalars().all())
