from __future__ import annotations

from typing import Any, Dict, List

from app.db.repositories.memory_repository import MemoryRepository
from app.db.session import AsyncSessionLocal


class MemoryService:
    async def store(
        self, user_id: str, content: str, memory_type: str = "episodic"
    ) -> Dict[str, Any]:
        async with AsyncSessionLocal() as session:
            repository = MemoryRepository(session)
            record = await repository.create(user_id, content, memory_type)
            return {
                "id": record.id,
                "user_id": record.user_id,
                "content": record.content,
                "memory_type": record.memory_type,
                "rank": record.rank,
            }

    async def list_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            repository = MemoryRepository(session)
            records = await repository.list_for_user(user_id)
            return [
                {
                    "id": record.id,
                    "user_id": record.user_id,
                    "content": record.content,
                    "memory_type": record.memory_type,
                    "rank": record.rank,
                }
                for record in records
            ]
