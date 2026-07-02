from app.core.config import settings
from app.db.base import Base
from app.db.models.role import Role
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

connect_args = (
    {"check_same_thread": False} if settings.postgres_url.startswith("sqlite") else {}
)
engine = create_async_engine(
    settings.postgres_url, future=True, echo=False, connect_args=connect_args
)
AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        default_roles = [
            ("student", "Student learner"),
            ("teacher", "Course instructor"),
            ("parent", "Student guardian"),
            ("admin", "Institution administrator"),
            ("super_admin", "Platform administrator"),
            ("researcher", "Research user"),
            ("developer", "Developer user"),
            ("institution_manager", "Institution manager"),
        ]
        for name, description in default_roles:
            existing = await session.execute(select(Role).where(Role.name == name))
            if existing.scalar_one_or_none() is None:
                session.add(Role(name=name, description=description))
        await session.commit()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
