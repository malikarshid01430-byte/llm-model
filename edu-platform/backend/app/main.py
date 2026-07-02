from contextlib import asynccontextmanager

from app.api.routers import (admin, agents, ai, analytics, auth, chat, courses,
                             files, health, mcp, memory, metrics, monitoring,
                             notifications, payments, security, student,
                             teacher, users, vision, voice)
from app.core.config import settings, validate_settings
from app.db.session import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from middleware.request_logging import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_settings()
    await init_db()
    yield


app = FastAPI(
    title="EduAI Platform API",
    description="Backend API for the AI-powered education platform.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(courses.router, prefix="/api/courses", tags=["courses"])
app.include_router(ai.router, prefix="/api/ai", tags=["ai"])
app.include_router(student.router, prefix="/api/student", tags=["student"])
app.include_router(teacher.router, prefix="/api/teacher", tags=["teacher"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(
    notifications.router, prefix="/api/notifications", tags=["notifications"]
)
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(payments.router, prefix="/api/payments", tags=["payments"])
app.include_router(voice.router, prefix="/api/voice", tags=["voice"])
app.include_router(vision.router, prefix="/api/vision", tags=["vision"])
app.include_router(security.router, prefix="/api/security", tags=["security"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(mcp.router, prefix="/api/mcp", tags=["mcp"])
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])


@app.get("/")
def root():
    return {"message": "EduAI Platform Backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "eduai-backend"}
