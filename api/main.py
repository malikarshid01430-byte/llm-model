from fastapi import FastAPI

from api.routers import generation_router, health_router

app = FastAPI(title="EduLLM API", version="1.0.0")
app.include_router(health_router)
app.include_router(generation_router)
