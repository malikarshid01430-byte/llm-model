# EduAI Platform

A production-ready AI-powered education platform for students, teachers, parents, administrators, researchers, institutions, and developers.

## Features
- FastAPI backend with JWT authentication, role-based access, enterprise domain routers, memory, agents, MCP support, and readiness checks.
- Next.js frontend with Tailwind CSS, Redux Toolkit, role-based dashboards, and resilient API loading.
- AI service support for chat, voice, vision, analytics, notifications, files, payments, monitoring, and tool routing.
- Local SQLite development database with schema initialization and migration-ready structure.
- Docker Compose deployment configuration and documentation.
- Clean architecture and modular service separation.

## Quick Start
1. Install dependencies with `pip install -r edu-platform/backend/requirements.txt`.
2. From the repo root, run `python edu-platform/backend/app/main.py` or start via `uvicorn app.main:app --reload` in the backend directory.
3. In the frontend folder, run `npm install && npm run dev`.
4. Open `http://localhost:3000`.

## Project Layout
- `backend/`: FastAPI service and API implementation.
- `frontend/`: Next.js user interface.
- `ai-service/`: RAG and document ingestion pipeline.
- `infra/`: Docker Compose, Nginx, and CI/CD workflows.
- `docs/`: architecture and deployment documentation.
- `database/`: SQL schema and database design files.

## Verified Commands
- `pytest -q tests/test_enterprise_backend.py`
- `python -m compileall edu-platform/backend`
