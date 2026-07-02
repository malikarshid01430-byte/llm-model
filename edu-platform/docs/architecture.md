# EduAI Platform Architecture

## Overview
The platform is designed as a modular AI operating system for education with separated frontend, backend, AI services, memory, agent orchestration, and deployment layers.

## Components
- Frontend: Next.js + Tailwind CSS application with role-based dashboards and auth flows.
- Backend: FastAPI service handling authentication, user management, course management, AI orchestration, memory, agents, and MCP tool routing.
- AI Service: Python-based RAG pipeline with vector storage, multimodal utilities, and custom reasoning modules.
- Databases: SQLite for local development, with a migration-ready path to PostgreSQL and other enterprise stores.
- Deployment: Docker Compose for local orchestration and Nginx reverse proxy for routing.

## Data Flow
1. User interacts with the Next.js frontend.
2. Frontend calls backend REST APIs for app logic and AI assistance.
3. Backend coordinates with the AI service, memory, agent runtime, and tool registry.
4. Results are returned to the client and logged for observability and future adaptation.
