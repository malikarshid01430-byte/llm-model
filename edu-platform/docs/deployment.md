# Deployment Guide

## Local Development
1. Install backend dependencies with `pip install -r edu-platform/backend/requirements.txt`.
2. Start the FastAPI backend from the backend folder with `uvicorn app.main:app --reload --port 8000`.
3. Start the Next.js frontend from the frontend folder with `npm install && npm run dev`.
4. Visit `http://localhost:3000` for the frontend.

## Production
- Use the provided `docker-compose.yml` and replace environment secrets with secure values.
- Configure a managed PostgreSQL instance and map relevant connection settings in the backend configuration.
- Configure TLS in Nginx and use external load balancers.
- Add autoscaling to the frontend and AI services when demand increases.
- Expose the new memory, agent, and MCP endpoints behind the same authentication and observability layers as the rest of the API.
