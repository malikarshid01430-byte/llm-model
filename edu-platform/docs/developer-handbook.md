# Developer Handbook

## Setup
1. Create and activate a Python environment.
2. Install backend dependencies with `pip install -r edu-platform/backend/requirements.txt`.
3. Install frontend dependencies with `npm install` in `edu-platform/frontend`.
4. Run the backend with `uvicorn app.main:app --reload` inside `edu-platform/backend`.
5. Run the frontend with `npm run dev` inside `edu-platform/frontend`.

## Development Workflow
- Keep the backend modular by adding services and routers rather than embedding logic in handlers.
- Preserve the test suite by adding regression tests whenever new endpoints are introduced.
- Validate changes with `pytest -q tests` before shipping.
