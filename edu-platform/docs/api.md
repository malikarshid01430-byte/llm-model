# API Documentation

## Auth
- POST `/api/auth/register` - Register a new user.
- POST `/api/auth/login` - Authenticate and receive JWT.

## Users
- GET `/api/users/me` - Retrieve current user profile.

## Courses
- GET `/api/courses` - List available courses.
- POST `/api/courses` - Create a new course.

## Enterprise
- GET `/api/student/dashboard` - Retrieve a student dashboard snapshot.
- GET `/api/student/courses` - List student course progress.
- GET `/api/teacher/dashboard` - Retrieve a teacher dashboard snapshot.
- GET `/api/teacher/assignments` - List teacher assignments.

## AI Ecosystem
- POST `/api/ai/query` - Submit a request to the AI tutor.
- POST `/api/memory` - Store a memory for the authenticated user.
- GET `/api/memory` - List memories for the authenticated user.
- POST `/api/agents/run` - Run a planning or tutoring agent.
- POST `/api/mcp/execute` - Execute a registered MCP tool.

## Health
- GET `/health` - Basic health check.
