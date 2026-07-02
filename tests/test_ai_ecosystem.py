import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "edu-platform" / "backend")
)

from app.main import app


def test_memory_store_and_retrieval() -> None:
    with TestClient(app) as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "email": "memory@example.com",
                "full_name": "Memory User",
                "password": "secret123",
                "role": "student",
            },
        )
        assert register_response.status_code in {201, 409}

        login_response = client.post(
            "/api/auth/login",
            data={"username": "memory@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        store_response = client.post(
            "/api/memory",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "I want to study linear algebra",
                "memory_type": "episodic",
            },
        )
        assert store_response.status_code == 200
        body = store_response.json()
        assert body["content"] == "I want to study linear algebra"

        list_response = client.get(
            "/api/memory",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_response.status_code == 200
        memories = list_response.json()
        assert any(
            memory["content"] == "I want to study linear algebra" for memory in memories
        )


def test_agent_router_returns_plan() -> None:
    with TestClient(app) as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "email": "agent@example.com",
                "full_name": "Agent User",
                "password": "secret123",
                "role": "student",
            },
        )
        assert register_response.status_code in {201, 409}

        login_response = client.post(
            "/api/auth/login",
            data={"username": "agent@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.post(
            "/api/agents/run",
            headers={"Authorization": f"Bearer {token}"},
            json={"agent": "tutor", "task": "Create a study plan for calculus"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["agent"] == "tutor"
        assert len(payload["steps"]) >= 2


def test_mcp_tool_registry_executes_tool() -> None:
    with TestClient(app) as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "email": "mcp@example.com",
                "full_name": "MCP User",
                "password": "secret123",
                "role": "student",
            },
        )
        assert register_response.status_code in {201, 409}

        login_response = client.post(
            "/api/auth/login",
            data={"username": "mcp@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.post(
            "/api/mcp/execute",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "tool_name": "plan_goal",
                "arguments": {"goal": "finish research review"},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "research review" in payload["result"].lower()
