import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "edu-platform" / "backend")
)

from app.main import app


def test_ai_query_returns_role_specific_guidance() -> None:
    with TestClient(app) as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "email": "learner@example.com",
                "full_name": "Learner User",
                "password": "secret123",
                "role": "student",
            },
        )
        assert register_response.status_code in {201, 409}

        login_response = client.post(
            "/api/auth/login",
            data={"username": "learner@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.post(
            "/api/ai/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "user_id": "ignored",
                "query": "Help me study for calculus",
                "language": "en",
                "level": "beginner",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["role"] == "student"
        assert "study plan" in body["answer"].lower()
        assert (
            body["recommended_action"]
            == "Review fundamentals and ask for a practice quiz"
        )
