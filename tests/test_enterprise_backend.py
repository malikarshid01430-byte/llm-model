import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "edu-platform" / "backend")
)

from app.main import app


def test_enterprise_routes_are_registered() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        register_response = client.post(
            "/api/auth/register",
            json={
                "email": "student@example.com",
                "full_name": "Student User",
                "password": "secret123",
                "role": "student",
            },
        )
        assert register_response.status_code in {201, 409}

        login_response = client.post(
            "/api/auth/login",
            data={"username": "student@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        dashboard_response = client.get(
            "/api/student/dashboard",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert dashboard_response.status_code == 200
