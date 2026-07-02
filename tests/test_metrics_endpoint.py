import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "edu-platform" / "backend")
)

from app.main import app


def _admin_token(client: TestClient) -> str:
    register_response = client.post(
        "/api/auth/register",
        json={
            "email": "metrics-admin@example.com",
            "full_name": "Metrics Admin",
            "password": "secret123",
            "role": "student",
        },
    )
    assert register_response.status_code in {201, 409}

    login_response = client.post(
        "/api/auth/login",
        data={"username": "metrics-admin@example.com", "password": "secret123"},
    )
    assert login_response.status_code == 200
    return login_response.json()["access_token"]


def test_metrics_endpoint_requires_admin_role() -> None:
    with TestClient(app) as client:
        token = _admin_token(client)
        response = client.get(
            "/api/metrics",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


def test_metrics_endpoint_is_available_for_admin() -> None:
    with TestClient(app) as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "email": "platform-admin@example.com",
                "full_name": "Platform Admin",
                "password": "secret123",
                "role": "admin",
            },
        )
        assert register_response.status_code in {201, 409}

        login_response = client.post(
            "/api/auth/login",
            data={"username": "platform-admin@example.com", "password": "secret123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/metrics",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
