import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "edu-platform" / "backend")
)

from app.main import app


def test_readiness_endpoint_is_available() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
