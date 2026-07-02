import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "edu-platform" / "backend")
)

from app.main import app


def test_request_logging_middleware_does_not_break_health_response() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
