"""Tests for the API health endpoint and app setup."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_title() -> None:
    assert app.title == "EduLLM API"


def test_openapi_schema() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/health" in schema["paths"]
