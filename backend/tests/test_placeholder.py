"""Placeholder test for backend to verify health check and settings."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_check() -> None:
    """Verify /health returns 200 OK."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data_loaded"] is True
