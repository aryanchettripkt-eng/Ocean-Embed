"""Tests for backend health check and manifest loading."""

from fastapi.testclient import TestClient

from app.data.loader import DataLoader
from app.main import app


def test_health_check_success() -> None:
    """Verify /health returns 200 OK and active run_id when data is loaded."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data_loaded"] is True
    assert data["run_id"] == "synthetic_v1"


def test_health_check_failure_when_data_missing(tmp_path: str) -> None:
    """Verify DataLoader reports failure when manifest is missing."""
    empty_loader = DataLoader(data_dir=tmp_path)
    assert empty_loader.is_loaded is False
    assert empty_loader.active_run_id is None
    assert "Manifest not found" in (empty_loader.load_error or "")
