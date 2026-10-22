"""Tests for backend health check and manifest loading."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as app_main
from app.data import loader as loader_module
from app.data.loader import DataLoader
from app.main import app


def test_health_check_success() -> None:
    """Verify /health and /api/v1/health return 200 OK and active run_id when data is loaded."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data_loaded"] is True
    assert data["run_id"] == "synthetic_v1"

    response_v1 = client.get("/api/v1/health")
    assert response_v1.status_code == 200
    assert response_v1.json() == data


def test_health_check_503_when_data_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify /health returns 503 Service Unavailable when data failed to load."""
    empty_loader = DataLoader(data_dir=tmp_path)
    monkeypatch.setattr(loader_module, "loader", empty_loader)
    monkeypatch.setattr(app_main, "loader", empty_loader)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["data_loaded"] is False
    assert data["run_id"] is None
    assert "Manifest not found" in (data["detail"] or "")

    response_v1 = client.get("/api/v1/health")
    assert response_v1.status_code == 503


def test_health_check_503_when_manifest_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify /health returns 503 Service Unavailable when manifest fails schema validation."""
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text('{"invalid": true}', encoding="utf-8")

    invalid_loader = DataLoader(data_dir=tmp_path)
    monkeypatch.setattr(loader_module, "loader", invalid_loader)
    monkeypatch.setattr(app_main, "loader", invalid_loader)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["data_loaded"] is False
    assert "Manifest validation error" in (data["detail"] or "")
