"""Tests for backend health, metadata, CORS, and OpenAPI export (PRD_BACKEND Tests 1, 8, 9)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.v1.meta as meta_module
import app.data.loader as loader_module
import app.data.store as store_module
from app.data.loader import DataLoader
from app.data.store import OutputStore
from app.export_openapi import get_openapi_spec
from app.main import app
from app.schemas.caveats import (
    ARGO_CAVEAT,
    GLORYS_CAVEAT,
    LIVE_CAVEAT,
    SYNTHETIC_CAVEAT,
)


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
    empty_store = OutputStore(empty_loader)
    monkeypatch.setattr(loader_module, "loader", empty_loader)
    monkeypatch.setattr(store_module, "store", empty_store)
    monkeypatch.setattr(meta_module, "loader", empty_loader)
    monkeypatch.setattr(meta_module, "store", empty_store)

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
    invalid_store = OutputStore(invalid_loader)
    monkeypatch.setattr(loader_module, "loader", invalid_loader)
    monkeypatch.setattr(store_module, "store", invalid_store)
    monkeypatch.setattr(meta_module, "loader", invalid_loader)
    monkeypatch.setattr(meta_module, "store", invalid_store)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["data_loaded"] is False
    assert "Manifest validation error" in (data["detail"] or "")


def test_meta_endpoint_success() -> None:
    """Verify /meta and /api/v1/meta return valid metadata adhering to Section 4.2."""
    client = TestClient(app)
    response = client.get("/meta")
    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "public, max-age=300"

    data = response.json()
    assert data["run_id"] == "synthetic_v1"
    assert data["synthetic"] is True
    assert data["dataset_version"] == "v1"
    assert data["model_name"] == "SYNTHETIC_DEMO"
    assert data["region"] == {"lon_min": 40.0, "lon_max": 100.0, "lat_min": 0.0, "lat_max": 30.0}
    assert data["grid_resolution_deg"] == 1.0
    assert data["grid_shape"] == {"lat": 30, "lon": 60}
    assert len(data["depths_m"]) == 15
    assert data["available_dates"] == {"start": "2020-01-01", "end": "2020-01-10"}
    assert data["latest_live_date"] == "2020-01-10"
    assert data["splits"]["train"] == ["1993-03-02", "2017-12-01"]
    assert data["splits"]["val"] == ["2018-01-01", "2020-12-01"]
    assert data["splits"]["test"] == ["2021-01-01", "2021-12-31"]
    assert data["test_evaluated"] is False

    # Color ranges
    assert len(data["color_ranges"]) == 15
    for cr in data["color_ranges"]:
        assert "depth_m" in cr
        assert "vmin" in cr
        assert "vmax" in cr
        assert cr["vmin"] < cr["vmax"]

    assert data["error_range"] == {"vmin": -2.0, "vmax": 2.0}
    assert data["caveats"]["glorys"] == GLORYS_CAVEAT
    assert data["caveats"]["argo"] == ARGO_CAVEAT
    assert data["caveats"]["live"] == LIVE_CAVEAT
    assert data["caveats"]["synthetic"] == SYNTHETIC_CAVEAT
    assert data["chat"]["enabled"] is False

    response_v1 = client.get("/api/v1/meta")
    assert response_v1.status_code == 200
    assert response_v1.json() == data


def test_meta_503_when_data_not_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /meta returns 503 with standardized error format when data is not loaded."""
    empty_loader = DataLoader(data_dir=tmp_path)
    empty_store = OutputStore(empty_loader)
    monkeypatch.setattr(loader_module, "loader", empty_loader)
    monkeypatch.setattr(store_module, "store", empty_store)
    monkeypatch.setattr(meta_module, "loader", empty_loader)
    monkeypatch.setattr(meta_module, "store", empty_store)

    client = TestClient(app)
    response = client.get("/meta")
    assert response.status_code == 503
    err = response.json()
    assert "error" in err
    assert err["error"]["code"] == "data_not_loaded"


def test_cors_headers() -> None:
    """Verify CORS headers for allowed vs disallowed origins (PRD_BACKEND Test 9)."""
    client = TestClient(app)
    # Allowed origin
    res_allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # Disallowed origin
    res_disallowed = client.get("/health", headers={"Origin": "http://malicious-site.example.com"})
    assert "access-control-allow-origin" not in res_disallowed.headers


def test_openapi_spec_up_to_date() -> None:
    """Verify openapi.json on disk is identical to active OpenAPI schema (PRD_BACKEND Test 8)."""
    openapi_file = Path(__file__).resolve().parent.parent / "openapi.json"
    assert openapi_file.exists()
    spec = get_openapi_spec()
    expected_json = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    actual_json = openapi_file.read_text(encoding="utf-8")
    assert actual_json == expected_json
