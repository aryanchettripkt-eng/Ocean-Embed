"""Tests for live date handling and manifest reload behavior.

Verifies PRD_BACKEND Sections 3, 4.3, 7, and 9 (Tests 10, 12), and Milestone B6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app.api.v1.meta as meta_module
import app.data.loader as loader_module
from app.data.loader import DataLoader
from app.data.store import OutputStore
from app.main import app
from app.schemas.caveats import LIVE_CAVEAT


def test_live_dates_full_contract() -> None:
    """Verify Test 10 / Milestone B6: live dates contract and LIVE_CAVEAT."""
    client = TestClient(app)

    # 1. /meta serves LIVE_CAVEAT and latest_live_date
    res_meta = client.get("/api/v1/meta")
    assert res_meta.status_code == 200
    meta_json = res_meta.json()
    assert meta_json["latest_live_date"] == "2020-01-10"
    assert meta_json["caveats"]["live"] == LIVE_CAVEAT

    # 2. /map?field=pred succeeds on live date and returns split='live'
    res_map_pred = client.get("/api/v1/map?date=2020-01-09&depth_index=0&field=pred")
    assert res_map_pred.status_code == 200
    assert res_map_pred.json()["split"] == "live"

    # 3. /map?field=glorys and field=error return 404 reference_unavailable on live date
    res_map_glorys = client.get("/api/v1/map?date=2020-01-09&depth_index=0&field=glorys")
    assert res_map_glorys.status_code == 404
    assert res_map_glorys.json()["error"]["code"] == "reference_unavailable"

    res_map_error = client.get("/api/v1/map?date=2020-01-09&depth_index=0&field=error")
    assert res_map_error.status_code == 404
    assert res_map_error.json()["error"]["code"] == "reference_unavailable"

    # 4. /profile returns split='live' with null glorys and null error
    res_profile = client.get("/api/v1/profile?lat=12.0&lon=65.0&date=2020-01-09")
    assert res_profile.status_code == 200
    prof = res_profile.json()
    assert prof["split"] == "live"
    assert all(g is None for g in prof["glorys"])
    assert all(e is None for e in prof["error"])


def test_manifest_reload_success(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify Test 12: replacing manifest.json updates data without server restart."""
    fixture_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"

    # Copy manifest to tmp_path and link prediction store
    with open(fixture_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Initial manifest
    manifest_data["active_run_id"] = "initial_run_v1"
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    test_loader = DataLoader(data_dir=tmp_path)
    test_loader.load()
    test_store = OutputStore(test_loader)

    monkeypatch.setattr(loader_module, "loader", test_loader)
    monkeypatch.setattr(meta_module, "loader", test_loader)
    monkeypatch.setattr(meta_module, "store", test_store)

    client = TestClient(app)
    res1 = client.get("/api/v1/health")
    assert res1.status_code == 200
    assert res1.json()["run_id"] == "initial_run_v1"

    # Replace manifest with updated active_run_id (simulating live mode day append)
    manifest_data["active_run_id"] = "updated_run_v2"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # Reload manifest
    test_loader.load()

    # Verify updated manifest is served immediately without restart
    res2 = client.get("/api/v1/health")
    assert res2.status_code == 200
    assert res2.json()["run_id"] == "updated_run_v2"


def test_manifest_reload_broken_keeps_previous(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify Test 12: a broken new manifest keeps serving the old valid manifest."""
    fixture_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"

    with open(fixture_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_data["active_run_id"] = "stable_run_v1"
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    test_loader = DataLoader(data_dir=tmp_path)
    test_loader.load()
    test_store = OutputStore(test_loader)

    monkeypatch.setattr(loader_module, "loader", test_loader)
    monkeypatch.setattr(meta_module, "loader", test_loader)
    monkeypatch.setattr(meta_module, "store", test_store)

    client = TestClient(app)
    res1 = client.get("/api/v1/health")
    assert res1.status_code == 200
    assert res1.json()["run_id"] == "stable_run_v1"

    # Overwrite with broken/invalid JSON
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write('{"schema_version": 1, "broken": "missing required fields"}')

    # Trigger reload
    test_loader.load()

    # Must still serve stable_run_v1 and remain healthy
    res2 = client.get("/api/v1/health")
    assert res2.status_code == 200
    assert res2.json()["run_id"] == "stable_run_v1"
    assert test_loader.is_loaded is True
