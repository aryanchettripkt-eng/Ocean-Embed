"""Tests for backend validation endpoints: /metrics, /argo, /argo/profiles, /headline.

Verifies PRD_BACKEND Sections 4.7-4.9, 7, 9 (Tests 2, 7), and Milestone B3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import app.api.v1.validation as validation_module
import app.data.loader as loader_module
from app.data.loader import DataLoader
from app.data.store import OutputStore
from app.main import app
from app.schemas.caveats import ARGO_CAVEAT, HEADLINE_CRITERION


def test_metrics_depth_happy_path() -> None:
    """Verify /metrics?split=val&group_by=depth returns model and baseline rows."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics?split=val&group_by=depth")
    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "public, max-age=86400"
    assert "ETag" in response.headers

    data = response.json()
    assert data["split"] == "val"
    assert data["group_by"] == "depth"
    assert data["synthetic"] is True
    assert len(data["rows"]) > 0

    # Models present: SYNTHETIC_DEMO, climatology, ridge
    models = {r["model"] for r in data["rows"]}
    assert "SYNTHETIC_DEMO" in models
    assert "climatology" in models
    assert "ridge" in models

    # Check a specific row contract
    sample_row = next(
        r for r in data["rows"] if r["model"] == "SYNTHETIC_DEMO" and r["depth_m"] == 0.5
    )
    assert sample_row["split"] == "val"
    assert sample_row["group"] is None
    assert isinstance(sample_row["mae_c"], float)
    assert isinstance(sample_row["rmse_c"], float)
    assert isinstance(sample_row["skill_vs_clim_pct"], float)

    # Check summary row with depth_m == "all"
    all_row = next(
        r for r in data["rows"] if r["model"] == "SYNTHETIC_DEMO" and r["depth_m"] == "all"
    )
    assert all_row["n"] > 0


def test_metrics_season_and_region_grouping() -> None:
    """Verify /metrics with season and region groupings."""
    client = TestClient(app)

    # Season grouping
    res_season = client.get("/api/v1/metrics?split=val&group_by=season")
    assert res_season.status_code == 200
    data_season = res_season.json()
    assert data_season["group_by"] == "season"
    seasons = {r["group"] for r in data_season["rows"]}
    assert {"DJF", "MAM", "JJA", "SON"}.issubset(seasons)

    # Region grouping
    res_region = client.get("/api/v1/metrics?split=val&group_by=region")
    assert res_region.status_code == 200
    data_region = res_region.json()
    assert data_region["group_by"] == "region"
    regions = {r["group"] for r in data_region["rows"]}
    assert {"arabian_sea", "bay_of_bengal", "equatorial"}.issubset(regions)


def test_metrics_test_split_unsealing_check() -> None:
    """Verify Test 2 / §0: split=test returns 404 test_not_evaluated when test is sealed."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics?split=test&group_by=depth")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "test_not_evaluated"


def test_argo_validation_metrics_and_caveat() -> None:
    """Verify /argo returns competitor comparison and exact ARGO caveat (§4.8, §7)."""
    client = TestClient(app)
    response = client.get("/api/v1/argo")
    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "public, max-age=86400"
    assert "ETag" in response.headers

    data = response.json()
    assert data["caveat"] == ARGO_CAVEAT
    assert data["n_profiles_total"] == 3
    assert data["n_obs_total"] == 45
    assert len(data["rows"]) == 15

    # Check row columns (including GREP and ARMOR3D)
    row0 = data["rows"][0]
    assert row0["depth_m"] == 0.5
    assert row0["n_profiles"] == 3
    assert row0["n_obs"] == 3
    assert isinstance(row0["glorys_rmse_c"], float)
    assert isinstance(row0["grep_rmse_c"], float)
    assert isinstance(row0["armor3d_rmse_c"], float)


def test_argo_profiles_markers_and_filters() -> None:
    """Verify /argo/profiles map markers, date filtering, and bbox validation (§4.9)."""
    client = TestClient(app)

    # All profiles
    res_all = client.get("/api/v1/argo/profiles")
    assert res_all.status_code == 200
    assert res_all.headers.get("Cache-Control") == "public, max-age=86400"
    assert "ETag" in res_all.headers
    data_all = res_all.json()
    assert len(data_all["profiles"]) == 3

    # Date filter matching one profile
    res_date = client.get("/api/v1/argo/profiles?date=2020-01-03")
    assert res_date.status_code == 200
    profiles = res_date.json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["platform_number"] == "2902123"
    assert profiles[0]["cycle"] == 14
    assert profiles[0]["n_obs"] == 15

    # Date filter matching none
    res_none = client.get("/api/v1/argo/profiles?date=2020-01-01")
    assert res_none.status_code == 200
    assert len(res_none.json()["profiles"]) == 0

    # Bounding box filter
    res_bbox = client.get("/api/v1/argo/profiles?lat_min=10.0&lat_max=15.0")
    assert res_bbox.status_code == 200
    assert len(res_bbox.json()["profiles"]) == 1
    assert res_bbox.json()["profiles"][0]["lat"] == 12.5

    # Invalid bounding box: lat_min > lat_max
    res_inv = client.get("/api/v1/argo/profiles?lat_min=20.0&lat_max=10.0")
    assert res_inv.status_code == 422
    assert res_inv.json()["error"]["code"] == "out_of_region"


def test_headline_criterion_and_met(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify /headline handles n/a rows and computes met=True, met=False, and met=None (§4.8b)."""
    client = TestClient(app)
    response = client.get("/api/v1/headline")
    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "public, max-age=86400"
    assert "ETag" in response.headers

    data = response.json()
    assert data["criterion"] == HEADLINE_CRITERION
    # In default fixtures, val competitors all have diff_ci_high_c < 0.0 -> met is True
    assert data["met"] is True
    assert len(data["rows"]) == 6

    # Test rows have null values
    test_rows = [r for r in data["rows"] if r["period"] == "test"]
    assert len(test_rows) == 2
    for r in test_rows:
        assert r["rmse_c"] is None
        assert r["diff_ci_high_c"] is None

    # Test case: met is None when evaluated period has no competitor data
    fixture_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"
    with open(fixture_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest_data["test_evaluated"] = True
    custom_loader = DataLoader(data_dir=fixture_dir)
    custom_loader.manifest = manifest_data
    custom_store = OutputStore(custom_loader)
    monkeypatch.setattr(loader_module, "loader", custom_loader)
    monkeypatch.setattr(validation_module, "store", custom_store)

    res_test = client.get("/api/v1/headline")
    assert res_test.status_code == 200
    assert res_test.json()["met"] is None

    # Test case: met is False when a competitor has diff_ci_high_c >= 0.0
    bad_headline_csv = tmp_path / "bad_headline.csv"
    bad_headline_csv.write_text(
        "period,product,n_floats,n_obs,rmse_c,ci_low_c,ci_high_c,diff_vs_model_c,diff_ci_low_c,diff_ci_high_c\n"
        "val,model,3,15,0.28,0.24,0.32,0.0,0.0,0.0\n"
        "val,glorys12,3,15,0.25,0.21,0.30,0.03,-0.01,0.05\n",
        encoding="utf-8",
    )
    manifest_data["test_evaluated"] = False
    manifest_data["files"]["headline"] = str(bad_headline_csv.name)
    custom_loader_bad = DataLoader(data_dir=tmp_path)
    custom_loader_bad.manifest = manifest_data
    custom_store_bad = OutputStore(custom_loader_bad)
    monkeypatch.setattr(loader_module, "loader", custom_loader_bad)
    monkeypatch.setattr(validation_module, "store", custom_store_bad)

    res_false = client.get("/api/v1/headline")
    assert res_false.status_code == 200
    assert res_false.json()["met"] is False


def test_headline_not_available_when_missing(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify /headline returns 404 headline_not_available if manifest omits headline."""
    fixture_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"

    # Create temporary manifest with headline=None
    with open(fixture_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_data["files"]["headline"] = None
    tmp_manifest = tmp_path / "manifest.json"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # Monkeypatch loader and store with headline=None manifest
    custom_loader = DataLoader(data_dir=fixture_dir)
    custom_loader.manifest = manifest_data
    custom_store = OutputStore(custom_loader)
    monkeypatch.setattr(loader_module, "loader", custom_loader)
    import app.api.v1.validation as validation_module

    monkeypatch.setattr(validation_module, "store", custom_store)

    client = TestClient(app)
    response = client.get("/api/v1/headline")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "headline_not_available"
