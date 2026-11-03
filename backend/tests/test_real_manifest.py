"""P6-3 contract check: the backend serves a real (non-synthetic) manifest (PRD_BACKEND §4, §7).

The fixture outputs are copied, then their manifest is switched to `synthetic: false` with ARGO and
headline files null, which is the shape the owner's real run may have. Nothing here touches the
network or the sealed test split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.api.v1.data as data_module
import app.api.v1.meta as meta_module
import app.api.v1.validation as validation_module
import app.data.loader as loader_module
import app.data.store as store_module
from app.data.loader import DataLoader
from app.data.store import OutputStore
from app.main import app

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"


@pytest.fixture()
def real_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["synthetic"] = False
    manifest["model_name"] = "oceanscope-unet-test-shape"
    manifest["files"]["argo"] = None
    manifest["files"]["argo_matches"] = None
    manifest["files"]["headline"] = None
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for entry in FIXTURES_DIR.iterdir():
        if entry.name != "manifest.json":
            os.symlink(entry, tmp_path / entry.name)

    test_loader = DataLoader(data_dir=tmp_path)
    test_loader.load()
    test_store = OutputStore(test_loader)
    monkeypatch.setattr(loader_module, "loader", test_loader)
    monkeypatch.setattr(store_module, "store", test_store)
    monkeypatch.setattr(meta_module, "loader", test_loader)
    monkeypatch.setattr(meta_module, "store", test_store)
    monkeypatch.setattr(data_module, "store", test_store)
    monkeypatch.setattr(validation_module, "store", test_store)
    return TestClient(app)


def test_real_manifest_is_served_as_not_synthetic(real_client: TestClient) -> None:
    meta: dict[str, Any] = real_client.get("/api/v1/meta").json()
    assert meta["synthetic"] is False
    assert meta["model_name"] == "oceanscope-unet-test-shape"
    assert real_client.get("/api/v1/health").json()["data_loaded"] is True


def test_data_endpoints_work_without_synthetic_flag(real_client: TestClient) -> None:
    res = real_client.get("/api/v1/map?date=2020-01-05&depth_m=100.0&field=pred")
    assert res.status_code == 200
    body = res.json()
    assert body["synthetic"] is False
    assert res.headers["Cache-Control"].startswith("public")


def test_missing_argo_and_headline_return_clean_404s(real_client: TestClient) -> None:
    argo = real_client.get("/api/v1/argo")
    assert argo.status_code == 404
    assert argo.json()["error"]["code"] == "argo_not_available"

    headline = real_client.get("/api/v1/headline")
    assert headline.status_code == 404
    assert headline.json()["error"]["code"] == "headline_not_available"


def test_events_and_meta_features_follow_the_real_manifest(real_client: TestClient) -> None:
    events = real_client.get("/api/v1/events").json()
    assert events["synthetic"] is False
    assert events["events"] == []
