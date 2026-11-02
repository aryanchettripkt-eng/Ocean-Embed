"""Startup must not wait on the HF download (Render fails a deploy with no open port)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.data.loader as loader_module
import app.main as main_module
from app.data.loader import DataLoader


def test_lifespan_serves_before_the_first_load_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    started = threading.Event()

    class SlowLoader(DataLoader):
        def load(self, data_dir: str | Path | None = None) -> None:
            started.set()
            release.wait(5)

    slow = SlowLoader(autoload=False)
    monkeypatch.setattr(main_module, "loader", slow)
    t0 = time.monotonic()
    with TestClient(main_module.app) as client:
        assert time.monotonic() - t0 < 2
        assert started.wait(2)
        res = client.get("/api/v1/openapi.json")
        assert res.status_code in (200, 404)
        release.set()


def test_download_is_limited_to_manifest_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote_manifest.json"
    remote.write_text(
        json.dumps(
            {
                "files": {
                    "predictions": "predictions/predictions_main2_ft.zarr/",
                    "metrics_val": "metrics/main2_ft_val.csv",
                    "metrics_test": None,
                    "baselines_val": ["metrics/baseline_ridge_val.csv"],
                }
            }
        )
    )
    calls: dict[str, Any] = {}

    def fake_hf_hub_download(**kwargs: Any) -> str:
        assert kwargs["filename"] == "manifest.json"
        return str(remote)

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(loader_module.settings, "HF_DATASET_REPO", "owner/data")

    DataLoader(data_dir=tmp_path / "data")
    patterns = calls["allow_patterns"]
    assert "manifest.json" in patterns
    assert "predictions/predictions_main2_ft.zarr/**" in patterns
    assert "metrics/main2_ft_val.csv" in patterns
    assert "metrics/baseline_ridge_val.csv" in patterns
    assert not any("main2.zarr" in p for p in patterns)


def test_download_falls_back_to_everything_without_a_remote_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def failing_hf_hub_download(**kwargs: Any) -> str:
        raise OSError("offline")

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(tmp_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", failing_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(loader_module.settings, "HF_DATASET_REPO", "owner/data")

    DataLoader(data_dir=tmp_path / "data")
    assert calls["allow_patterns"] is None
