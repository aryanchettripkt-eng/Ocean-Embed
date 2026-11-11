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


class _File:
    def __init__(self, path: str, blob: str) -> None:
        self.path = path
        self.blob_id = blob


class _Folder:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeHub:
    """A dataset with two runs; only `files` of the remote manifest may be listed or fetched."""

    def __init__(self, tmp_path: Path, manifest: dict[str, Any]) -> None:
        self.remote = tmp_path / "remote_manifest.json"
        self.remote.write_text(json.dumps(manifest))
        self.sha = "sha1"
        self.tree: dict[str, str] = {
            "predictions/predictions_main2_ft.zarr/.zmetadata": "b1",
            "predictions/predictions_main2_ft.zarr/temp_pred/0.0.0.0": "b2",
            "predictions/predictions_main2.zarr/.zmetadata": "old-run",
            "metrics/main2_ft_val.csv": "b3",
            "metrics/baseline_ridge_val.csv": "b4",
        }
        self.fetched: list[str] = []
        self.listed: list[str] = []
        self.fail_on: str | None = None
        self.offline = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huggingface_hub

        hub = self

        class Api:
            def __init__(self, token: str | None = None) -> None:
                pass

            def dataset_info(self, repo_id: str, revision: str) -> Any:
                if hub.offline:
                    raise OSError("offline")
                return type("Info", (), {"sha": hub.sha})()

            def get_paths_info(self, repo_id: str, paths: list[str], **kw: Any) -> list[Any]:
                out: list[Any] = []
                for p in paths:
                    if p in hub.tree:
                        out.append(_File(p, hub.tree[p]))
                    elif any(k.startswith(p + "/") for k in hub.tree):
                        out.append(_Folder(p))
                return out

            def list_repo_tree(self, repo_id: str, path_in_repo: str, **kw: Any) -> list[Any]:
                hub.listed.append(path_in_repo)
                prefix = path_in_repo + "/"
                return [_File(k, v) for k, v in hub.tree.items() if k.startswith(prefix)]

        def fake_download(**kwargs: Any) -> str:
            name = kwargs["filename"]
            if name == "manifest.json":
                return str(hub.remote)
            if name == hub.fail_on:
                raise OSError("connection reset")
            hub.fetched.append(name)
            target = Path(kwargs["local_dir"]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(hub.tree[name])
            return str(target)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        monkeypatch.setattr(loader_module.settings, "HF_DATASET_REPO", "owner/data")


MANIFEST = {
    "files": {
        "predictions": "predictions/predictions_main2_ft.zarr/",
        "metrics_val": "metrics/main2_ft_val.csv",
        "metrics_test": None,
        "baselines_val": ["metrics/baseline_ridge_val.csv"],
    }
}


def _sync(loader: DataLoader) -> None:
    loader._sync_from_hf()


def test_download_is_limited_to_manifest_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path, MANIFEST)
    hub.install(monkeypatch)
    loader = DataLoader(data_dir=tmp_path / "data", autoload=False)
    _sync(loader)

    assert sorted(hub.fetched) == [
        "metrics/baseline_ridge_val.csv",
        "metrics/main2_ft_val.csv",
        "predictions/predictions_main2_ft.zarr/.zmetadata",
        "predictions/predictions_main2_ft.zarr/temp_pred/0.0.0.0",
    ]
    assert hub.listed == ["predictions/predictions_main2_ft.zarr"]  # never the whole dataset
    assert json.loads((tmp_path / "data" / "manifest.json").read_text()) == MANIFEST


def test_a_new_commit_fetches_only_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path, MANIFEST)
    hub.install(monkeypatch)
    loader = DataLoader(data_dir=tmp_path / "data", autoload=False)
    _sync(loader)

    hub.fetched.clear()
    _sync(loader)  # same commit: nothing to do
    assert hub.fetched == []

    hub.sha = "sha2"  # a live day: new chunk and new metadata
    hub.tree["predictions/predictions_main2_ft.zarr/.zmetadata"] = "b1-new"
    hub.tree["predictions/predictions_main2_ft.zarr/temp_pred/1.0.0.0"] = "b5"
    _sync(loader)
    assert sorted(hub.fetched) == [
        "predictions/predictions_main2_ft.zarr/.zmetadata",
        "predictions/predictions_main2_ft.zarr/temp_pred/1.0.0.0",
    ]


def test_a_failed_download_keeps_the_old_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path, MANIFEST)
    hub.install(monkeypatch)
    loader = DataLoader(data_dir=tmp_path / "data", autoload=False)
    _sync(loader)
    old = (tmp_path / "data" / "manifest.json").read_text()

    new_manifest = {"files": {**MANIFEST["files"], "metrics_val": "metrics/main3_val.csv"}}
    hub.remote.write_text(json.dumps(new_manifest))
    hub.tree["metrics/main3_val.csv"] = "b9"
    hub.sha = "sha2"
    hub.fail_on = "metrics/main3_val.csv"
    _sync(loader)
    assert (tmp_path / "data" / "manifest.json").read_text() == old  # still the complete run

    hub.fail_on = None  # the next reload retries and completes
    hub.fetched.clear()
    _sync(loader)
    assert hub.fetched == ["metrics/main3_val.csv"]
    assert json.loads((tmp_path / "data" / "manifest.json").read_text()) == new_manifest


def test_hf_unreachable_never_downloads_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub(tmp_path, MANIFEST)
    hub.offline = True
    hub.install(monkeypatch)
    loader = DataLoader(data_dir=tmp_path / "data", autoload=False)
    loader.load()
    assert hub.fetched == []
    assert not loader.is_loaded
    assert "HF dataset" in (loader.load_error or "")


def test_reload_with_an_unchanged_manifest_keeps_the_open_store(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "sample_outputs" / "manifest.json"
    (data / "manifest.json").write_text(fixture.read_text())
    loader = DataLoader(data_dir=data)
    first = loader.manifest
    assert first is not None
    loader.load()
    assert loader.manifest is first  # same object: the store keeps its dataset and caches

    changed = json.loads(fixture.read_text())
    changed["available_dates"]["end"] = "2020-01-09"
    (data / "manifest.json").write_text(json.dumps(changed))
    loader.load()
    assert loader.manifest is not first
