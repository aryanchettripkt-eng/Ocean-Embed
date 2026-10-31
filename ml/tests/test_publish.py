"""Publishing a run: checks before upload, only this run's files, manifest.json last (§6.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.evaluation.evaluate import evaluate_predictions, write_manifest
from oceanscope_ml.inference.predict import predict_store
from oceanscope_ml.publish import publish
from oceanscope_ml.train import train_run


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, **kw: Any) -> None:
        self.calls.append(("create_repo", kw))

    def upload_folder(self, **kw: Any) -> None:
        self.calls.append(("upload_folder", kw))

    def upload_file(self, **kw: Any) -> None:
        self.calls.append(("upload_file", kw))


@pytest.fixture
def outputs(config: ExperimentConfig, store: Path) -> Path:
    train_run(config, store, "run1", device_choice="cpu", epochs=1)
    ckpt = Path(config.paths.checkpoints) / "run1" / "best.pt"
    pred = predict_store(config, store, ckpt, device_choice="cpu")
    write_manifest(config, pred, evaluate_predictions(config, pred, store))
    return Path(config.paths.outputs)


def test_uploads_run_files_then_manifest_last(config: ExperimentConfig, outputs: Path) -> None:
    (outputs / "metrics" / "stale_other_run_val.csv").write_text("x\n")
    api = FakeApi()
    publish(config, repo_id="org/repo", api=api)
    names = [c[0] for c in api.calls]
    assert names == ["create_repo", "upload_folder", "upload_file"]
    folder = api.calls[1][1]
    assert folder["repo_type"] == "dataset"
    assert "predictions/predictions_run1.zarr/**" in folder["allow_patterns"]
    assert "metrics/run1_val.csv" in folder["allow_patterns"]
    assert not any("stale" in p for p in folder["allow_patterns"])
    assert api.calls[2][1]["path_in_repo"] == "manifest.json"


def test_dry_run_uploads_nothing(config: ExperimentConfig, outputs: Path) -> None:
    api = FakeApi()
    publish(config, repo_id="org/repo", dry_run=True, api=api)
    assert api.calls == []


def test_refuses_missing_files_and_synthetic_manifests(
    config: ExperimentConfig, outputs: Path
) -> None:
    manifest_path = outputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    (outputs / manifest["files"]["metrics_val"]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        publish(config, repo_id="org/repo", api=FakeApi())

    manifest["synthetic"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="synthetic"):
        publish(config, repo_id="org/repo", api=FakeApi())
