"""Final test (P8-1): owner go, once only, and the test split is read only through final_test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.evaluation.evaluate import evaluate_predictions, write_manifest
from oceanscope_ml.evaluation.final_test import check_may_run, run_final_test
from oceanscope_ml.inference.predict import predict_store
from oceanscope_ml.train import train_run


def _validation_run(config: ExperimentConfig, store: Path) -> Path:
    """Train one epoch, predict validation days and write the validation manifest."""
    train_run(config, store, "run1", device_choice="cpu", epochs=1)
    ckpt = Path(config.paths.checkpoints) / "run1" / "best.pt"
    pred = predict_store(config, store, ckpt, device_choice="cpu")
    write_manifest(config, pred, evaluate_predictions(config, pred, store))
    return ckpt


def test_refuses_without_owner_go(config: ExperimentConfig, tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="run the final test"):
        check_may_run(tmp_path, go=False)


def test_test_split_is_refused_outside_final_test(config: ExperimentConfig, store: Path) -> None:
    train_run(config, store, "run1", device_choice="cpu", epochs=1)
    ckpt = Path(config.paths.checkpoints) / "run1" / "best.pt"
    with pytest.raises(TestLeakageError, match="final_test"):
        predict_store(config, store, ckpt, device_choice="cpu", split="test")


def test_scores_test_period_once_and_flags_manifest(config: ExperimentConfig, store: Path) -> None:
    ckpt = _validation_run(config, store)
    out = Path(config.paths.outputs)

    csv = run_final_test(config, store, ckpt, go=True, device_choice="cpu")

    assert csv == out / "metrics" / "run1_test.csv"
    rows = pd.read_csv(csv)
    assert set(rows["split"]) == {"test"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_evaluated"] is True
    assert manifest["files"]["metrics_test"] == "metrics/run1_test.csv"
    # Sealed: a second run is refused, and the validation files are untouched.
    with pytest.raises(FileExistsError):
        run_final_test(config, store, ckpt, go=True, device_choice="cpu")
