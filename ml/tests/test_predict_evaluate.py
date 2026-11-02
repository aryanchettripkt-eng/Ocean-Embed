"""Prediction store, validation metrics and manifest from a trained checkpoint (PRD §6.2, §6.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.evaluation.evaluate import evaluate_predictions, write_manifest
from oceanscope_ml.inference.predict import predict_store, split_runs
from oceanscope_ml.train import train_run


@pytest.fixture
def checkpoint(config: ExperimentConfig, store: Path) -> Path:
    train_run(config, store, "run1", device_choice="cpu", epochs=1)
    return Path(config.paths.checkpoints) / "run1" / "best.pt"


def test_predict_evaluate_and_manifest(
    config: ExperimentConfig, store: Path, checkpoint: Path
) -> None:
    pred_path = predict_store(config, store, checkpoint, device_choice="cpu")
    assert pred_path.name == "predictions_run1.zarr"
    pred = xr.open_zarr(pred_path)
    dates = pd.to_datetime(pred["time"].values)
    assert dates.min() >= pd.Timestamp(config.splits.val[0])
    assert dates.max() <= pd.Timestamp(config.splits.val[1])
    assert pred["temp_pred"].dtype == np.float16
    assert pred["temp_pred"].encoding["chunks"] == (1, 15, 9, 11)
    assert json.loads(pred.attrs["split_of_each_date"]) == {
        f"{dates[0].date()}/{dates[-1].date()}": "val"
    }

    cov = pred["coverage"].values
    temp = pred["temp_pred"].values.astype(np.float32)
    assert set(np.unique(cov)) <= {0, 1, 2} and (cov == 1).any()
    assert np.isnan(temp[np.broadcast_to((cov != 1)[:, None], temp.shape)]).all()
    surface = temp[:, 0][cov == 1]
    assert np.isfinite(surface).all() and 0.0 < float(np.mean(surface)) < 40.0

    paths = evaluate_predictions(config, pred_path, store)
    val = pd.read_csv(paths["metrics_val"])
    assert list(val.columns) == [
        "model", "split", "depth_m", "n", "mae_c", "rmse_c", "bias_c", "skill_vs_clim_pct"
    ]  # fmt: skip
    assert len(val) == 16 and val["depth_m"].iloc[-1] == "all"
    assert (val["n"] > 0).all() and (val["rmse_c"] >= val["mae_c"]).all()
    season = pd.read_csv(paths["metrics_val_by_season"])
    assert set(season["season"]) == {"DJF", "MAM", "JJA", "SON"}
    region = pd.read_csv(paths["metrics_val_by_region"])
    assert set(region["region"]) == {"arabian_sea", "bay_of_bengal", "equatorial"}

    manifest_path = write_manifest(config, pred_path, paths)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["active_run_id"] == "run1" and manifest["synthetic"] is False
    assert manifest["files"]["predictions"] == "predictions/predictions_run1.zarr"
    assert manifest["files"]["metrics_test"] is None and manifest["test_evaluated"] is False
    assert manifest["available_dates"]["start"] == str(dates[0].date())


def test_predict_never_writes_test_dates(
    config: ExperimentConfig, store: Path, checkpoint: Path
) -> None:
    config.inference["publish_splits"] = ["test"]
    with pytest.raises(ValueError, match="no split predict may write"):
        predict_store(config, store, checkpoint, device_choice="cpu")


def test_split_runs_groups_unbroken_days() -> None:
    dates = pd.DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-05"])
    assert split_runs(dates, "val") == {
        "2018-01-01/2018-01-02": "val",
        "2018-01-05/2018-01-05": "val",
    }
