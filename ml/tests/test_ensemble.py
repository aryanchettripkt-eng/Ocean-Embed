"""Ensemble of prediction stores, on synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.inference.ensemble import ensemble

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
LAT = np.arange(8.125, 9.0, 0.25)
LON = np.arange(84.125, 85.0, 0.25)
DEPTHS = np.array([5.0, 100.0])


def _store(tmp_path: Path, run_id: str, value: float, days: pd.DatetimeIndex) -> Path:
    shape = (days.size, DEPTHS.size, LAT.size, LON.size)
    cov = np.ones((days.size, LAT.size, LON.size), dtype="int8")
    if run_id == "b":
        cov[:, 0, 0] = 0
    dims = ("time", "depth", "lat", "lon")
    ds = xr.Dataset(
        {
            "temp_pred": (dims, np.full(shape, value, dtype="float32")),
            "temp_glorys": (dims, np.full(shape, 20.0, dtype="float32")),
            "coverage": (("time", "lat", "lon"), cov),
        },
        coords={"time": days, "depth": DEPTHS, "lat": LAT, "lon": LON},
        attrs={"run_id": run_id},
    )
    path = tmp_path / f"predictions_{run_id}.zarr"
    ds.to_zarr(path, mode="w", zarr_format=2, consolidated=True)
    return path


def test_mean_coverage_and_attrs(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    days = pd.date_range("2019-03-01", periods=20, freq="D")
    a = _store(tmp_path, "a", 21.0, days)
    b = _store(tmp_path, "b", 23.0, days)
    dest = ensemble(config, [a, b], "ens2", out_dir=tmp_path / "out")
    out = xr.open_zarr(dest)
    assert out.attrs["run_id"] == "ens2"
    temp = out["temp_pred"].values
    assert np.allclose(temp[:, :, 1:, :], 22.0)
    assert out["coverage"].values[:, 0, 0].max() == 0 and np.isnan(temp[:, :, 0, 0]).all()
    assert np.allclose(out["temp_glorys"].values, 20.0)


def test_refuses_test_dates_and_mismatched_grids(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    val = pd.date_range("2019-03-01", periods=5, freq="D")
    a = _store(tmp_path, "a", 21.0, val)
    test = _store(tmp_path / "t", "b", 23.0, pd.date_range("2021-01-01", periods=5, freq="D"))
    with pytest.raises((TestLeakageError, ValueError)):
        ensemble(config, [a, test], "x", out_dir=tmp_path / "out")
    other = _store(tmp_path / "o", "b", 23.0, pd.date_range("2019-03-02", periods=5, freq="D"))
    with pytest.raises(ValueError, match="time"):
        ensemble(config, [a, other], "x", out_dir=tmp_path / "out")
