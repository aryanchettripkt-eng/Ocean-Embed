"""ARGO residual correction on synthetic data only. No real float data is used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.inference import argo_correct as ac

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
DAYS = pd.date_range("2019-03-01", periods=20, freq="D")
LAT = np.arange(8.125, 14.0, 0.25)
LON = np.arange(84.125, 90.0, 0.25)
BIAS_C = 0.8  # the model is this much too warm everywhere in the band


def _truth(depths: np.ndarray) -> np.ndarray:
    return 29.0 - 0.05 * depths


def _store(tmp_path: Path, days: pd.DatetimeIndex = DAYS) -> tuple[Path, np.ndarray]:
    config = load_config(CONFIG_PATH)
    depths = np.asarray(config.depths_m, dtype=float)
    band = ((depths >= 50) & (depths <= 300)).astype(float)
    shape = (days.size, depths.size, LAT.size, LON.size)
    truth = np.broadcast_to(_truth(depths)[None, :, None, None], shape)
    pred = truth + BIAS_C * band[None, :, None, None]
    dims = ("time", "depth", "lat", "lon")
    ds = xr.Dataset(
        {
            "temp_pred": (dims, pred.astype("float32")),
            "temp_glorys": (dims, truth.astype("float32")),
            "coverage": (("time", "lat", "lon"), np.ones(shape[:1] + shape[2:], dtype="int8")),
        },
        coords={"time": days, "depth": depths, "lat": LAT, "lon": LON},
        attrs={"run_id": "base_run"},
    )
    path = tmp_path / "predictions_base_run.zarr"
    ds.to_zarr(path, mode="w", zarr_format=2, consolidated=True)
    return path, depths


def _profiles(depths: np.ndarray, n_floats: int, first_id: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_floats):
        for c in range(6):
            row = {
                "platform_number": str(first_id + f),
                "cycle": c + 1,
                "day": DAYS[rng.integers(DAYS.size)],
                "lat": float(rng.uniform(LAT[0], LAT[-1])),
                "lon": float(rng.uniform(LON[0], LON[-1])),
            }
            row.update({f"t_{i}": float(t) for i, t in enumerate(_truth(depths))})
            rows.append(row)
    return pd.DataFrame(rows)


def test_correction_removes_a_shared_bias_and_stays_zero_without_floats() -> None:
    depths = np.array([10.0, 100.0])
    res = ac.Residuals(
        np.array(["1", "2"]),
        np.array([100, 101]),
        np.array([10.0, 10.5]),
        np.array([85.0, 85.5]),
        np.array([[0.0, -1.0], [0.0, -1.0]]),
    )
    params = ac.OIParams(length_deg=1.0, tau_days=10.0, window_days=10, noise=0.1)
    near = ac.correction(res, params, 101, np.array([10.2]), np.array([85.2]))
    assert near[0, 1] < -0.8 and abs(near[0, 0]) < 1e-12
    far = ac.correction(res, params, 101, np.array([25.0]), np.array([60.0]))
    assert np.allclose(far, 0.0, atol=1e-6)
    # Future profiles are never used: before day 100 there is nothing to correct with.
    assert np.allclose(ac.correction(res, params, 99, np.array([10.2]), np.array([85.2])), 0.0)
    assert depths.size == near.shape[1]


def test_run_corrects_towards_argo_and_never_reads_held_out_floats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    path, depths = _store(tmp_path)
    fit = _profiles(depths, 30, 2900000, seed=1)
    held = _profiles(depths, 5, 5900000, seed=2)
    holdout = set(held["platform_number"])
    seen: list[set[str]] = []

    def fake_read_levels(paths, start, end):  # noqa: ANN001, ANN202
        return pd.DataFrame()

    def fake_profile_levels(levels, depths_m, exclude):  # noqa: ANN001, ANN202
        both = pd.concat([fit, held], ignore_index=True)
        out = both[~both["platform_number"].isin(exclude)].reset_index(drop=True)
        seen.append(set(out["platform_number"]))
        return out

    monkeypatch.setattr(ac, "read_levels", fake_read_levels)
    monkeypatch.setattr(ac, "profile_levels", fake_profile_levels)
    monkeypatch.setattr(ac, "GRID", {"length_deg": (1.0, 3.0), "tau_days": (10.0,),
                                     "window_days": (30,), "noise": (0.25,)})  # fmt: skip
    dest, params = ac.run(config, path, [], holdout, "base_run_oi", out_dir=tmp_path / "out")
    assert seen and not (seen[0] & holdout)

    out = xr.open_zarr(dest)
    assert out.attrs["run_id"] == "base_run_oi" and out.attrs["base_run_id"] == "base_run"
    band = (depths >= 50) & (depths <= 300)
    before = xr.open_zarr(path)["temp_pred"].values.astype(float)
    after = out["temp_pred"].values.astype(float)
    truth = out["temp_glorys"].values.astype(float)
    # Later days have more floats in the window, so the bias is mostly gone there.
    late = slice(10, None)
    err_before = np.abs(before[late][:, band] - truth[late][:, band]).mean()
    err_after = np.abs(after[late][:, band] - truth[late][:, band]).mean()
    assert err_after < 0.4 * err_before
    # Depths without bias stay (almost) unchanged.
    assert np.abs(after[:, ~band] - before[:, ~band]).max() < 0.02
    table = pd.read_csv(tmp_path / "out" / "metrics" / "argo_correct_tuning_base_run_oi.csv")
    assert table["bob_band_rmse_c"].iloc[1:].min() < table["bob_band_rmse_c"].iloc[0]


def test_refuses_test_dates(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    path, _ = _store(tmp_path, pd.date_range("2021-01-01", periods=3, freq="D"))
    with pytest.raises((TestLeakageError, ValueError)):
        ac.run(config, path, [], {"1"}, "x_oi", out_dir=tmp_path / "out")


def test_per_depth_turns_the_correction_off_where_it_hurts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    path, depths = _store(tmp_path)
    fit = _profiles(depths, 30, 2900000, seed=1)
    rng = np.random.default_rng(3)
    noisy = fit.copy()
    noisy["t_0"] = noisy["t_0"] + rng.normal(0.0, 3.0, len(noisy))  # surface: noise only
    monkeypatch.setattr(ac, "read_levels", lambda paths, start, end: pd.DataFrame())
    monkeypatch.setattr(
        ac, "profile_levels", lambda levels, depths_m, exclude: noisy.reset_index(drop=True)
    )
    monkeypatch.setattr(ac, "GRID", {"length_deg": (1.0, 3.0), "tau_days": (10.0,),
                                     "window_days": (30,), "noise": (0.25,)})  # fmt: skip
    dest, plan = ac.run(config, path, [], set(), "base_run_oid", out_dir=tmp_path / "out",
                        per_depth=True)  # fmt: skip
    band = (depths >= 50) & (depths <= 300)
    assert plan[0] is None  # the noisy surface level is left alone
    assert all(plan[k] is not None for k in np.flatnonzero(band))
    before = xr.open_zarr(path)["temp_pred"].values.astype(float)
    after = xr.open_zarr(dest)["temp_pred"].values.astype(float)
    assert np.array_equal(after[:, 0], before[:, 0])
    choices = pd.read_csv(tmp_path / "out" / "metrics" / "argo_correct_depths_base_run_oid.csv")
    assert len(choices) == depths.size and not bool(choices["corrected"].iloc[0])
