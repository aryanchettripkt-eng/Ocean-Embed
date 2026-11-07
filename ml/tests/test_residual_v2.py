"""Run F: v2 features of the tree-model residual correction and the max-depth guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.inference import argo_correct as ac
from oceanscope_ml.inference import residual_model as rm
from tests.test_argo_correct import BIAS_C, DAYS, LAT, LON, _profiles, _store

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def test_v2_residual_model_learns_the_bias_and_leaves_deep_levels_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    path, depths = _store(tmp_path)
    fit = _profiles(depths, 40, 2900000, seed=1)
    held = _profiles(depths, 5, 5900000, seed=2)
    holdout = set(held["platform_number"])
    seen: list[set[str]] = []

    def fake_profile_levels(levels, depths_m, exclude):  # noqa: ANN001, ANN202
        both = pd.concat([fit, held], ignore_index=True)
        out = both[~both["platform_number"].isin(exclude)].reset_index(drop=True)
        seen.append(set(out["platform_number"]))
        return out

    monkeypatch.setattr(rm, "read_levels", lambda *a: pd.DataFrame())
    monkeypatch.setattr(rm, "profile_levels", fake_profile_levels)
    monkeypatch.setattr(rm, "PARAMS", {**rm.PARAMS, "min_data_in_leaf": 5, "num_threads": 1})
    monkeypatch.setattr(rm, "MAX_ROUNDS", 300)
    dest = rm.run(config, path, [], holdout, "base_run_rm2", out_dir=tmp_path / "out",
                  features="v2", max_depth_m=300.0)  # fmt: skip
    assert seen and not (seen[0] & holdout)

    out = xr.open_zarr(dest)
    names = out.attrs["residual_model"]
    for f in ("near_res_c", "near_wsum", "pred_dtdz", "pred_mld_m"):
        assert f in names
    assert float(out.attrs["residual_model_max_depth_m"]) == 300.0
    band = (depths >= 50) & (depths <= 300)
    before = xr.open_zarr(path)["temp_pred"].values.astype(float)
    after = out["temp_pred"].values.astype(float)
    truth = out["temp_glorys"].values.astype(float)
    assert np.abs(after[:, band] - truth[:, band]).mean() < 0.3 * BIAS_C
    deep = depths > 300
    assert deep.any()
    np.testing.assert_array_equal(after[:, deep], before[:, deep])


def test_near_residual_weights_other_floats_and_leaves_the_own_float_out() -> None:
    res = rm.Residuals(
        np.array(["a", "b", "c"]),
        np.array([10, 10, 0]),  # c is outside the 10-day window
        np.array([10.0, 10.5, 10.0]),
        np.array([85.0, 85.0, 85.0]),
        np.array([[1.0, np.nan], [3.0, 2.0], [9.0, 9.0]]),
    )
    val, wsum = rm._near_residual(res, 10, np.array([10.0]), np.array([85.0]), None)
    assert 1.0 < val[0, 0] < 3.0 and val[0, 1] == pytest.approx(2.0)
    assert wsum[0, 0] > wsum[0, 1] > 0
    val, _ = rm._near_residual(res, 10, np.array([10.0]), np.array([85.0]), np.array(["b"]))
    assert val[0, 0] == pytest.approx(1.0) and np.isnan(val[0, 1])
    far, w = rm._near_residual(res, 10, np.array([-5.0]), np.array([60.0]), None)
    assert np.isnan(far).all() and (w < rm.NEAR_MIN_WEIGHT).all()


def test_profile_shape_finds_the_mixed_layer_from_the_prediction() -> None:
    depths = np.array([0.5, 10.0, 30.0, 65.0, 130.0])
    pred = np.array([[29.0, 29.0, 28.8, 28.0, 20.0], [29.0, 29.0, 29.0, 29.0, 29.0]])
    dtdz, mld = rm._profile_shape(depths, pred)
    assert mld[0] == 65.0 and mld[1] == 130.0
    assert dtdz[0, -1] < 0 and np.abs(dtdz[1]).max() < 1e-12


def test_surface_lagged_is_nan_where_the_earlier_day_is_missing() -> None:
    times = pd.date_range("2019-01-01", periods=30, freq="D")
    sst = np.arange(30, dtype=float)[:, None, None] * np.ones((30, 2, 3))
    dims = ("time", "lat", "lon")
    coords = {"time": times, "lat": [0.0, 1.0], "lon": [80.0, 81.0, 82.0]}
    surface = xr.Dataset({"sst": (dims, sst), "sla": (dims, sst)}, coords=coords)
    out = rm._surface_lagged(surface, times[20])
    np.testing.assert_allclose(out["sst_d14"], 14.0)
    assert np.isnan(out["sla_d28"]).all()


def test_drop_deep_removes_levels_below_the_cutoff() -> None:
    depths = np.array([10.0, 300.0, 643.0])
    res = ac.Residuals(np.array(["a", "b"]), np.array([1, 1]), np.zeros(2), np.zeros(2),
                       np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, 4.0]]))  # fmt: skip
    out = ac.drop_deep(res, depths, 600.0)
    assert len(out) == 1 and np.isnan(out.values[0, 2]) and out.values[0, 1] == 2.0


def test_v2_reads_surface_trends_and_climatology_from_the_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    path, depths = _store(tmp_path)
    fit = _profiles(depths, 30, 2900000, seed=1)
    times = pd.date_range(DAYS[0] - pd.Timedelta(days=40), DAYS[-1], freq="D")
    grid = (times.size, LAT.size, LON.size)
    rng = np.random.default_rng(0)
    dims = ("time", "lat", "lon")
    clim = np.broadcast_to(np.linspace(29.0, 5.0, depths.size)[None, :, None, None],
                           (366, depths.size, LAT.size, LON.size))  # fmt: skip
    parts = xr.Dataset(
        {
            "sst": (dims, 28 + rng.normal(size=grid)),
            "sla": (dims, rng.normal(scale=0.1, size=grid)),
            "u_wind": (dims, rng.normal(size=grid)),
            "v_wind": (dims, rng.normal(size=grid)),
            "thetao": (("time", "depth", "lat", "lon"), np.zeros((times.size, depths.size,
                                                                  LAT.size, LON.size))),
            "clim": (("dayofyear", "depth", "lat", "lon"), clim.copy()),
        },
        coords={"time": times, "depth": depths, "lat": LAT, "lon": LON,
                "dayofyear": np.arange(1, 367)},
    )  # fmt: skip
    monkeypatch.setattr(rm, "open_parts", lambda store_paths, stats_path=None: parts)
    monkeypatch.setattr(rm, "read_levels", lambda *a: pd.DataFrame())
    monkeypatch.setattr(rm, "profile_levels", lambda levels, depths_m, exclude: fit)
    monkeypatch.setattr(rm, "PARAMS", {**rm.PARAMS, "min_data_in_leaf": 5, "num_threads": 1})
    monkeypatch.setattr(rm, "MAX_ROUNDS", 50)
    dest = rm.run(config, path, [], set(), "base_run_rm2", store_paths=[tmp_path / "part.zarr"],
                  out_dir=tmp_path / "out", features="v2")  # fmt: skip
    names = xr.open_zarr(dest).attrs["residual_model"]
    for f in ("sst", "sla", "wind", "pred_anom_c", "sst_d14", "sla_d28", "near_res_c"):
        assert f in names
    assert "thetao" not in names


def test_nearby_features_follow_the_window() -> None:
    res = rm.Residuals(np.array(["a"]), np.array([80]), np.array([10.0]), np.array([85.0]),
                       np.array([[1.0]]))  # fmt: skip
    dist, age = rm._nearest_other(res, 100, np.array([10.0]), np.array([85.0]), None)
    assert dist[0] == rm.FAR_DEG and age[0] == rm.WINDOW_DAYS
    dist, age = rm._nearest_other(res, 100, np.array([10.0]), np.array([85.0]), None, 30)
    assert dist[0] == pytest.approx(0.0) and age[0] == 20
    val, _ = rm._near_residual(res, 100, np.array([10.0]), np.array([85.0]), None, 30)
    assert val[0, 0] == pytest.approx(1.0)
