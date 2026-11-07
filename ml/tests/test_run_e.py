"""Run E pieces on synthetic data only: residual model, ensemble weights, regimes, audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.evaluation import matchup_audit as ma
from oceanscope_ml.evaluation import regimes as rg
from oceanscope_ml.inference import ensemble_weights as ew
from oceanscope_ml.inference import residual_model as rm
from oceanscope_ml.inference.ensemble import ensemble
from tests.test_argo_correct import BIAS_C, _profiles, _store

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def test_residual_model_learns_a_systematic_bias_and_skips_held_out_floats(
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
    dest = rm.run(config, path, [], holdout, "base_run_rm", out_dir=tmp_path / "out")
    assert seen and not (seen[0] & holdout)

    out = xr.open_zarr(dest)
    assert out.attrs["run_id"] == "base_run_rm" and out.attrs["base_run_id"] == "base_run"
    band = (depths >= 50) & (depths <= 300)
    before = xr.open_zarr(path)["temp_pred"].values.astype(float)
    after = out["temp_pred"].values.astype(float)
    truth = out["temp_glorys"].values.astype(float)
    err_before = np.abs(before[:, band] - truth[:, band]).mean()
    err_after = np.abs(after[:, band] - truth[:, band]).mean()
    assert err_before == pytest.approx(BIAS_C, abs=1e-3)
    assert err_after < 0.3 * err_before
    assert np.abs(after[:, ~band] - before[:, ~band]).mean() < 0.1
    cv = pd.read_csv(tmp_path / "out" / "metrics" / "residual_model_cv_base_run_rm.csv")
    row = cv.set_index("group").loc["bob_50_300"]
    assert row["rmse_corrected_c"] < 0.3 * row["rmse_none_c"]


def test_nearest_other_leaves_the_point_s_own_float_out() -> None:
    res = rm.Residuals(np.array(["a", "b"]), np.array([10, 8]), np.array([10.0, 12.0]),
                       np.array([85.0, 85.0]), np.zeros((2, 1)))  # fmt: skip
    dist, age = rm._nearest_other(res, 10, np.array([10.0]), np.array([85.0]), np.array(["a"]))
    assert dist[0] == pytest.approx(2.0) and age[0] == 2.0
    dist, _ = rm._nearest_other(res, 10, np.array([10.0]), np.array([85.0]), None)
    assert dist[0] == pytest.approx(0.0)


def test_simplex_grid_and_choose_find_the_best_mix() -> None:
    grid = ew.simplex_grid(3, 0.25)
    assert np.allclose(grid.sum(axis=1), 1.0) and (grid >= 0).all() and len(grid) == 15
    rng = np.random.default_rng(0)
    truth_err = rng.normal(0, 1, 5000)
    # member 0 and 1 have opposite errors, so 50/50 cancels them; member 2 is noise.
    err = np.column_stack([truth_err, -truth_err, rng.normal(0, 1, 5000)])
    w, table = ew.choose(err, np.zeros(5000, bool), 0.05)
    assert np.allclose(w, [0.5, 0.5, 0.0])
    assert len(table) == len(ew.simplex_grid(3, 0.05))


def test_weighted_ensemble_store(tmp_path: Path) -> None:
    from tests.test_ensemble import _store as ens_store

    config = load_config(CONFIG_PATH)
    days = pd.date_range("2019-03-01", periods=5, freq="D")
    a = ens_store(tmp_path, "a", 20.0, days)
    b = ens_store(tmp_path, "c", 24.0, days)
    out = xr.open_zarr(ensemble(config, [a, b], "w", tmp_path / "out", weights=[0.75, 0.25]))
    assert np.allclose(out["temp_pred"].values, 21.0)
    with pytest.raises(ValueError, match="Weights"):
        ensemble(config, [a, b], "w", tmp_path / "out", weights=[0.7, 0.7])


def _matches(n: int, offset: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs = rng.normal(25, 1, n)
    days = pd.Timestamp("2018-11-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    return pd.DataFrame(
        {
            "platform_number": [f"p{k % 9}" for k in range(n)],
            "cycle": np.arange(n) // 9,
            "time": days,
            "lat": rng.uniform(5, 15, n),
            "lon": rng.uniform(80, 90, n),
            "depth_m": np.where(np.arange(n) % 2 == 0, 92.326, 643.567),
            "temp_obs_c": obs,
            "temp_pred_c": obs + offset + rng.normal(0, 0.1, n),
            "temp_glorys_c": obs + 0.2,
            "temp_grep_c": obs - 0.3,
            "temp_armor3d_c": obs + 0.05,
        }
    )


def test_regimes_table(tmp_path: Path) -> None:
    a, b = tmp_path / "argo_matches_ra_dev.parquet", tmp_path / "argo_matches_rb_dev.parquet"
    first = _matches(3000, 0.5, 1)
    first.to_parquet(a)
    second = first.copy()
    second["temp_pred_c"] = first["temp_obs_c"] - 1.0
    second.to_parquet(b)
    events = tmp_path / "events.json"
    events.write_text(
        '{"events": [{"kind": "cyclone", "start": "2019-04-26", "end": "2019-05-05",'
        ' "bbox": {"lat_min": 5, "lat_max": 15, "lon_min": 80, "lon_max": 90}}]}'
    )
    dest = rg.run([a, b], events, "t", tmp_path, fetch=False)
    t = pd.read_csv(dest)
    assert set(t["product"]) == {"ra_dev", "rb_dev", "glorys12", "grep", "armor3d"}
    allrows = t[(t["regime"] == "season") & (t["group"] == "all")]
    assert np.allclose(allrows[allrows["product"] == "glorys12"]["bias_c"], 0.2, atol=1e-6)
    assert np.allclose(allrows[allrows["product"] == "rb_dev"]["rmse_c"], 1.0, atol=1e-6)
    assert set(t[t["regime"] == "iod"]["value"]) == {"unknown"}
    assert {"yes", "no"} <= set(t[t["regime"] == "cyclone"]["value"])
    assert set(t[t["regime"] == "anomaly"]["value"]) == {"cold", "normal", "warm"}


def test_parse_psl_monthly_skips_missing_values() -> None:
    text = "1870 2023\n2019  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 -99.99\n-99.99\n"
    s = rg.parse_psl_monthly(text)
    assert s[pd.Period("2019-11", freq="M")] == pytest.approx(1.1)
    assert pd.Period("2019-12", freq="M") not in s.index


def test_saunders_depth_and_duplicates() -> None:
    z = ma.saunders_depth(np.array([0.0, 1000.0]), np.array([15.0, 15.0]))
    assert z[0] == 0.0 and 990 < z[1] < 995
    levels = pd.DataFrame(
        {
            "platform_number": ["1", "1", "1", "2"],
            "cycle": [1, 1, 1, 1],
            "time": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-01"]),
            "lat": [10.0, 10.0, 10.0, 10.0],
            "lon": [85.0, 85.0, 85.0, 85.0],
            "pres": [10.0, 10.0, 20.0, 10.0],
            "temp_obs_c": [28.0, 28.0, 27.0, 28.0],
        }
    )
    d = ma.duplicates(levels)
    assert d["repeated_float_cycle_pressure_rows"] == 1
    assert d["cycles_with_more_than_one_time"] == 1
    assert d["time_position_shared_by_several_floats"] == 1


def test_time_window_rows_cover_every_shift() -> None:
    from tests.test_argo_validation import MODEL_BETTER, _datasets
    from tests.test_argo_validation import _profiles as av_profiles

    depths = np.asarray(load_config(CONFIG_PATH).depths_m, dtype=float)
    pred, refs = _datasets(depths, MODEL_BETTER, days=pd.date_range("2020-01-01", periods=5))
    profiles = av_profiles()
    profiles["time"] = pd.to_datetime(profiles["time"]) + pd.Timedelta(days=1)  # mid-range
    table = ma.time_window(profiles, pred, refs)
    assert set(table["window"]) == {"-1 day", "same day", "+1 day", "3-day mean"}
    same = table[(table["window"] == "same day") & (table["group"] == "all")].set_index("product")
    assert same.loc["model", "rmse_c"] < same.loc["armor3d", "rmse_c"]


def test_grid_rmse_matches_the_direct_error_for_five_members() -> None:
    rng = np.random.default_rng(0)
    err = rng.normal(size=(2000, 5))
    grid = ew.simplex_grid(5, 0.1)
    direct = np.sqrt(np.mean((err @ grid.T) ** 2, axis=0))
    np.testing.assert_allclose(ew._grid_rmse(err, grid), direct, rtol=1e-10)
