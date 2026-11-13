"""Bayesian last layer with Batch-and-Match (evaluation.uncertainty)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.evaluation import uncertainty as un
from oceanscope_ml.inference.predict import predict_store
from oceanscope_ml.train import train_run


def test_bam_recovers_a_gaussian_target() -> None:
    rng = np.random.default_rng(0)
    d = 4
    a = rng.standard_normal((d, d))
    cov_true = a @ a.T + 0.5 * np.eye(d)
    mu_true = rng.standard_normal(d)
    prec = np.linalg.inv(cov_true)
    mu, cov = un.bam_fit(lambda x: -(x - mu_true) @ prec, np.zeros(d), np.eye(d),
                         niter=100, batch=16)  # fmt: skip
    np.testing.assert_allclose(mu, mu_true, atol=1e-3)
    np.testing.assert_allclose(cov, cov_true, atol=1e-2 * np.abs(cov_true).max())


def _synthetic(n_floats: int, seed: int) -> pd.DataFrame:
    """Errors whose size grows with feature f0; the constant error bar can't follow that."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_floats):
        for cycle in range(30):
            f = rng.standard_normal(3)
            for depth in (5.0, 100.0):
                sd = 0.3 * np.exp(0.6 * f[0]) * (2 if depth == 100.0 else 1)
                pred = 20.0 + rng.standard_normal()
                rows.append({"platform_number": f"{k:07d}", "cycle": cycle,
                             "lat": 10.0 + k % 5, "lon": 70.0 + 5 * (k % 4), "depth_m": depth,
                             "temp_pred_c": pred, "temp_obs_c": pred + sd * rng.standard_normal(),
                             "f0": f[0], "f1": f[1], "f2": f[2]})  # fmt: skip
    return pd.DataFrame(rows)


def test_last_layer_is_calibrated_where_the_constant_bar_is_not() -> None:
    fit = un.fit_heads(_synthetic(40, 1), niter=150)
    scored = un.predictive(_synthetic(40, 2), fit)
    table = un.coverage_table(scored).set_index(["group", "method"])
    for p in (68, 95):
        assert abs(table.loc[("all", "bll_bam"), f"cover{p}"] - p / 100) < 0.03
    # Where errors are small (f0 < -1) the constant bar is far too wide
    low = scored[scored["f0"] < -1]
    assert un._cover(low["const_pit"].to_numpy(), 0.68) > 0.85
    assert abs(un._cover(low["bll_bam_pit"].to_numpy(), 0.68) - 0.68) < 0.08
    rel = un.reliability_table(scored)
    assert set(rel["method"]) == set(un.METHODS) and rel["observed"].between(0, 1).all()


def test_row_weights_count_each_profile_level_once() -> None:
    df = pd.DataFrame({"platform_number": ["a"] * 3 + ["b"], "cycle": [1, 1, 1, 1],
                       "depth_m": [5.0, 5.0, 100.0, 5.0], "temp_obs_c": 1.0})  # fmt: skip
    np.testing.assert_allclose(un.row_weights(df), [0.5, 0.5, 1.0, 1.0])


def test_feature_sampler_picks_the_nearest_cell_on_the_right_day() -> None:
    profiles = pd.DataFrame({"platform_number": ["a", "a", "b"], "cycle": [1, 1, 7],
                             "time": pd.to_datetime(["2018-01-02 10:00"] * 2 + ["2018-01-03"],
                                                    format="mixed"),
                             "lat": [0.13, 0.13, 0.6], "lon": [0.4, 0.4, 0.1]})  # fmt: skip
    lat = lon = np.array([0.125, 0.375, 0.625])
    sampler = un.FeatureSampler(profiles, lat, lon)
    feats = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
    sampler(pd.DatetimeIndex(["2018-01-02", "2018-01-03"]), feats)
    t = sampler.table().set_index("platform_number")
    assert t.loc["a", "f0"] == feats[0, 0, 0, 1] and t.loc["a", "f1"] == feats[0, 1, 0, 1]
    assert t.loc["b", "f0"] == feats[1, 0, 2, 0]


def test_collect_refuses_the_test_year_without_go(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        un.collect(None, tmp_path, [], [], tmp_path, "test", out_dir=tmp_path)  # type: ignore[arg-type]


def test_predict_feature_sink_gets_the_head_input(config: ExperimentConfig, store: Path) -> None:
    train_run(config, store, "run1", device_choice="cpu", epochs=1)
    ckpt = Path(config.paths.checkpoints) / "run1" / "best.pt"
    seen: list[tuple[pd.DatetimeIndex, np.ndarray]] = []
    pred_path = predict_store(config, store, ckpt, device_choice="cpu",
                              feature_sink=lambda d, f: seen.append((d, f)))  # fmt: skip
    pred = xr.open_zarr(pred_path)
    dates = pd.DatetimeIndex(np.concatenate([d.values for d, _ in seen]))
    assert dates.equals(pd.DatetimeIndex(pred["time"].values))
    feats = seen[0][1]
    assert feats.shape[1] == int(config.train.get("base_channels", 32))
    assert feats.shape[2:] == (pred.sizes["lat"], pred.sizes["lon"])


def test_collect_and_analyse_end_to_end(
    config: ExperimentConfig, store: Path, tmp_path: Path
) -> None:
    import json

    train_run(config, store, "run1", device_choice="cpu", epochs=1)
    ckpt = Path(config.paths.checkpoints) / "run1" / "best.pt"
    ds = xr.open_zarr(store)
    lat, lon = ds["lat"].values, ds["lon"].values
    days = pd.date_range(*config.splits.val)
    rng = np.random.default_rng(3)
    rows = []
    for f in range(8):
        for c in range(4):
            i, j = rng.integers(2, len(lat) - 2), rng.integers(2, len(lon) - 2)
            for z in (5.0, 60.0, 120.0):
                rows.append((1900000 + f, c + 1, days[rng.integers(days.size)], lat[i], lon[j], z,
                             20.0 + rng.standard_normal()))  # fmt: skip
    df = pd.DataFrame(rows, columns=["pn", "cy", "t", "la", "lo", "p", "temp"])
    argo = xr.Dataset({
        "platform_number": ("row", df["pn"].to_numpy()), "cycle_number": ("row", df["cy"]),
        "time": ("row", df["t"].to_numpy()), "latitude": ("row", df["la"].to_numpy(float)),
        "longitude": ("row", df["lo"].to_numpy(float)), "pres": ("row", df["p"].to_numpy()),
        "temp": ("row", df["temp"].to_numpy()), "temp_qc": ("row", np.ones(len(df), "int8")),
        "pres_qc": ("row", np.ones(len(df), "int8")),
        "position_qc": ("row", np.ones(len(df), "int8")),
    })  # fmt: skip
    argo_path = tmp_path / "argo.nc"
    argo.to_netcdf(argo_path)
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"held_out_floats": [str(1900000 + f) for f in range(8)]}))

    out = un.collect(config, ckpt, [store], [argo_path], holdout, "val", out_dir=tmp_path / "o",
                     device_choice="cpu")  # fmt: skip
    table = pd.read_parquet(out)
    assert len(table) > 0
    assert len(un.feature_columns(table)) == int(config.train.get("base_channels", 32))
    assert set(table["platform_number"]) <= {str(1900000 + f) for f in range(8)}
    tables = un.analyse(table, {"again": table}, tmp_path / "o", niter=20)
    assert set(tables) == {"val_halves", "again"}
    assert (tmp_path / "o" / "metrics" / "uncertainty_coverage_again.csv").exists()


def test_sd_cap_and_scores() -> None:
    fit = un.fit_heads(_synthetic(40, 1), niter=30, n_dirs=2, clip=True, sd_cap=1.5)
    scored = un.predictive(_synthetic(40, 2), fit)
    # the full posterior adds the spread of its mean draws on top of the capped noise
    assert (scored["bll_bam_sd"] <= 1.6 * scored["const_sd"]).all()
    assert (scored["bll_plugin_sd"] <= 1.5 * scored["const_sd"] * 1.001).all()
    table = un.coverage_table(scored).set_index(["group", "method"])
    assert table.loc[("all", "bll_bam"), "frac_blowup"] == 0.0
    allrows = table.loc["all"]
    assert np.isfinite(allrows["crps_c"]).all() and np.isfinite(allrows["nll"]).all()


def test_crps_closed_forms_agree() -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(50)
    m, sd = rng.standard_normal(50), np.exp(rng.standard_normal(50) * 0.3)
    one = un.gaussian_crps(obs, m, sd)
    mix = un.mixture_crps(obs, np.repeat(m[:, None], 5, 1), np.repeat(sd[:, None], 5, 1))
    np.testing.assert_allclose(mix, one, rtol=1e-10)
    draws = m[:, None] + sd[:, None] * rng.standard_normal((50, 20000))
    mc = np.abs(draws - obs[:, None]).mean(1) - 0.5 * np.abs(draws[:, ::2] - draws[:, 1::2]).mean(1)
    np.testing.assert_allclose(one, mc, atol=0.03)


@pytest.mark.parametrize("centred", [True, False])
def test_hierarchical_score_matches_finite_differences(centred: bool) -> None:
    from oceanscope_ml.evaluation import uncertainty_hier as uh

    rng = np.random.default_rng(4)
    n, p, n_reg = 60, 3, 3
    x = np.column_stack([rng.standard_normal((n, p - 1)), np.ones(n)])
    r, g = rng.standard_normal(n), rng.integers(0, n_reg, n)
    log_post, score = uh.hier_log_post(x, r, g, n_reg, centred)
    theta = 0.3 * rng.standard_normal((2, uh.n_params(p, n_reg)))
    theta[:, -1] = [-1.0, -0.3]
    eps = 1e-6
    num = np.empty_like(theta)
    for j in range(theta.shape[1]):
        e = np.zeros(theta.shape[1])
        e[j] = eps
        num[:, j] = (log_post(theta + e) - log_post(theta - e)) / (2 * eps)
    np.testing.assert_allclose(score(theta), num, rtol=1e-5, atol=1e-5)


def test_folds_split_each_region_evenly() -> None:
    from oceanscope_ml.evaluation import uncertainty_hier as uh

    df = pd.DataFrame({"platform_number": [f"{k}" for k in range(15)],
                       "lat": [15.0] * 10 + [2.0] * 5,
                       "lon": [88.0] * 5 + [65.0] * 5 + [80.0] * 5})  # fmt: skip
    folds = uh.float_folds(df)
    for lo in (0, 5, 10):
        assert sorted(folds[str(k)] for k in range(lo, lo + 5)) == [0, 1, 2, 3, 4]
