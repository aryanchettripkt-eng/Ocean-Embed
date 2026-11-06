"""The one-shot 2021 ARGO test: saved corrections applied unchanged, scored once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.evaluation import argo_test as at
from oceanscope_ml.inference import argo_correct as ac
from oceanscope_ml.inference import residual_model as rm
from tests.test_argo_correct import BIAS_C, DAYS, LAT, LON, _profiles, _store, _truth

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
TEST_DAYS = pd.date_range("2021-03-01", periods=20, freq="D")
SHIFT = TEST_DAYS[0] - DAYS[0]


def _fit_on_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: set[str]) -> tuple:
    config = load_config(CONFIG_PATH)
    path, depths = _store(tmp_path / "val")
    fit = _profiles(depths, 40, 2900000, seed=1)
    for mod in (rm, ac):
        monkeypatch.setattr(mod, "read_levels", lambda *a: pd.DataFrame())
        monkeypatch.setattr(mod, "profile_levels", lambda levels, depths_m, exclude: fit)
    monkeypatch.setattr(rm, "PARAMS", {**rm.PARAMS, "min_data_in_leaf": 5, "num_threads": 1})
    monkeypatch.setattr(rm, "MAX_ROUNDS", 200)
    out = tmp_path / "val_out"
    tree = rm.run(config, path, [], held, "ens3w_rm2", out_dir=out, features="v2",
                  max_depth_m=600.0)  # fmt: skip
    params = ac.OIParams(length_deg=1.0, tau_days=5.0, window_days=10, noise=0.5)
    ac.run(config, tree, [], held, "ens3w_rm2_oi", out_dir=out, params=params, min_depth_m=1.0,
           max_depth_m=600.0, outlier_sigma=4.0)  # fmt: skip
    return (config, depths, out / "models" / "residual_model_ens3w_rm2.txt",
            out / "models" / "argo_correct_ens3w_rm2_oi.json")  # fmt: skip


def test_saved_settings_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, depths, model_path, oi_path = _fit_on_val(tmp_path, monkeypatch, set())
    model, meta = rm.load_saved(model_path)
    assert meta["features"] == "v2" and meta["max_depth_m"] == 600.0
    assert model.num_feature() == len(meta["names"])
    plan, oi = ac.load_settings(oi_path)
    assert len(plan) == depths.size and oi["outlier_sigma"] == 4.0
    assert plan[0] is None  # 0.5 m: min-depth guard
    assert all(p is None for p, d in zip(plan, depths, strict=True) if d > 600)
    assert plan[8] == ac.OIParams(1.0, 5.0, 10, 0.5)


def _test_member(tmp_path: Path, name: str, offset: float) -> Path:
    path, _ = _store(tmp_path / name, days=TEST_DAYS)
    ds = xr.open_zarr(path).load()
    ds["temp_pred"] = ds["temp_pred"] + np.float32(offset)
    ds.attrs["run_id"] = f"{name}_test"
    dest = tmp_path / name / f"predictions_{name}_test.zarr"
    ds.to_zarr(dest, mode="w", zarr_format=2, consolidated=True)
    return dest


def _held_obs(depths: np.ndarray, floats: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for f in floats:
        for c in range(4):
            day = TEST_DAYS[rng.integers(TEST_DAYS.size)]
            lat, lon = float(rng.uniform(LAT[2], LAT[-3])), float(rng.uniform(LON[2], LON[-3]))
            for d, t in zip(depths, _truth(depths), strict=True):
                rows.append({"platform_number": f, "cycle": c + 1, "time": day, "lat": lat,
                             "lon": lon, "pres": float(d), "temp_obs_c": float(t)})  # fmt: skip
    return pd.DataFrame(rows)


def test_run_argo_test_applies_the_saved_corrections_and_scores_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = ["5900001", "5900002", "5900003", "5900004"]
    config, depths, model_path, oi_path = _fit_on_val(tmp_path, monkeypatch, set(held))
    holdout = tmp_path / "holdout.json"
    holdout.write_text('{"held_out_floats": ' + str(held).replace("'", '"') + "}")
    offsets = {"a": -0.2, "b": 0.0, "c": 0.2}

    def fake_predict(config, store_paths, ckpt, **kw):  # noqa: ANN001, ANN003, ANN202
        assert kw["split"] == "test" and kw["allow_test"] and kw["holdout"] == set(held)
        name = kw["run_id"][: -len("_test")]
        return _test_member(tmp_path / "members", name, offsets[name])

    inputs = _profiles(depths, 40, 2900000, seed=3)
    inputs["day"] = inputs["day"] + SHIFT
    seen: list[set[str]] = []

    def fake_profile_levels(levels, depths_m, exclude):  # noqa: ANN001, ANN202
        seen.append(set(exclude))
        return inputs

    shape = (TEST_DAYS.size, depths.size, LAT.size, LON.size)
    dims = ("time", "depth", "lat", "lon")
    refs = xr.Dataset(
        {"grep_mean": (dims, np.full(shape, np.nan)),
         "armor3d": (dims, np.broadcast_to(_truth(depths)[None, :, None, None] + 0.5, shape))},
        coords={"time": TEST_DAYS, "depth": depths, "lat": LAT, "lon": LON},
    )  # fmt: skip
    monkeypatch.setattr(at, "predict_store", fake_predict)
    monkeypatch.setattr(at, "read_levels", lambda *a: pd.DataFrame())
    monkeypatch.setattr(at, "profile_levels", fake_profile_levels)
    monkeypatch.setattr(at, "load_profiles", lambda *a: _held_obs(depths, held))
    monkeypatch.setattr(at, "open_parts", lambda paths, stats=None: refs)
    monkeypatch.setattr(rm, "open_parts", lambda paths, stats=None: refs)

    members = {k: tmp_path / f"{k}.pt" for k in offsets}
    kw = dict(go=True, out_dir=tmp_path / "out")
    with pytest.raises(PermissionError):
        at.run_argo_test(config, members, [0.2, 0.3, 0.5], "ens3", model_path, oi_path, [],
                         [], holdout, go=False, out_dir=tmp_path / "out")  # fmt: skip
    summary = at.run_argo_test(config, members, [0.2, 0.3, 0.5], "ens3", model_path, oi_path,
                               [], [], holdout, **kw)  # fmt: skip
    assert seen and all(s == set(held) for s in seen)
    assert summary["run_id"] == "ens3w_rm2_oi"
    by_label = {r["label"]: r for r in summary["results"]}
    assert set(by_label) == {"ens3w_rm2_oi_test", "ens3_test"}
    assert by_label["ens3_test"]["references"] == ["glorys12", "armor3d"]

    m = tmp_path / "out" / "metrics"
    final = pd.read_csv(m / "ens3w_rm2_oi_test_headline.csv").set_index("product")
    plain = pd.read_csv(m / "ens3_test_headline.csv").set_index("product")
    assert (final["period"] == "test").all()
    assert np.isnan(final.loc["grep", "rmse_c"])
    assert plain.loc["model", "rmse_c"] == pytest.approx(BIAS_C, abs=0.02)
    assert final.loc["model", "rmse_c"] < 0.5 * plain.loc["model", "rmse_c"]
    assert final.loc["model", "n_obs"] == plain.loc["model", "n_obs"] > 0
    stores = tmp_path / "out" / "argo_test" / "predictions"
    out = xr.open_zarr(stores / "predictions_ens3w_rm2_oi_test.zarr")
    before = xr.open_zarr(stores / "predictions_ens3w_test.zarr")
    deep = depths > 600
    np.testing.assert_allclose(
        out["temp_pred"].values[:, deep], before["temp_pred"].values[:, deep]
    )

    with pytest.raises(FileExistsError):
        at.run_argo_test(config, members, [0.2, 0.3, 0.5], "ens3", model_path, oi_path, [], [],
                         holdout, **kw)  # fmt: skip


def test_reference_with_partial_cover_is_left_out() -> None:
    n = 100
    band = pd.DataFrame({"temp_glorys_c": np.ones(n), "temp_armor3d_c": np.ones(n),
                         "temp_grep_c": [1.0] * 50 + [np.nan] * 50})  # fmt: skip
    assert at.available_references(band) == ["glorys12", "armor3d"]
    head = pd.DataFrame({"product": ["model", "glorys12", "armor3d"], "rmse_c": [0.5, 0.6, 0.7],
                         "diff_ci_high_c": [0.0, -0.01, 0.02]})  # fmt: skip
    assert at.verdict(head, ["glorys12"]) is True
    assert at.verdict(head, ["glorys12", "armor3d"]) is False
    assert at.verdict(head, []) is None
