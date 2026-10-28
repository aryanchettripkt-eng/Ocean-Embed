"""ARGO validation on synthetic data only (PRD_AI_ML §1.1, §7). No real float data is used."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.evaluation import argo_validation as av

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
DAYS = pd.date_range("2020-01-01", periods=3, freq="D")
LAT = np.arange(5.0, 8.01, 0.25)
LON = np.arange(78.0, 82.01, 0.25)
PRES_LEVELS = [10.0, 40.0, 60.0, 120.0, 200.0, 280.0, 400.0, 900.0]
# Constant offsets (°C) of each product from the truth. The truth is linear in depth.
MODEL_BETTER = {"model": 0.1, "glorys12": 0.3, "grep": 0.4, "armor3d": 0.5}
MODEL_WORSE = {"model": 0.6, "glorys12": 0.1, "grep": 0.1, "armor3d": 0.1}


def _truth(z: np.ndarray | float) -> np.ndarray | float:
    return 28.0 - 0.02 * z


def _datasets(
    depths: np.ndarray,
    offsets: dict[str, float],
    days: pd.DatetimeIndex = DAYS,
    grep_missing: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(0)
    shape = (days.size, depths.size, LAT.size, LON.size)
    base = np.broadcast_to(np.asarray(_truth(depths))[None, :, None, None], shape)

    def field(name: str) -> np.ndarray:
        return base + offsets[name] + rng.normal(0.0, 0.05, shape)

    dims = ("time", "depth", "lat", "lon")
    coords = {"time": days, "depth": depths, "lat": LAT, "lon": LON}
    pred = xr.Dataset(
        {
            "temp_pred": (dims, field("model").astype("float32")),
            "temp_glorys": (dims, field("glorys12").astype("float32")),
            "coverage": (
                ("time", "lat", "lon"),
                np.ones((days.size, LAT.size, LON.size), dtype="int8"),
            ),
        },
        coords=coords,
        attrs={"run_id": "test_run"},
    )
    grep = np.full(shape, np.nan) if grep_missing else field("grep")
    refs = xr.Dataset(
        {"grep_mean": (dims, grep), "armor3d": (dims, field("armor3d"))}, coords=coords
    )
    return pred, refs


def _profiles(n_floats: int = 12, per_float: int = 4, first_id: int = 1900000) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for f in range(n_floats):
        pid = str(first_id + f)
        for c in range(per_float):
            day = DAYS[rng.integers(DAYS.size)]
            lat = LAT[rng.integers(LAT.size)] + rng.uniform(-0.1, 0.1)
            lon = LON[rng.integers(LON.size)] + rng.uniform(-0.1, 0.1)
            hour = float(rng.uniform(0.0, 23.0))
            for z in PRES_LEVELS:
                rows.append(
                    {
                        "platform_number": pid,
                        "cycle": c + 1,
                        "time": day + pd.Timedelta(hours=hour),
                        "lat": lat,
                        "lon": lon,
                        "pres": z,
                        "temp_obs_c": float(_truth(z)),
                    }
                )
    return pd.DataFrame(rows)


def _write_argo(path: Path, df: pd.DataFrame) -> Path:
    """An ERDDAP-like NetCDF table with the QC columns the validation requires."""
    ds = xr.Dataset(
        {
            "platform_number": ("row", df["platform_number"].astype("int64").to_numpy()),
            "cycle_number": ("row", df["cycle"].to_numpy().astype("int32")),
            "time": ("row", pd.to_datetime(df["time"]).to_numpy()),
            "latitude": ("row", df["lat"].to_numpy(float)),
            "longitude": ("row", df["lon"].to_numpy(float)),
            "pres": ("row", df["pres"].to_numpy(float)),
            "temp": ("row", df["temp"].to_numpy(float)),
            "temp_qc": ("row", df["temp_qc"].to_numpy().astype("int8")),
            "pres_qc": ("row", df["pres_qc"].to_numpy().astype("int8")),
            "position_qc": ("row", df["position_qc"].to_numpy().astype("int8")),
        }
    )
    ds.to_netcdf(path)
    return path


def _with_qc(df: pd.DataFrame, qc: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["temp"] = out["temp_obs_c"]
    for col in ("temp_qc", "pres_qc", "position_qc"):
        out[col] = qc
    return out


@pytest.fixture
def depths() -> np.ndarray:
    return np.asarray(load_config(CONFIG_PATH).depths_m, dtype=float)


def test_interp_depth_is_linear_and_refuses_extrapolation() -> None:
    depths = np.array([0.0, 10.0, 20.0])
    levels = np.array([[0.0, 10.0, 20.0], [1.0, np.nan, 3.0]])
    assert av._interp_depth(levels[:1], depths, np.array([5.0]))[0] == pytest.approx(5.0)
    assert np.isnan(av._interp_depth(levels[1:], depths, np.array([15.0]))[0])
    assert np.isnan(av._interp_depth(levels[:1], depths, np.array([25.0]))[0])


def test_nearest_cell_matches_within_half_a_cell() -> None:
    _, inside = av._nearest_cell(LAT, np.array([5.1, 4.8, 8.1, 8.2]))
    assert inside.tolist() == [True, False, True, False]


def test_headline_met_when_model_beats_every_reference(depths: np.ndarray) -> None:
    pred, refs = _datasets(depths, MODEL_BETTER)
    result = av.validate(_profiles(), pred, refs)
    headline = result.headline
    assert av.headline_verdict(headline, "val") is True
    comp = headline[(headline["period"] == "val") & (headline["product"] != "model")]
    assert len(comp) == 3 and (comp["diff_ci_high_c"] < 0).all()
    assert (headline.loc[headline["period"] == "test", "n_floats"] == 0).all()


def test_headline_not_met_when_a_reference_beats_the_model(depths: np.ndarray) -> None:
    pred, refs = _datasets(depths, MODEL_WORSE)
    headline = av.validate(_profiles(), pred, refs).headline
    assert av.headline_verdict(headline, "val") is False


def test_no_competitor_data_means_no_verdict(depths: np.ndarray) -> None:
    pred, refs = _datasets(depths, MODEL_BETTER, grep_missing=True)
    headline = av.validate(_profiles(), pred, refs).headline
    val = headline[headline["period"] == "val"]
    assert (val["n_floats"] == 0).all() and val["rmse_c"].isna().all()
    assert av.headline_verdict(headline, "val") is None


def test_headline_uses_bay_of_bengal_levels_in_50_to_300_m(depths: np.ndarray) -> None:
    pred, refs = _datasets(depths, MODEL_BETTER)
    result = av.validate(_profiles(), pred, refs)
    m = result.matches
    band = m["depth_m"].between(av.HEADLINE_DEPTH_MIN_M, av.HEADLINE_DEPTH_MAX_M)
    bob = (m["lon"] >= 78.0) & (m["lat"] >= 5.0)
    model_row = result.headline[
        (result.headline["period"] == "val") & (result.headline["product"] == "model")
    ]
    assert int(model_row["n_obs"].iloc[0]) == int((band & bob).sum())
    # 40 dbar is nearer the 47.4 m level, which is outside the band.
    assert set(m.loc[np.isclose(m["temp_obs_c"], _truth(40.0)), "depth_m"]) == {depths[5]}


def test_bootstrap_is_reproducible_and_brackets_the_point_estimate(depths: np.ndarray) -> None:
    pred, refs = _datasets(depths, MODEL_BETTER)
    first = av.validate(_profiles(), pred, refs).headline
    second = av.validate(_profiles(), pred, refs).headline
    pd.testing.assert_frame_equal(first, second)
    model = first[(first["period"] == "val") & (first["product"] == "model")].iloc[0]
    assert model["ci_low_c"] <= model["rmse_c"] <= model["ci_high_c"]


def test_run_writes_contract_files_and_ignores_unlisted_and_sealed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depths: np.ndarray
) -> None:
    config = load_config(CONFIG_PATH)
    pred, refs = _datasets(depths, MODEL_BETTER)
    pred_path = tmp_path / "predictions_test_run.zarr"
    pred.to_zarr(pred_path)
    monkeypatch.setattr(av, "open_parts", lambda store_paths, stats_path=None: refs)

    clean = _with_qc(_profiles())
    holdout = [str(1900000 + f) for f in range(12)]
    extra = pd.concat(
        [
            # A float outside the hold-out with a 10 °C error must not change the headline.
            _with_qc(_profiles(n_floats=1, first_id=1999999)).assign(temp=lambda d: d.temp + 10),
            # Flag 4 rows must not count.
            _with_qc(_profiles(n_floats=1, first_id=1900000), qc=4).assign(
                temp=lambda d: d.temp + 10
            ),
            # Profiles in 2021 are outside the validation window and are never used here.
            _with_qc(_profiles(n_floats=1, first_id=1900000)).assign(
                time=lambda d: d.time + pd.Timedelta(days=365 * 3), temp=lambda d: d.temp + 10
            ),
        ],
        ignore_index=True,
    )
    argo_path = _write_argo(tmp_path / "argo.nc", pd.concat([clean, extra], ignore_index=True))
    holdout_path = tmp_path / "argo_eval_floats.json"
    holdout_path.write_text(json.dumps({"held_out_floats": holdout}), encoding="utf-8")

    paths = av.run(
        config,
        pred_path,
        [tmp_path / "store.zarr"],
        [argo_path],
        holdout_path,
        out_dir=tmp_path / "out",
    )

    headline = pd.read_csv(paths["headline"])
    assert list(headline.columns) == av.HEADLINE_COLUMNS
    assert av.headline_verdict(headline, "val") is True
    # Same numbers as validating the hold-out profiles alone: the extra rows changed nothing.
    expected = av.validate(clean, pred, refs).headline
    pd.testing.assert_frame_equal(
        headline.reset_index(drop=True), expected.round(6).reset_index(drop=True), check_dtype=False
    )
    per_depth = pd.read_csv(paths["argo"])
    assert list(per_depth.columns) == av.DEPTH_COLUMNS
    assert len(per_depth) == len(config.depths_m)
    matches = pd.read_parquet(paths["argo_matches"])
    assert list(matches.columns) == av.MATCH_COLUMNS
    assert set(matches["platform_number"]) <= set(holdout)


def test_run_refuses_test_period_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, depths: np.ndarray
) -> None:
    config = load_config(CONFIG_PATH)
    days = DAYS.append(pd.DatetimeIndex(["2021-01-01"]))
    pred, refs = _datasets(depths, MODEL_BETTER, days=days)
    pred_path = tmp_path / "predictions_test_run.zarr"
    pred.to_zarr(pred_path)
    monkeypatch.setattr(av, "open_parts", lambda store_paths, stats_path=None: refs)
    argo_path = _write_argo(tmp_path / "argo.nc", _with_qc(_profiles()))
    holdout_path = tmp_path / "argo_eval_floats.json"
    holdout_path.write_text(json.dumps({"held_out_floats": ["1900000"]}), encoding="utf-8")
    with pytest.raises(TestLeakageError):
        av.run(config, pred_path, [tmp_path / "s.zarr"], [argo_path], holdout_path)
