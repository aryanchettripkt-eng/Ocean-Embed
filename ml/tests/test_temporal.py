"""PRD_AI_ML §5.8 temporal and physics checks, on the synthetic store (no torch needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.evaluation.temporal import (
    _Block,
    _count_physics,
    _Flicker,
    _flicker_inputs,
    _Physics,
    _Ridge,
    heat_content_weights,
    temporal_checks,
)
from oceanscope_ml.models.baselines import CANDIDATE_2D_VARS


def _block(pred: np.ndarray, glorys: np.ndarray, dates: pd.DatetimeIndex) -> _Block:
    shape = pred.shape
    return _Block(
        start=0,
        dates=dates,
        pred=pred.astype(np.float32),
        glorys=glorys.astype(np.float32),
        clim=np.zeros(shape, np.float32),
        valid=np.ones(shape, bool),
        flux=np.zeros((shape[0], shape[2], shape[3]), np.float32),
    )


def test_heat_weights_cover_the_0_to_300_m_column(config: ExperimentConfig) -> None:
    depths = np.asarray(config.depths_m, dtype=float)
    w = heat_content_weights(depths)
    assert np.all(w >= 0)
    assert w[depths > 300].sum() == 0
    assert w.sum() == pytest.approx(300.0)
    # A uniform 1 degC column holds rho * c_p * 300 m of heat.
    assert float((w * 1.0).sum()) * 1025.0 * 3990.0 == pytest.approx(1025.0 * 3990.0 * 300.0)


def test_flicker_flags_a_jumpy_prediction_and_not_a_smooth_one() -> None:
    rng = np.random.default_rng(0)
    days = pd.date_range("2019-01-01", periods=40, freq="D")
    smooth = np.cumsum(rng.normal(0, 0.05, size=(40, 1, 3, 3)), axis=0)
    jumpy = smooth + np.where(np.arange(40)[:, None, None, None] % 2 == 0, 0.5, -0.5)
    for pred, flagged in ((smooth, False), (jumpy, True)):
        blk = _block(pred, smooth, days)
        acc = _Flicker(1)
        inputs = _flicker_inputs(blk, blk.pred, first_new=0)
        assert inputs is not None
        acc.add(*inputs)
        row = acc.rows([0.5], "m")[0]
        assert row["flagged"] is flagged
        assert row["n"] == 39 * 9


def test_flicker_skips_pairs_across_a_gap() -> None:
    days = pd.DatetimeIndex(["2019-01-01", "2019-01-02", "2019-01-05"])
    x = np.zeros((3, 1, 1, 1), np.float32)
    blk = _block(x, x, days)
    acc = _Flicker(1)
    inputs = _flicker_inputs(blk, blk.pred, first_new=0)
    assert inputs is not None
    acc.add(*inputs)
    assert acc.rows([0.5], "m")[0]["n"] == 1  # only 01-01 -> 01-02 is consecutive


def test_inversion_and_outlier_counts() -> None:
    days = pd.date_range("2019-01-01", periods=2, freq="D")
    glorys = np.zeros((2, 3, 1, 1), np.float32)
    glorys[:, :, 0, 0] = [[20.0, 19.0, 18.0]] * 2  # cooling with depth: no inversion
    pred = glorys.copy()
    pred[0, :, 0, 0] = [20.0, 19.0, 19.8]  # warm layer under cold by 0.8 degC: inversion
    pred[1, :, 0, 0] = [20.0, 19.0, 17.0]  # outlier 1 degC is not > 3 degC
    pred[1, 2, 0, 0] = 14.0  # deepest level 4 degC off GLORYS: gross outlier
    blk = _block(pred, glorys, days)
    phys = _Physics()
    _count_physics(blk, 0, phys)
    assert phys.profiles == 2
    assert phys.pred_inv == 1
    assert phys.glorys_inv == 0
    assert phys.pred_inv_not_glorys == 1
    assert phys.outliers == 1


def _write_predictions(
    path: Path,
    config: ExperimentConfig,
    store: Path,
    offset: float,
    extra_end: str | None = None,
    start: str | None = None,
    stride: int = 1,
) -> Path:
    parts = open_parts([store])
    end = extra_end or config.splits.val[1]
    sel = parts.sel(time=slice(pd.Timestamp(start or config.splits.train[0]), pd.Timestamp(end)))
    sel = sel.isel(time=slice(None, None, stride))
    truth = np.asarray(sel["thetao"].values, dtype=np.float32)
    ds = xr.Dataset(
        {
            "temp_pred": (("time", "depth", "lat", "lon"), (truth + offset).astype(np.float16)),
            "temp_glorys": (("time", "depth", "lat", "lon"), truth.astype(np.float16)),
            "coverage": (
                ("time", "lat", "lon"),
                np.ones((truth.shape[0],) + truth.shape[2:], np.int8),
            ),  # noqa: E501
        },
        coords={
            "time": sel["time"].values,
            "depth": np.asarray(config.depths_m, dtype=np.float64),
            "lat": sel["lat"].values,
            "lon": sel["lon"].values,
        },
        attrs={"run_id": "syn1", "model_name": "unet"},
    )
    ds.to_zarr(path, mode="w", zarr_format=2)
    return path


def test_temporal_checks_end_to_end(config: ExperimentConfig, store: Path, tmp_path: Path) -> None:
    pred_path = _write_predictions(tmp_path / "pred.zarr", config, store, offset=0.3)
    out = tmp_path / "outputs"
    paths = temporal_checks(config, pred_path, store, out_dir=out)

    temporal = pd.read_csv(paths["temporal"])
    assert set(temporal["model"]) == {"unet", "climatology"}
    assert len(temporal) == 2 * (len(config.depths_m) + 1)
    assert "all" in set(temporal["depth_m"].astype(str))

    bias = pd.read_csv(paths["bias_by_year"])
    assert set(bias["split"]) == {"train", "val"}
    unet = bias[bias["model"] == "unet"]
    # The model is 0.3 degC warm everywhere, up to float16 rounding.
    assert np.allclose(unet["bias_c"].to_numpy(), 0.3, atol=0.02)
    trend = pd.read_csv(paths["bias_trend"])
    assert trend["n_years"].max() == 1 and trend["slope_c_per_yr"].isna().all()

    physics = pd.read_csv(paths["physics"])
    assert {"heat_content_monthly", "inversion_gt_0p5c", "gross_outlier_gt_3c"} <= set(
        physics["check"]
    )
    outlier = physics[physics["metric"] == "frac_cells_abs_diff_gt_3c"]["value"].iloc[0]
    assert outlier == 0.0  # a 0.3 degC offset is far from the 3 degC threshold

    png = paths["figure"].read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_temporal_checks_refuses_test_dates(
    config: ExperimentConfig, store: Path, tmp_path: Path
) -> None:
    pred_path = _write_predictions(
        tmp_path / "pred_test.zarr", config, store, offset=0.0, extra_end="2000-11-30"
    )
    with pytest.raises(TestLeakageError):
        temporal_checks(config, pred_path, store, out_dir=tmp_path / "out")


def _zero_ridge(path: Path, store: Path, n_depths: int, intercept: float = 0.0) -> Path:
    """A Ridge file with zero weights: its prediction is the climatology plus ``intercept``."""
    ds = open_parts([store])
    n_features = (
        len([v for v in CANDIDATE_2D_VARS if v in ds.data_vars])
        + int("bathymetry" in ds.data_vars)
        + int("woa_clim_profile" in ds.data_vars)
        + 4
    )
    np.savez(
        path,
        weights=np.zeros((n_features, n_depths), np.float32),
        intercept=np.full(n_depths, intercept, np.float32),
        mu_X=np.zeros(n_features, np.float32),
        sigma_X=np.ones(n_features, np.float32),
    )
    return path


def test_ridge_anomaly_is_its_intercept_on_valid_cells(
    store: Path, config: ExperimentConfig
) -> None:
    ds = open_parts([store])
    n_depths = len(config.depths_m)
    ridge = _Ridge.load(_zero_ridge(store.parent / "ridge0.npz", store, n_depths, 0.25), ds,
                        [float(d) for d in config.depths_m])  # fmt: skip
    day = pd.Timestamp(config.splits.val[0])
    k = int(np.flatnonzero(pd.to_datetime(ds["time"].values) == day)[0])
    anom = ridge.anomaly(ds, k, int(day.day_of_year))
    assert anom.shape == (n_depths, ds.sizes["lat"], ds.sizes["lon"])
    finite = np.isfinite(anom)
    assert finite.any()
    assert np.allclose(anom[finite], 0.25)


def test_ridge_with_zero_weights_matches_climatology(
    config: ExperimentConfig, store: Path, tmp_path: Path
) -> None:
    pred_path = _write_predictions(tmp_path / "pred_r.zarr", config, store, offset=0.3)
    ridge_path = _zero_ridge(tmp_path / "ridge_zero.npz", store, len(config.depths_m))
    paths = temporal_checks(
        config, pred_path, store, out_dir=tmp_path / "out_r", ridge_path=ridge_path
    )
    temporal = pd.read_csv(paths["temporal"])
    assert set(temporal["model"]) == {"unet", "climatology", "ridge"}
    clim = temporal[temporal["model"] == "climatology"].reset_index(drop=True)
    ridge = temporal[temporal["model"] == "ridge"].reset_index(drop=True)
    assert np.allclose(ridge["tend_rmse_pred_c"], clim["tend_rmse_pred_c"], equal_nan=True)
    assert np.allclose(ridge["lag1_autocorr_pred"], clim["lag1_autocorr_pred"], equal_nan=True)
    bias = pd.read_csv(paths["bias_by_year"])
    assert "ridge" in set(bias["model"])


def test_train_sample_adds_train_years_to_the_bias_check(
    config: ExperimentConfig, store: Path, tmp_path: Path
) -> None:
    val_path = _write_predictions(tmp_path / "pred_v.zarr", config, store, offset=0.3)
    train_path = _write_predictions(
        tmp_path / "pred_train.zarr", config, store, offset=0.3,
        extra_end=config.splits.train[1], start=config.splits.train[0], stride=7,
    )  # fmt: skip
    paths = temporal_checks(
        config, val_path, store, out_dir=tmp_path / "out_t", train_predictions=train_path
    )
    bias = pd.read_csv(paths["bias_by_year"])
    unet = bias[bias["model"] == "unet"]
    assert {"train", "val"} <= set(unet["split"])
    train_rows = unet[unet["split"] == "train"]
    assert np.allclose(train_rows["bias_c"].to_numpy(), 0.3, atol=0.02)
    # The day-sampled store adds no flicker: its days are not consecutive.
    temporal = pd.read_csv(paths["temporal"])
    assert (
        int(temporal[(temporal["model"] == "unet") & (temporal["depth_m"] == "all")]["n"].iloc[0])
        > 0
    )
    physics = pd.read_csv(paths["physics"])
    assert physics["value"].notna().any()


def test_train_sample_from_another_run_is_refused(
    config: ExperimentConfig, store: Path, tmp_path: Path
) -> None:
    val_path = _write_predictions(tmp_path / "pred_v2.zarr", config, store, offset=0.0)
    train_path = _write_predictions(
        tmp_path / "pred_t2.zarr", config, store, offset=0.0,
        extra_end=config.splits.train[1], start=config.splits.train[0], stride=7,
    )  # fmt: skip
    xr.open_zarr(train_path).load().assign_attrs(run_id="other").to_zarr(
        tmp_path / "pred_other.zarr", mode="w", zarr_format=2
    )
    with pytest.raises(ValueError, match="is from run"):
        temporal_checks(
            config,
            val_path,
            store,
            out_dir=tmp_path / "out_t2",
            train_predictions=tmp_path / "pred_other.zarr",
        )
