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
    heat_content_weights,
    temporal_checks,
)


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
    path: Path, config: ExperimentConfig, store: Path, offset: float, extra_end: str | None = None
) -> Path:
    parts = open_parts([store])
    end = extra_end or config.splits.val[1]
    sel = parts.sel(time=slice(pd.Timestamp(config.splits.train[0]), pd.Timestamp(end)))
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
