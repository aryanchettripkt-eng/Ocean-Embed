"""Unit and integration tests for oceanscope_ml.models.baselines and ARMOR3D reduction.

Implements and verifies:
- PRD_AI_ML §4 Stage 3: oceanscope_ml.models.baselines
- PRD_AI_ML §5.6: Baselines (Climatology and Per-pixel Ridge)
- PRD_AI_ML §6.3: CSV contracts (baseline_climatology_val.csv, baseline_ridge_val.csv)
- PRD_AI_ML §3, §4.2: reduce_armor3d
- PRD_AI_ML §0: No test leakage guard
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.data.preprocess import (
    run_preprocessing,
)
from oceanscope_ml.data.reduce import TargetGrid, reduce_armor3d
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.models.baselines import (
    build_cli,
    evaluate_climatology_baseline,
    evaluate_ridge_baseline,
    extract_pixel_features_and_targets,
    fit_ridge_baseline,
    run_baselines,
)
from tests.synthetic_data import generate_synthetic_raw_sample


class MockConfig:
    """Mock configuration for testing baselines."""

    def __init__(self, raw_dir: Path, processed_dir: Path, outputs_dir: Path):
        self.project = "oceanscope-test"
        self.dataset_version = "v1"
        self.depths_m = [
            0.494,
            5.078,
            9.573,
            18.496,
            29.445,
            47.374,
            65.807,
            92.326,
            130.666,
            186.126,
            222.475,
            266.040,
            380.213,
            643.567,
            1062.440,
        ]

        class DummyRegion:
            lon_min = 60.0
            lon_max = 63.0
            lat_min = 10.0
            lat_max = 13.0

        class DummyGrid:
            resolution_deg = 0.25

        class DummySplits:
            train = ["2019-01-01", "2019-01-03"]
            val = ["2019-01-04", "2019-01-05"]
            test = ["2021-01-01", "2021-12-31"]
            gap_days = 0

        class DummyPaths:
            raw = raw_dir
            processed = processed_dir
            outputs = outputs_dir
            checkpoints = outputs_dir / "checkpoints"

        class DummyTrain:
            seed = 42

        class DummyBaselines:
            max_train_samples = 500_000
            ridge_alpha = 1.0

        self.region = DummyRegion()
        self.grid = DummyGrid()
        self.splits = DummySplits()
        self.paths = DummyPaths()
        self.train = DummyTrain()
        self.baselines = DummyBaselines()
        self.raw_dict = {
            "project": self.project,
            "dataset_version": self.dataset_version,
            "depths_m": self.depths_m,
        }


@pytest.fixture
def test_processed_store(tmp_path: Path) -> tuple[Path, MockConfig]:
    """Create a minimal synthetic processed Zarr store with 5 days."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    outputs_dir = tmp_path / "outputs"
    raw_dir.mkdir()
    processed_dir.mkdir()
    outputs_dir.mkdir()

    cfg = MockConfig(raw_dir, processed_dir, outputs_dir)
    target_grid = TargetGrid.from_bounds(
        lon_min=cfg.region.lon_min,
        lon_max=cfg.region.lon_max,
        lat_min=cfg.region.lat_min,
        lat_max=cfg.region.lat_max,
        resolution_deg=cfg.grid.resolution_deg,
    )

    # 5 consecutive days: 2019-01-01 through 2019-01-05
    dates = pd.date_range("2019-01-01", periods=5, freq="D")
    generate_synthetic_raw_sample(
        raw_dir,
        target_grid,
        cfg.depths_m,
        dates=dates,
    )

    _ = run_preprocessing(
        cfg,  # type: ignore[arg-type]
        raw_dir=raw_dir,
        out_dir=processed_dir,
    )

    store_path = processed_dir / f"oceanscope_{cfg.dataset_version}.zarr"
    return store_path, cfg


def test_reduce_armor3d() -> None:
    """Test reduce_armor3d regridding to 0.25° grid and 15 depths."""
    target_grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    depths_m = [5.0, 50.0, 100.0]

    src_lats = np.arange(9.5, 12.5, 0.125)
    src_lons = np.arange(59.5, 62.5, 0.125)
    src_depths = np.array([0.0, 10.0, 30.0, 60.0, 120.0], dtype=np.float64)
    time_vals = np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")

    t_data = np.full(
        (1, len(src_depths), len(src_lats), len(src_lons)),
        24.0,
        dtype=np.float32,
    )

    ds_raw = xr.Dataset(
        data_vars={"to": (("time", "depth", "lat", "lon"), t_data)},
        coords={
            "time": time_vals,
            "depth": src_depths,
            "lat": src_lats,
            "lon": src_lons,
        },
    )

    ds_red = reduce_armor3d(ds_raw, target_grid, depths_m)

    assert "armor3d" in ds_red.data_vars
    assert ds_red["armor3d"].shape == (1, 3, len(target_grid.lats), len(target_grid.lons))
    assert np.allclose(ds_red["armor3d"].values, 24.0)


def test_climatology_baseline(test_processed_store: tuple[Path, MockConfig]) -> None:
    """Test climatology baseline calculation and CSV schema."""
    store_path, cfg = test_processed_store
    ds = xr.open_zarr(store_path)

    val_dates = pd.date_range("2019-01-04", periods=2, freq="D")
    df_clim = evaluate_climatology_baseline(ds, val_dates, cfg.depths_m)

    # 15 depths + 1 overall row = 16 rows
    assert len(df_clim) == 16

    expected_cols = [
        "model",
        "split",
        "depth_m",
        "n",
        "mae_c",
        "rmse_c",
        "bias_c",
        "skill_vs_clim_pct",
    ]
    assert list(df_clim.columns) == expected_cols

    assert (df_clim["model"] == "climatology").all()
    assert (df_clim["split"] == "val").all()
    assert (df_clim["skill_vs_clim_pct"] == 0.0).all()

    # All error metrics must be finite non-negative
    assert (df_clim["mae_c"] >= 0.0).all()
    assert (df_clim["rmse_c"] >= 0.0).all()
    assert (df_clim["n"] > 0).all()

    # Last row is 'all'
    assert df_clim.iloc[-1]["depth_m"] == "all"


def test_ridge_baseline_fit_and_evaluate(
    test_processed_store: tuple[Path, MockConfig],
) -> None:
    """Test per-pixel Ridge feature extraction, fitting, and evaluation."""
    store_path, cfg = test_processed_store
    ds = xr.open_zarr(store_path)

    train_dates = pd.date_range("2019-01-01", periods=3, freq="D")
    val_dates = pd.date_range("2019-01-04", periods=2, freq="D")

    X_train, Y_train, feature_names = extract_pixel_features_and_targets(
        ds=ds,
        target_dates=train_dates,
        depths_m=cfg.depths_m,
        max_samples=1000,
    )

    assert len(X_train) > 0
    assert X_train.shape[1] == len(feature_names)
    assert Y_train.shape == (len(X_train), 15)

    W, mu_Y, mu_X, sigma_X = fit_ridge_baseline(X_train, Y_train, alpha=1.0)
    assert W.shape == (len(feature_names), 15)
    assert mu_Y.shape == (15,)
    assert mu_X.shape == (len(feature_names),)
    assert sigma_X.shape == (len(feature_names),)

    df_clim = evaluate_climatology_baseline(ds, val_dates, cfg.depths_m)
    df_ridge = evaluate_ridge_baseline(
        ds=ds,
        val_dates=val_dates,
        depths_m=cfg.depths_m,
        W=W,
        mu_Y=mu_Y,
        mu_X=mu_X,
        sigma_X=sigma_X,
        clim_df=df_clim,
    )

    assert len(df_ridge) == 16
    assert (df_ridge["model"] == "ridge").all()
    assert (df_ridge["split"] == "val").all()
    assert (df_ridge["mae_c"] >= 0.0).all()
    assert (df_ridge["rmse_c"] >= 0.0).all()


def test_run_baselines_end_to_end(
    test_processed_store: tuple[Path, MockConfig], tmp_path: Path
) -> None:
    """Test full baseline runner pipeline and file artifacts."""
    store_path, cfg = test_processed_store
    metrics_dir = tmp_path / "out_metrics"
    models_dir = tmp_path / "out_models"

    df_clim, df_ridge = run_baselines(
        config=cfg,  # type: ignore[arg-type]
        store_path=store_path,
        out_dir=metrics_dir,
        models_dir=models_dir,
        max_train_samples=2000,
        ridge_alpha=1.0,
    )

    clim_csv = metrics_dir / "baseline_climatology_val.csv"
    ridge_csv = metrics_dir / "baseline_ridge_val.csv"
    model_npz = models_dir / "baseline_ridge.npz"

    assert clim_csv.exists()
    assert ridge_csv.exists()
    assert model_npz.exists()

    df_c = pd.read_csv(clim_csv)
    df_r = pd.read_csv(ridge_csv)

    assert len(df_c) == 16
    assert len(df_r) == 16
    assert list(df_c.columns) == list(df_r.columns)
    # PRD_AI_ML §5.6 & REVIEW_P3 A7: Climatology and Ridge MUST evaluate on exact same cells
    assert (df_c["n"] == df_r["n"]).all()


def test_baselines_missing_dates_raises_no_fallback(
    test_processed_store: tuple[Path, MockConfig],
) -> None:
    """Test that missing train or val dates raise ValueError rather than falling back to 70/30."""
    store_path, cfg = test_processed_store

    # Point train split to dates not in store (store has 2019-01-01..05)
    cfg.splits.train = ["2018-01-01", "2018-01-05"]
    with pytest.raises(ValueError, match="No training dates found"):
        run_baselines(config=cfg, store_path=store_path)  # type: ignore[arg-type]

    # Point val split to dates not in store
    cfg.splits.train = ["2019-01-01", "2019-01-03"]
    cfg.splits.val = ["2018-01-01", "2018-01-05"]
    with pytest.raises(ValueError, match="No validation dates found"):
        run_baselines(config=cfg, store_path=store_path)  # type: ignore[arg-type]


def test_baselines_empty_depth_records_nan(
    test_processed_store: tuple[Path, MockConfig],
) -> None:
    """Test that depths with zero valid cells record NaN rather than fabricated 0.0."""
    store_path, cfg = test_processed_store
    ds = xr.open_zarr(store_path)

    # Invalidate all cells at depth index 2 (9.573 m)
    ds_mod = ds.copy(deep=True)
    target_v = ds_mod["target_valid"].values.copy()
    target_v[:, 2, :, :] = False
    ds_mod["target_valid"].values = target_v

    val_dates = pd.date_range("2019-01-04", periods=2, freq="D")
    df_clim = evaluate_climatology_baseline(ds_mod, val_dates, cfg.depths_m)

    row_depth2 = df_clim[df_clim["depth_m"] == cfg.depths_m[2]].iloc[0]
    # Because target_valid is False at depth 2, n for that depth must be 0 and rmse_c must be NaN
    assert row_depth2["n"] == 0
    assert np.isnan(row_depth2["rmse_c"])
    assert np.isnan(row_depth2["mae_c"])
    assert np.isnan(row_depth2["bias_c"])


def test_no_test_leakage_in_baselines(
    test_processed_store: tuple[Path, MockConfig],
) -> None:
    """Test that evaluating baselines on test dates raises TestLeakageError."""
    store_path, cfg = test_processed_store

    # Deliberately violate sealed test split
    cfg.splits.val = ["2021-05-01", "2021-06-01"]

    with pytest.raises(TestLeakageError):
        run_baselines(
            config=cfg,  # type: ignore[arg-type]
            store_path=store_path,
        )


def test_baselines_cli() -> None:
    """Test CLI argument parser."""
    parser = build_cli()
    args = parser.parse_args(["--max-samples", "10000", "--alpha", "2.5"])
    assert args.max_samples == 10000
    assert args.alpha == 2.5
