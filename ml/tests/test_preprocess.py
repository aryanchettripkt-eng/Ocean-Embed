"""Tests for preprocessing, reduce functions, climatology, and transforms.

Implements required tests from PRD_AI_ML §8:
- Test 3: test_no_test_leakage
- Test 4: test_climatology
- Test 5: test_inverse_transform
- Reduce functions tests
- Processed store schema and figure creation tests
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.data.climatology import (
    fit_harmonic_climatology,
)
from oceanscope_ml.data.preprocess import run_preprocessing
from oceanscope_ml.data.reduce import (
    TargetGrid,
    area_mean_regrid_2d,
    compute_wind_stress_and_curl,
    reduce_gebco,
    reduce_glorys12,
    reduce_sst,
)
from oceanscope_ml.data.transform import (
    TestLeakageError,
    check_no_test_leakage,
    compute_norm_stats,
    denormalize_inputs,
    denormalize_target,
    normalize_inputs,
    normalize_target,
)

# ---------------------------------------------------------------------------
# Test 4: Climatology (PRD_AI_ML §8 test 4)
# "harmonic fit on a pure sinusoid recovers it within 1e-3"
# ---------------------------------------------------------------------------


def test_climatology() -> None:
    """Verify that harmonic fit on a pure sinusoid recovers it within 1e-3 (PRD_AI_ML §8 Test 4)."""
    # 2 years of daily data
    dates = pd.date_range("2010-01-01", "2011-12-31", freq="D")
    doy = dates.day_of_year.values.astype(np.float64)

    # Pure sinusoidal signal: T(d) = 22 + 3.5*cos(w*t) - 2.0*sin(w*t)
    t_rad = 2.0 * np.pi * (doy - 1.0) / 365.25
    pure_sinusoid = 22.0 + 3.5 * np.cos(t_rad) - 2.0 * np.sin(t_rad)

    # Place in a 4D array (time, depth=1, lat=2, lon=2)
    n_time = len(dates)
    thetao = np.zeros((n_time, 1, 2, 2), dtype=np.float32)
    thetao[:, 0, 0, 0] = pure_sinusoid.astype(np.float32)
    thetao[:, 0, 0, 1] = pure_sinusoid.astype(np.float32) + 1.0

    # Fit harmonic climatology (n_harmonics=3)
    clim_366 = fit_harmonic_climatology(thetao, dates, n_harmonics=3)

    assert clim_366.shape == (366, 1, 2, 2)
    assert clim_366.dtype == np.float32

    # Expected values for dayofyear 1..366
    all_doy = np.arange(1, 367, dtype=np.float64)
    expected_rad = 2.0 * np.pi * (all_doy - 1.0) / 365.25
    expected_366 = 22.0 + 3.5 * np.cos(expected_rad) - 2.0 * np.sin(expected_rad)

    fitted_signal = clim_366[:, 0, 0, 0]
    max_abs_err = np.max(np.abs(fitted_signal - expected_366))

    # Strict contract: error must be <= 1e-3
    assert max_abs_err < 1e-3, f"Harmonic fit error {max_abs_err} exceeded 1e-3"


# ---------------------------------------------------------------------------
# Test 5: Inverse Transform (PRD_AI_ML §8 test 5)
# "normalize -> denormalize round-trip within 1e-5"
# ---------------------------------------------------------------------------


def test_inverse_transform() -> None:
    """Verify normalize -> denormalize round-trip within 1e-5 (PRD_AI_ML §8 Test 5)."""
    # 1. Inputs normalization round-trip
    orig_inputs = np.array([[-0.25, 0.12, 1.45], [0.89, -1.2, 0.05]], dtype=np.float32)
    mean_val = 0.15
    std_val = 0.85

    norm_in = normalize_inputs(orig_inputs, mean_val, std_val)
    rec_in = denormalize_inputs(norm_in, mean_val, std_val)
    max_in_diff = float(np.max(np.abs(rec_in - orig_inputs)))
    assert max_in_diff < 1e-5, f"Input inverse transform difference {max_in_diff} >= 1e-5"

    # 2. Target normalization round-trip: (T - clim) / sigma_d
    depths = 15
    lats, lons = 4, 4
    orig_temp = np.random.uniform(4.0, 30.0, size=(depths, lats, lons)).astype(np.float32)
    clim_temp = np.random.uniform(4.0, 30.0, size=(depths, lats, lons)).astype(np.float32)
    sigma_d = np.linspace(0.5, 2.5, depths, dtype=np.float32)

    norm_target = normalize_target(orig_temp, clim_temp, sigma_d)
    rec_temp = denormalize_target(norm_target, clim_temp, sigma_d)

    max_target_diff = float(np.max(np.abs(rec_temp - orig_temp)))
    assert max_target_diff < 1e-5, f"Target inverse transform difference {max_target_diff} >= 1e-5"


# ---------------------------------------------------------------------------
# Test 3: No Test Leakage (PRD_AI_ML §8 test 3)
# "dataset/stat functions raise if asked for test dates outside final_test"
# ---------------------------------------------------------------------------


def test_no_test_leakage() -> None:
    """Verify stat functions raise TestLeakageError on test dates (PRD §0, §8 Test 3)."""
    test_split = ["2021-01-01", "2021-12-31"]

    # Passing test dates must raise TestLeakageError
    test_dates = pd.date_range("2021-06-01", "2021-06-10", freq="D")
    with pytest.raises(TestLeakageError):
        check_no_test_leakage(test_dates, test_split)

    # Create dummy dataset containing test dates
    all_dates = pd.date_range("2020-12-25", "2021-01-05", freq="D")
    dummy_ds = xr.Dataset(
        data_vars={
            "sla": (("time", "lat", "lon"), np.zeros((len(all_dates), 2, 2), dtype=np.float32)),
            "thetao": (
                ("time", "depth", "lat", "lon"),
                np.zeros((len(all_dates), 1, 2, 2), dtype=np.float32),
            ),
            "clim": (
                ("time", "depth", "lat", "lon"),
                np.zeros((len(all_dates), 1, 2, 2), dtype=np.float32),
            ),
            "ocean_mask": (("lat", "lon"), np.ones((2, 2), dtype=bool)),
        },
        coords={"time": all_dates, "lat": [10.0, 10.25], "lon": [60.0, 60.25]},
    )

    with pytest.raises(TestLeakageError):
        compute_norm_stats(
            dummy_ds, train_split=["2020-12-25", "2021-01-05"], test_split=test_split
        )


# ---------------------------------------------------------------------------
# Reduce Functions Tests
# ---------------------------------------------------------------------------


def test_area_mean_regrid_50pct_rule() -> None:
    """Verify the >= 50% valid sub-cells rule for spatial area reduction (PRD_AI_ML §3)."""
    target_grid = TargetGrid.from_bounds(
        lon_min=40.0, lon_max=40.5, lat_min=0.0, lat_max=0.5, resolution_deg=0.25
    )
    # 2 target cells in lat (0.125, 0.375), 2 in lon (40.125, 40.375)

    # 4x4 native sub-cells (2x2 sub-cells per target cell)
    src_lats = np.array([0.0625, 0.1875, 0.3125, 0.4375])
    src_lons = np.array([40.0625, 40.1875, 40.3125, 40.4375])

    # Target cell (0, 0) covers sub-cells [0:2, 0:2]
    # Target cell (0, 1) covers sub-cells [0:2, 2:4]
    # Target cell (1, 0) covers sub-cells [2:4, 0:2]
    # Target cell (1, 1) covers sub-cells [2:4, 2:4]

    vals = np.ones((4, 4), dtype=np.float32) * 10.0

    # In cell (0, 0): 3 valid out of 4 (75% >= 50%) -> should be 10.0
    vals[0, 0] = np.nan

    # In cell (0, 1): 2 valid out of 4 (50% >= 50%) -> should be 10.0
    vals[0, 2] = np.nan
    vals[0, 3] = np.nan

    # In cell (1, 0): 1 valid out of 4 (25% < 50%) -> should be NaN
    vals[2, 0] = np.nan
    vals[2, 1] = np.nan
    vals[3, 0] = np.nan

    # In cell (1, 1): 0 valid out of 4 (0% < 50%) -> should be NaN
    vals[2:4, 2:4] = np.nan

    reduced = area_mean_regrid_2d(vals, src_lats, src_lons, target_grid, valid_frac_threshold=0.5)

    assert reduced.shape == (2, 2)
    assert not np.isnan(reduced[0, 0]), "Cell with 75% valid should be valid"
    assert np.isclose(reduced[0, 0], 10.0)
    assert not np.isnan(reduced[0, 1]), "Cell with 50% valid should be valid"
    assert np.isclose(reduced[0, 1], 10.0)
    assert np.isnan(reduced[1, 0]), "Cell with 25% valid (< 50%) must be NaN"
    assert np.isnan(reduced[1, 1]), "Cell with 0% valid must be NaN"


def test_reduce_glorys_15_depths() -> None:
    """Verify GLORYS reduction extracts exactly the 15 configured native depths."""
    native_depths = [
        0.494,
        2.0,
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
    target_depths = [
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

    target_grid = TargetGrid.from_bounds(
        lon_min=40.0, lon_max=40.5, lat_min=0.0, lat_max=0.5, resolution_deg=0.25
    )
    t_vals = np.ones((1, len(native_depths), 4, 4), dtype=np.float32) * 20.0

    ds_glorys = xr.Dataset(
        data_vars={"thetao": (("time", "depth", "latitude", "longitude"), t_vals)},
        coords={
            "time": [np.datetime64("2020-01-01")],
            "depth": native_depths,
            "latitude": [0.0625, 0.1875, 0.3125, 0.4375],
            "longitude": [40.0625, 40.1875, 40.3125, 40.4375],
        },
    )

    reduced = reduce_glorys12(ds_glorys, target_grid, target_depths)
    assert reduced["thetao"].shape == (1, 15, 2, 2)
    assert np.allclose(reduced["thetao"].values, 20.0)
    assert "ocean_mask" in reduced
    assert "target_valid" in reduced


def test_reduce_sst_kelvin_conversion() -> None:
    """Verify SST Kelvin temperatures (> 200) are converted to Celsius (PRD_AI_ML §3)."""
    target_grid = TargetGrid.from_bounds(
        lon_min=40.0, lon_max=40.5, lat_min=0.0, lat_max=0.5, resolution_deg=0.25
    )
    # 300 K = 26.85 °C
    sst_kelvin = np.ones((1, 2, 2), dtype=np.float32) * 300.0

    ds_sst = xr.Dataset(
        data_vars={"analysed_sst": (("time", "lat", "lon"), sst_kelvin)},
        coords={
            "time": [np.datetime64("2020-01-01")],
            "lat": [0.125, 0.375],
            "lon": [40.125, 40.375],
        },
    )

    reduced = reduce_sst(ds_sst, target_grid)
    assert np.allclose(reduced["sst"].values, 300.0 - 273.15, atol=1e-3)


def test_wind_stress_and_curl() -> None:
    """Verify Large & Pond wind stress bulk formula and curl computation."""
    target_grid = TargetGrid.from_bounds(
        lon_min=40.0, lon_max=41.0, lat_min=0.0, lat_max=1.0, resolution_deg=0.25
    )
    # Uniform wind: u=10 m/s, v=0 m/s -> curl should be 0 in interior
    u = np.ones((1, 4, 4), dtype=np.float32) * 10.0
    v = np.zeros((1, 4, 4), dtype=np.float32)

    u_out, v_out, curl_out = compute_wind_stress_and_curl(u, v, target_grid)
    # Interior curl of uniform field should be 0
    assert np.isclose(curl_out[0, 1:3, 1:3], 0.0, atol=1e-8).all()


def test_reduce_gebco_log_depth() -> None:
    """Verify GEBCO elevation is converted to log(1 + depth) (PRD_AI_ML §3)."""
    target_grid = TargetGrid.from_bounds(
        lon_min=40.0, lon_max=40.5, lat_min=0.0, lat_max=0.5, resolution_deg=0.25
    )
    # Ocean depth 1000m -> elevation -1000m
    elev = np.array([[-1000.0, -1000.0], [50.0, -2000.0]], dtype=np.float32)

    ds_gebco = xr.Dataset(
        data_vars={"elevation": (("lat", "lon"), elev)},
        coords={"lat": [0.125, 0.375], "lon": [40.125, 40.375]},
    )

    reduced = reduce_gebco(ds_gebco, target_grid)
    expected_depth_00 = np.log1p(1000.0)
    expected_land_10 = np.log1p(0.0)  # max(0, -50) = 0 -> log(1+0) = 0
    assert np.isclose(reduced["bathymetry"].values[0, 0], expected_depth_00, atol=1e-4)
    assert np.isclose(reduced["bathymetry"].values[1, 0], expected_land_10, atol=1e-4)


# ---------------------------------------------------------------------------
# Pipeline & Processed Store Contract Test (PRD_AI_ML §6.1, §9 M1)
# ---------------------------------------------------------------------------


def test_preprocess_pipeline_contract(tmp_path: Path) -> None:
    """Verify run_preprocessing creates a contract-exact Zarr store and M1 sample figure."""
    config_file = tmp_path / "exp.yaml"
    config_file.write_text(
        """
project: test-preprocess
dataset_version: v1
region:
  lon_min: 40.0
  lon_max: 42.0
  lat_min: 0.0
  lat_max: 2.0
grid:
  resolution_deg: 0.25
time:
  start: "2019-01-01"
  end: "2019-01-05"
splits:
  train: ["2019-01-01", "2019-01-03"]
  val: ["2019-01-04", "2019-01-05"]
  test: ["2021-01-01", "2021-12-31"]
depths_m:
  - 0.494
  - 5.078
  - 9.573
  - 18.496
  - 29.445
  - 47.374
  - 65.807
  - 92.326
  - 130.666
  - 186.126
  - 222.475
  - 266.040
  - 380.213
  - 643.567
  - 1062.440
paths:
  raw: {raw_dir}
  processed: {proc_dir}
  outputs: {out_dir}
""".format(
            raw_dir=str(tmp_path / "raw").replace("\\", "/"),
            proc_dir=str(tmp_path / "processed").replace("\\", "/"),
            out_dir=str(tmp_path / "outputs").replace("\\", "/"),
        )
    )

    cfg = load_config(config_file)
    target_grid = TargetGrid.from_bounds(
        lon_min=cfg.region.lon_min,
        lon_max=cfg.region.lon_max,
        lat_min=cfg.region.lat_min,
        lat_max=cfg.region.lat_max,
        resolution_deg=cfg.grid.resolution_deg,
    )
    from tests.synthetic_data import generate_synthetic_raw_sample

    sample_dates = pd.date_range("2019-01-01", "2019-01-05", freq="D")
    generate_synthetic_raw_sample(tmp_path / "raw", target_grid, cfg.depths_m, dates=sample_dates)

    # Run preprocessing
    run_preprocessing(
        config=cfg,
        raw_dir=tmp_path / "raw",
        out_dir=tmp_path / "processed",
    )

    # 1. Verify Zarr store on disk
    zarr_path = tmp_path / "processed" / "oceanscope_v1.zarr"
    assert zarr_path.exists(), f"Processed store missing at {zarr_path}"

    ds_reloaded = xr.open_zarr(str(zarr_path))

    # 2. Section 6.1 required variables
    req_vars = {
        "sla",
        "u_current",
        "v_current",
        "u_wind",
        "v_wind",
        "thetao",
        "clim",
        "ocean_mask",
        "target_valid",
    }
    assert req_vars.issubset(set(ds_reloaded.data_vars)), (
        f"Missing variables: {req_vars - set(ds_reloaded.data_vars)}"
    )

    # Additional inputs
    extra_vars = {"adt", "sst", "sss", "wind_stress_curl", "net_heat_flux", "precip", "bathymetry"}
    assert extra_vars.issubset(set(ds_reloaded.data_vars))

    # 3. Section 6.1 required coordinates
    assert {"time", "depth", "lat", "lon", "dayofyear"}.issubset(set(ds_reloaded.coords))
    assert len(ds_reloaded.depth) == 15
    assert len(ds_reloaded.dayofyear) == 366
    assert len(ds_reloaded.lat) == 8  # 0.0 to 2.0 at 0.25 deg = 8 cells
    assert len(ds_reloaded.lon) == 8  # 40.0 to 42.0 at 0.25 deg = 8 cells

    # 4. Section 6.1 attributes
    req_attrs = [
        "dataset_version",
        "config_hash",
        "created_utc",
        "source_files",
        "norm_stats",
        "split_dates",
    ]
    for attr in req_attrs:
        assert attr in ds_reloaded.attrs, f"Missing root attribute '{attr}' in processed store"

    norm_stats = json.loads(ds_reloaded.attrs["norm_stats"])
    assert "channels" in norm_stats
    assert "sigma_d" in norm_stats
    assert len(norm_stats["sigma_d"]) == 15

    # 5. Milestone 1 sample day figure
    fig_path = tmp_path / "outputs" / "figures" / "m1_sample_day.png"
    assert fig_path.exists(), f"Milestone 1 figure missing at {fig_path}"
    assert fig_path.stat().st_size > 1000, "Figure file is empty or suspiciously small"


def test_missing_raw_product_raises_error(tmp_path: Path) -> None:
    """Verify run_preprocessing fails loudly when a raw product is missing (PRD §0 Rule 1)."""
    config_file = tmp_path / "exp.yaml"
    config_file.write_text(
        """
project: test-preprocess
dataset_version: v1
region:
  lon_min: 40.0
  lon_max: 42.0
  lat_min: 0.0
  lat_max: 2.0
grid:
  resolution_deg: 0.25
time:
  start: "2019-01-01"
  end: "2019-01-05"
splits:
  train: ["2019-01-01", "2019-01-03"]
  val: ["2019-01-04", "2019-01-05"]
  test: ["2021-01-01", "2021-12-31"]
depths_m:
  - 0.494
  - 5.078
  - 9.573
  - 18.496
  - 29.445
  - 47.374
  - 65.807
  - 92.326
  - 130.666
  - 186.126
  - 222.475
  - 266.040
  - 380.213
  - 643.567
  - 1062.440
paths:
  raw: {raw_dir}
  processed: {proc_dir}
  outputs: {out_dir}
""".format(
            raw_dir=str(tmp_path / "raw").replace("\\", "/"),
            proc_dir=str(tmp_path / "processed").replace("\\", "/"),
            out_dir=str(tmp_path / "outputs").replace("\\", "/"),
        )
    )
    cfg = load_config(config_file)
    (tmp_path / "raw").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing GLORYS12 raw file"):
        run_preprocessing(cfg, raw_dir=tmp_path / "raw", out_dir=tmp_path / "processed")


def test_reducer_missing_variable_raises() -> None:
    """Verify reducers fail loudly rather than falling back to arbitrary vars (REVIEW_P3 A1)."""
    from oceanscope_ml.data.reduce import reduce_sla

    target_grid = TargetGrid.from_bounds(40.0, 42.0, 0.0, 2.0, 0.25)
    # Dataset containing only adt, passed to reduce_sla
    ds_wrong = xr.Dataset(
        data_vars={"adt": (("time", "lat", "lon"), np.zeros((1, 8, 8)))},
        coords={
            "time": [np.datetime64("2019-01-01")],
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )
    with pytest.raises(KeyError, match="Expected variable 'sla' in SLA dataset"):
        reduce_sla(ds_wrong, target_grid)
