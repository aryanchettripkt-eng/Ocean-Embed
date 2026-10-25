"""Unit and integration tests for build_store pipeline and Test 13.

Implements PRD_AI_ML §4.2 and §8 (Test 13):
13. Build/preprocess parity: on a tiny synthetic remote-like dataset (no network),
    build_store's reduce path and preprocess give identical values for the same day;
    a second run after a simulated interruption skips finished years.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from oceanscope_ml.data.build_store import (
    build_store,
    check_budget,
    export_f16_store,
    parse_year_spec,
    plan_store_build,
)
from oceanscope_ml.data.download import PRODUCT_SPECS
from oceanscope_ml.data.preprocess import (
    run_preprocessing,
)
from oceanscope_ml.data.reduce import TargetGrid
from tests.synthetic_data import generate_synthetic_raw_sample


class MockConfig:
    """Mock config for build_store tests."""

    def __init__(self, raw_path: Path, processed_path: Path, outputs_path: Path) -> None:
        self.project = "oceanscope-ai"
        self.dataset_version = "v1"
        self.region = type(
            "Region",
            (),
            {"lon_min": 60.0, "lon_max": 62.0, "lat_min": 10.0, "lat_max": 12.0},
        )()
        self.grid = type("Grid", (), {"resolution_deg": 0.25})()
        self.time = type("Time", (), {"start": "2018-01-01", "end": "2019-12-31"})()
        self.splits = type(
            "Splits",
            (),
            {
                "train": ["2018-01-01", "2019-01-02"],
                "val": ["2019-01-03", "2019-01-03"],
                "test": ["2021-01-01", "2021-12-31"],
            },
        )()
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
        self.paths = type(
            "Paths",
            (),
            {
                "raw": raw_path,
                "processed": processed_path,
                "outputs": outputs_path,
                "checkpoints": outputs_path / "checkpoints",
            },
        )()
        self.raw_dict = {
            "project": "oceanscope-ai",
            "dataset_version": "v1",
            "depths_m": self.depths_m,
        }


def test_parse_year_spec() -> None:
    """Verify parsing comma-separated and hyphenated year ranges."""
    assert parse_year_spec("2019") == [2019]
    assert parse_year_spec("2018,2019,2020") == [2018, 2019, 2020]
    assert parse_year_spec("2017-2020") == [2017, 2018, 2019, 2020]
    assert parse_year_spec("2010-2012, 2015, 2018-2019") == [2010, 2011, 2012, 2015, 2018, 2019]


def test_dry_run_plan() -> None:
    """Verify dry-run calculation without downloading or disk writes."""
    cfg = MockConfig(Path("dummy/raw"), Path("dummy/processed"), Path("dummy/outputs"))
    plan = plan_store_build([2018, 2019], cfg)

    assert plan["n_years"] == 2
    assert plan["years"] == [2018, 2019]
    assert plan["total_transfer_mb"] > 0
    assert plan["total_store_growth_mb"] > 0
    assert len(plan["years_detail"]) == 2
    assert plan["years_detail"][0]["year"] == 2018
    assert "requests" in plan
    assert len(plan["requests"]) >= 8


def test_build_preprocess_parity_test_13(tmp_path: Path) -> None:
    """Test 13: Parity between build_store and preprocess on identical raw inputs."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    prep_out_dir = tmp_path / "prep_out"
    prep_out_dir.mkdir()
    build_out_dir = tmp_path / "build_out"
    build_out_dir.mkdir()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    cfg = MockConfig(raw_dir, prep_out_dir, outputs_dir)
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)

    # Generate synthetic raw sample dataset for January 2019 (3 days)
    sample_dates = pd.date_range("2019-01-01", periods=3, freq="D")
    generate_synthetic_raw_sample(raw_dir, grid, cfg.depths_m, dates=sample_dates)

    # 1. Run preprocess.py pipeline
    prep_zarr = run_preprocessing(
        config=cfg,  # type: ignore[arg-type]
        raw_dir=raw_dir,
        out_dir=prep_out_dir,
    )

    # 2. Run build_store.py pipeline for 2019
    build_zarr = build_store(
        config=cfg,
        years=[2019],
        raw_dir=raw_dir,
        out_dir=build_out_dir,
        scratch_dir=tmp_path / "scratch",
        dry_run=False,
        force=True,
    )

    # 3. Assert parity on all common data variables
    ds_prep = prep_zarr
    ds_build = xr.open_zarr(build_zarr)

    # Compare values on common time-dependent variables
    common_vars = [
        v
        for v in ds_prep.data_vars
        if v in ds_build.data_vars and "time" in ds_prep[v].dims and "time" in ds_build[v].dims
    ]
    assert len(common_vars) >= 8

    # Compare across the 3 days
    for var in common_vars:
        for t in sample_dates:
            t_str = t.strftime("%Y-%m-%d")
            prep_arr = ds_prep[var].sel(time=t_str).values
            build_arr = ds_build[var].sel(time=t_str).values

            assert prep_arr.shape == build_arr.shape, f"Shape mismatch on {var}"

            if np.issubdtype(prep_arr.dtype, np.bool_):
                assert np.array_equal(prep_arr, build_arr), f"Boolean mismatch on {var}"
            else:
                valid_mask = (~np.isnan(prep_arr)) & (~np.isnan(build_arr))
                assert np.array_equal(np.isnan(prep_arr), np.isnan(build_arr)), (
                    f"NaN pattern mismatch on {var}"
                )

                assert np.any(valid_mask), f"No finite values in {var} - box must be open ocean"
                max_diff = float(np.max(np.abs(prep_arr[valid_mask] - build_arr[valid_mask])))
                assert max_diff <= 1e-4, f"Parity violated on {var}: max diff {max_diff} > 1e-4"

    # Also assert WOA parity and finite values (REVIEW_P3 R1, R5)
    assert "woa_clim_profile" in ds_prep.data_vars, "woa_clim_profile missing from prep"
    assert "woa_clim_profile" in ds_build.data_vars, "woa_clim_profile missing from build"
    assert ds_prep["woa_clim_profile"].shape == ds_build["woa_clim_profile"].shape
    assert np.any(np.isfinite(ds_prep["woa_clim_profile"].values)), "WOA is all NaN in prep"
    assert np.any(np.isfinite(ds_build["woa_clim_profile"].values)), "WOA is all NaN in build"
    assert np.allclose(
        ds_prep["woa_clim_profile"].values,
        ds_build["woa_clim_profile"].values,
        equal_nan=True,
    )


def test_resumption_skips_finished_years_test_13(tmp_path: Path) -> None:
    """Test 13 (part 2): Run after interruption skips finished years without wiping attrs."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    cfg = MockConfig(raw_dir, out_dir, outputs_dir)
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)

    # Generate genuine test data for BOTH 2018 and 2019 in raw_dir
    dates_2018 = pd.date_range("2018-01-01", periods=2, freq="D")
    generate_synthetic_raw_sample(raw_dir, grid, cfg.depths_m, dates=dates_2018)

    dates_2019 = pd.date_range("2019-01-01", periods=2, freq="D")
    generate_synthetic_raw_sample(raw_dir, grid, cfg.depths_m, dates=dates_2019)

    # Step 1: Build year 2018
    store_path = build_store(
        config=cfg,
        years=[2018],
        raw_dir=raw_dir,
        out_dir=out_dir,
        scratch_dir=scratch_dir,
        dry_run=False,
    )

    # Verify 2018 was completed and recorded on disk
    root = zarr.open_group(str(store_path), mode="r")
    built_years = root.attrs.get("built_years", [])
    assert built_years == [2018]
    assert "dataset_version" in root.attrs
    assert "config_hash" in root.attrs

    # Step 2: Resume with [2018, 2019] - should skip 2018 and complete 2019 without wiping attrs
    store_path_2 = build_store(
        config=cfg,
        years=[2018, 2019],
        raw_dir=raw_dir,
        out_dir=out_dir,
        scratch_dir=scratch_dir,
        dry_run=False,
    )
    root2 = zarr.open_group(str(store_path_2), mode="r")
    built_years_2 = root2.attrs.get("built_years", [])
    assert built_years_2 == [2018, 2019]
    assert "dataset_version" in root2.attrs


def test_missing_raw_product_raises_no_synthetic(tmp_path: Path) -> None:
    """Test that missing raw files raise FileNotFoundError and never fabricate synthetic data."""
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cfg = MockConfig(empty_raw, out_dir, tmp_path / "outputs")
    with pytest.raises(FileNotFoundError):
        build_store(
            config=cfg,
            years=[2018],
            raw_dir=empty_raw,
            out_dir=out_dir,
        )


def test_budget_enforcement(tmp_path: Path) -> None:
    """Test that exceeding scratch or store budget raises RuntimeError."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    store = tmp_path / "store"
    store.mkdir()

    # Create dummy file of 100 bytes
    dummy = scratch / "dummy.bin"
    dummy.write_bytes(b"0" * 100)

    # Budget limit 1e-9 GB (~1 byte) should trigger
    with pytest.raises(RuntimeError, match="budget exceeded"):
        check_budget(scratch, store, max_scratch_gb=1e-9)


def test_export_f16_store(tmp_path: Path) -> None:
    """Verify export_f16_store creates valid float16 Zarr store with _f16.zarr naming."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    cfg = MockConfig(raw_dir, out_dir, outputs_dir)
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    sample_dates = pd.date_range("2019-01-01", periods=2, freq="D")
    generate_synthetic_raw_sample(raw_dir, grid, cfg.depths_m, dates=sample_dates)

    master_zarr = build_store(
        config=cfg,
        years=[2019],
        raw_dir=raw_dir,
        out_dir=out_dir,
        scratch_dir=tmp_path / "scratch",
        dry_run=False,
    )

    f16_zarr = out_dir / "oceanscope_v1_f16.zarr"
    export_f16_store(master_zarr, f16_zarr)

    ds_f16 = xr.open_zarr(f16_zarr)
    assert ds_f16["thetao"].dtype == np.float16
    assert ds_f16["sst"].dtype == np.float16
    assert ds_f16.attrs["dtype"] == "float16"


def test_filter_files_by_year() -> None:
    r"""Verify filter_files_by_year uses regex (?<!\d)(19|20)\d\d(?!\d) on filename only."""
    from oceanscope_ml.data.build_store import filter_files_by_year

    files = [
        Path("/data/raw/glorys/glorys_2018.nc"),
        Path("/data/raw/glorys/glorys_2019.nc"),
        Path("/data/raw/glorys/glorys_20190101.nc"),
        Path("/data/raw/2019/unnamed_file.nc"),
        Path("/data/raw/sla/sla_2018-11-02-2019-01-31.nc"),
    ]
    # For 2019: matches glorys_2019.nc and sla_2018-11-02-2019-01-31.nc
    m_2019 = filter_files_by_year(files, 2019)
    assert Path("/data/raw/glorys/glorys_2019.nc") in m_2019
    assert Path("/data/raw/sla/sla_2018-11-02-2019-01-31.nc") in m_2019
    # /data/raw/2019/unnamed_file.nc must NOT match:
    # 2019 is only in parent path, not filename
    assert Path("/data/raw/2019/unnamed_file.nc") not in m_2019
    # 2018 file must not be included
    assert Path("/data/raw/glorys/glorys_2018.nc") not in m_2019

    # When looking for 2020: nothing matches -> returns []
    assert filter_files_by_year(files, 2020) == []


def test_stream_build_store_full_year_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_store with stream=True builds full year from empty raw folder."""
    import oceanscope_ml.data.build_store as bs_mod

    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    cfg = MockConfig(empty_raw, out_dir, tmp_path / "outputs")
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    sample_dates = pd.date_range("2019-01-01", periods=3, freq="D")

    lats = grid.lats
    lons = grid.lons
    n_t = len(sample_dates)
    n_d = len(cfg.depths_m)
    n_lat = len(lats)
    n_lon = len(lons)

    ds_mock_glorys = xr.Dataset(
        data_vars={
            "thetao": (
                ("time", "depth", "lat", "lon"),
                np.full((n_t, n_d, n_lat, n_lon), 24.0, dtype=np.float32),
            )
        },
        coords={"time": sample_dates, "depth": cfg.depths_m, "lat": lats, "lon": lons},
    )
    ds_mock_sla = xr.Dataset(
        data_vars={
            "sla": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.05, dtype=np.float32)),
            "adt": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.85, dtype=np.float32)),
        },
        coords={"time": sample_dates, "lat": lats, "lon": lons},
    )
    ds_mock_sst = xr.Dataset(
        data_vars={
            "analysed_sst": (
                ("time", "lat", "lon"),
                np.full((n_t, n_lat, n_lon), 299.15, dtype=np.float32),
            )
        },
        coords={"time": sample_dates, "lat": lats, "lon": lons},
    )
    ds_mock_sst["analysed_sst"].attrs["units"] = "Kelvin"
    ds_mock_sss = xr.Dataset(
        data_vars={
            "sos": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 35.0, dtype=np.float32))
        },
        coords={"time": sample_dates, "lat": lats, "lon": lons},
    )
    ds_mock_grep = xr.Dataset(
        data_vars={
            "thetao_mean": (
                ("time", "depth", "lat", "lon"),
                np.full((n_t, n_d, n_lat, n_lon), 23.5, dtype=np.float32),
            )
        },
        coords={"time": sample_dates, "depth": cfg.depths_m, "lat": lats, "lon": lons},
    )
    ds_mock_armor = xr.Dataset(
        data_vars={
            "to": (
                ("time", "depth", "lat", "lon"),
                np.full((n_t, n_d, n_lat, n_lon), 24.2, dtype=np.float32),
            )
        },
        coords={"time": sample_dates, "depth": cfg.depths_m, "lat": lats, "lon": lons},
    )
    ds_mock_era5 = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), -3.0, dtype=np.float32)),
            "v10": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), -4.0, dtype=np.float32)),
            "ssr": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 15.0, dtype=np.float32)),
            "str": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 10.0, dtype=np.float32)),
            "sshf": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 5.0, dtype=np.float32)),
            "slhf": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 5.0, dtype=np.float32)),
            "tp": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.002, dtype=np.float32)),
        },
        coords={"time": sample_dates, "lat": lats, "lon": lons},
    )
    for v in ["ssr", "str", "sshf", "slhf"]:
        ds_mock_era5[v].attrs["units"] = "W m**-2"
    ds_mock_era5["tp"].attrs["units"] = "m"

    ds_mock_oscar = xr.Dataset(
        data_vars={
            "u": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.1, dtype=np.float32)),
            "v": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), -0.05, dtype=np.float32)),
        },
        coords={"time": sample_dates, "lat": lats, "lon": lons},
    )
    ds_mock_gebco = xr.Dataset(
        data_vars={
            "elevation": (("lat", "lon"), np.full((n_lat, n_lon), -1500.0, dtype=np.float32))
        },
        coords={"lat": lats, "lon": lons},
    )
    da_mock_woa = xr.DataArray(
        np.full((12, n_d, n_lat, n_lon), 25.0, dtype=np.float32),
        dims=("month", "depth", "lat", "lon"),
        coords={"month": np.arange(1, 13), "depth": cfg.depths_m, "lat": lats, "lon": lons},
    )

    expected_datasets: dict[str, xr.Dataset] = {
        PRODUCT_SPECS["glorys12"]["dataset_id"]: ds_mock_glorys,
        PRODUCT_SPECS["sla"]["dataset_id"]: ds_mock_sla,
        PRODUCT_SPECS["sst"]["dataset_id"]: ds_mock_sst,
        PRODUCT_SPECS["sss"]["dataset_id"]: ds_mock_sss,
        PRODUCT_SPECS["grep"]["dataset_id"]: ds_mock_grep,
        PRODUCT_SPECS["armor3d"]["dataset_id"]: ds_mock_armor,
    }

    def mock_stream_copernicus(dataset_id: str, **kwargs: Any) -> xr.Dataset:
        if dataset_id not in expected_datasets:
            raise ValueError(f"Wrong dataset_id: {dataset_id}")
        return expected_datasets[dataset_id]

    monkeypatch.setattr(bs_mod, "stream_copernicus", mock_stream_copernicus)
    monkeypatch.setattr(bs_mod, "stream_era5_daily", lambda *args, **kwargs: ds_mock_era5)
    monkeypatch.setattr(bs_mod, "stream_oscar", lambda *args, **kwargs: ds_mock_oscar)
    monkeypatch.setattr(bs_mod, "stream_gebco", lambda *args, **kwargs: ds_mock_gebco)
    monkeypatch.setattr(bs_mod, "stream_woa", lambda *args, **kwargs: da_mock_woa)

    # Build store in streaming mode from empty raw folder
    store_dest = build_store(
        config=cfg,
        years=[2019],
        raw_dir=empty_raw,
        out_dir=out_dir,
        scratch_dir=scratch_dir,
        stream=True,
    )

    assert store_dest.exists()
    ds_built = xr.open_zarr(store_dest)
    root = zarr.open_group(str(store_dest), mode="r")

    assert root.attrs["built_years"] == [2019]
    for required_var in [
        "thetao",
        "ocean_mask",
        "target_valid",
        "sla",
        "adt",
        "sst",
        "sss",
        "u_current",
        "v_current",
        "u_wind",
        "v_wind",
        "wind_stress_curl",
        "net_heat_flux",
        "precip",
        "bathymetry",
        "woa_clim_profile",
        "grep_mean",
        "armor3d",
    ]:
        assert required_var in ds_built.data_vars
        val = ds_built[required_var].values
        assert np.sum(np.isfinite(val)) > 0


def test_era5_hourly_and_daily_precipitation_parity() -> None:
    """Verify streamed daily-statistics precipitation (scaled x24) matches hourly accumulation."""
    from oceanscope_ml.data.reduce import reduce_era5_fluxes

    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    lats = grid.lats
    lons = grid.lons
    n_lat = len(lats)
    n_lon = len(lons)

    # Hourly dataset for 2019-01-01: 24 hours with constant 0.001 m/hr precipitation
    hourly_times = pd.date_range("2019-01-01 01:00:00", "2019-01-02 00:00:00", freq="h")
    n_h = len(hourly_times)
    ds_hourly = xr.Dataset(
        data_vars={
            "ssr": (("time", "lat", "lon"), np.full((n_h, n_lat, n_lon), 40.0, dtype=np.float32)),
            "str": (("time", "lat", "lon"), np.full((n_h, n_lat, n_lon), 10.0, dtype=np.float32)),
            "sshf": (("time", "lat", "lon"), np.full((n_h, n_lat, n_lon), 5.0, dtype=np.float32)),
            "slhf": (("time", "lat", "lon"), np.full((n_h, n_lat, n_lon), 5.0, dtype=np.float32)),
            "tp": (("time", "lat", "lon"), np.full((n_h, n_lat, n_lon), 0.001, dtype=np.float32)),
        },
        coords={"time": hourly_times, "lat": lats, "lon": lons},
    )
    for v in ["ssr", "str", "sshf", "slhf"]:
        ds_hourly[v].attrs["units"] = "W m**-2"
    ds_hourly["tp"].attrs["units"] = "m"

    # Daily dataset: 1 daily value with mean rate 0.001 m/hr
    daily_times = pd.date_range("2019-01-01", periods=1, freq="D")
    ds_daily = xr.Dataset(
        data_vars={
            "ssr": (("time", "lat", "lon"), np.full((1, n_lat, n_lon), 40.0, dtype=np.float32)),
            "str": (("time", "lat", "lon"), np.full((1, n_lat, n_lon), 10.0, dtype=np.float32)),
            "sshf": (("time", "lat", "lon"), np.full((1, n_lat, n_lon), 5.0, dtype=np.float32)),
            "slhf": (("time", "lat", "lon"), np.full((1, n_lat, n_lon), 5.0, dtype=np.float32)),
            "tp": (("time", "lat", "lon"), np.full((1, n_lat, n_lon), 0.001, dtype=np.float32)),
        },
        coords={"time": daily_times, "lat": lats, "lon": lons},
    )
    for v in ["ssr", "str", "sshf", "slhf"]:
        ds_daily[v].attrs["units"] = "W m**-2"
    ds_daily["tp"].attrs["units"] = "m"

    target_time = np.array(["2019-01-01"], dtype="datetime64[ns]")
    red_hourly = reduce_era5_fluxes(ds_hourly, grid, target_time=target_time)
    red_daily = reduce_era5_fluxes(ds_daily, grid, target_time=target_time)

    # 24 hours * 0.001 m = 0.024 m daily accumulation
    assert np.allclose(red_hourly["precip"].values, 0.024, atol=1e-5)
    assert np.allclose(red_daily["precip"].values, 0.024, atol=1e-5)
    assert np.allclose(red_hourly["precip"].values, red_daily["precip"].values, atol=1e-5)
    assert np.allclose(
        red_hourly["net_heat_flux"].values, red_daily["net_heat_flux"].values, atol=1e-5
    )


def test_filter_files_by_year_time_range(tmp_path: Path) -> None:
    """Verify filter_files_by_year inspects NetCDF time range for non-year-named ERA5 files."""
    from oceanscope_ml.data.build_store import filter_files_by_year

    era5_file = tmp_path / "data_stream-oper_stepType-instant.nc"
    ds = xr.Dataset(coords={"time": pd.date_range("2019-01-01", "2019-12-31")})
    ds.to_netcdf(era5_file)

    files = [
        Path("/data/raw/glorys/glorys_2018.nc"),
        Path("/data/raw/glorys/glorys_2019.nc"),
        era5_file,
    ]

    # For 2019: glorys_2019.nc and era5_file (covers 2019) match
    m_2019 = filter_files_by_year(files, 2019)
    assert Path("/data/raw/glorys/glorys_2019.nc") in m_2019
    assert era5_file in m_2019
    assert Path("/data/raw/glorys/glorys_2018.nc") not in m_2019

    # For 2020: era5_file ends in 2019, so nothing matches
    assert filter_files_by_year(files, 2020) == []


def test_stream_silent_failures_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify streaming failures for GEBCO or WOA raise RuntimeError instead of silent warning."""
    import oceanscope_ml.data.build_store as bs_mod

    cfg = MockConfig(tmp_path / "raw", tmp_path / "out", tmp_path / "outputs")
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    sample_dates = pd.date_range("2019-01-01", periods=2, freq="D")
    n_t = len(sample_dates)
    n_d = len(cfg.depths_m)
    n_lat = len(grid.lats)
    n_lon = len(grid.lons)

    ds_mock = xr.Dataset(
        data_vars={
            "thetao": (
                ("time", "depth", "lat", "lon"),
                np.full((n_t, n_d, n_lat, n_lon), 25.0, dtype=np.float32),
            ),
            "sla": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.05, dtype=np.float32)),
            "adt": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.85, dtype=np.float32)),
            "analysed_sst": (
                ("time", "lat", "lon"),
                np.full((n_t, n_lat, n_lon), 299.0, dtype=np.float32),
            ),
            "sos": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 35.0, dtype=np.float32)),
            "u": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.1, dtype=np.float32)),
            "v": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), -0.1, dtype=np.float32)),
            "u10": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 5.0, dtype=np.float32)),
            "v10": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), -3.0, dtype=np.float32)),
            "ssr": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 30.0, dtype=np.float32)),
            "str": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 10.0, dtype=np.float32)),
            "sshf": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 5.0, dtype=np.float32)),
            "slhf": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 5.0, dtype=np.float32)),
            "tp": (("time", "lat", "lon"), np.full((n_t, n_lat, n_lon), 0.001, dtype=np.float32)),
        },
        coords={"time": sample_dates, "depth": cfg.depths_m, "lat": grid.lats, "lon": grid.lons},
    )
    for v in ["ssr", "str", "sshf", "slhf"]:
        ds_mock[v].attrs["units"] = "W m**-2"
    ds_mock["tp"].attrs["units"] = "m"
    ds_mock["analysed_sst"].attrs["units"] = "Kelvin"

    monkeypatch.setattr(bs_mod, "stream_copernicus", lambda *args, **kwargs: ds_mock)
    monkeypatch.setattr(bs_mod, "stream_era5_daily", lambda *args, **kwargs: ds_mock)
    monkeypatch.setattr(bs_mod, "stream_oscar", lambda *args, **kwargs: ds_mock)

    def failing_stream(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("Network unreachable")

    monkeypatch.setattr(bs_mod, "stream_gebco", failing_stream)

    with pytest.raises(RuntimeError, match="Failed to stream GEBCO bathymetry"):
        bs_mod.reduce_year_dataset(
            year=2019,
            raw_path=tmp_path / "raw",
            target_grid=grid,
            depths_m=cfg.depths_m,
            stream=True,
            scratch_dir=tmp_path / "scratch",
        )
