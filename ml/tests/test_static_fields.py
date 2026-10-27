"""Tests for the static-fields file and zipped CDS downloads (Kaggle pre-flight fixes)."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.data.build_store import _open_cds_download, build_store, load_static_fields
from oceanscope_ml.data.export_static import export_static
from oceanscope_ml.data.reduce import TargetGrid
from tests.synthetic_data import generate_synthetic_raw_sample
from tests.test_build_store import MockConfig


def test_open_cds_download_plain_and_zip(tmp_path: Path) -> None:
    """A CDS download opens whether it is a netCDF or a zip of per-stepType netCDFs."""
    time = pd.date_range("2017-01-01", periods=2, freq="D")
    coords = {"valid_time": time, "latitude": [10.0, 10.25], "longitude": [60.0, 60.25]}
    inst = xr.Dataset(
        {"u10": (("valid_time", "latitude", "longitude"), np.ones((2, 2, 2)))}, coords
    )
    accum = xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"), np.zeros((2, 2, 2)))}, coords
    )

    plain = tmp_path / "plain.nc"
    inst.to_netcdf(plain)
    assert list(_open_cds_download(plain).data_vars) == ["u10"]

    parts = {"stepType-instant.nc": inst, "stepType-accum.nc": accum}
    for name, ds in parts.items():
        ds.to_netcdf(tmp_path / name)
    zipped = tmp_path / "era5_daily_2017.nc"  # CDS keeps the requested name, even for a zip
    with zipfile.ZipFile(zipped, "w") as zf:
        for name in parts:
            zf.write(tmp_path / name, arcname=name)
    ds = _open_cds_download(zipped)
    assert set(ds.data_vars) == {"u10", "tp"}
    assert ds.sizes["valid_time"] == 2


def test_static_file_replaces_gebco_and_woa(tmp_path: Path) -> None:
    """Export static fields from a store, then build without any GEBCO/WOA raw files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = MockConfig(raw_dir, tmp_path / "out1", tmp_path / "outputs")
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    generate_synthetic_raw_sample(
        raw_dir, grid, cfg.depths_m, dates=pd.date_range("2019-01-01", periods=3, freq="D")
    )

    first = build_store(
        config=cfg,
        years=[2019],
        raw_dir=raw_dir,
        out_dir=tmp_path / "out1",
        scratch_dir=tmp_path / "scratch1",
        force=True,
    )
    static = export_static(first, tmp_path / "static_fields.nc")
    bathy, woa = load_static_fields(static, grid, cfg.depths_m)
    ds_first = xr.open_zarr(first)
    np.testing.assert_allclose(bathy.values, ds_first["bathymetry"].values, equal_nan=True)
    np.testing.assert_allclose(woa.values, ds_first["woa_clim_profile"].values, equal_nan=True)

    shutil.rmtree(raw_dir / "gebco")
    shutil.rmtree(raw_dir / "woa23")
    (raw_dir / "MANIFEST.json").unlink(missing_ok=True)
    second = build_store(
        config=cfg,
        years=[2019],
        raw_dir=raw_dir,
        out_dir=tmp_path / "out2",
        scratch_dir=tmp_path / "scratch2",
        force=True,
        static_file=static,
    )
    ds_second = xr.open_zarr(second)
    np.testing.assert_allclose(
        ds_second["bathymetry"].values, ds_first["bathymetry"].values, equal_nan=True
    )
    np.testing.assert_allclose(
        ds_second["woa_clim_profile"].values, ds_first["woa_clim_profile"].values, equal_nan=True
    )


def test_static_file_grid_mismatch_raises(tmp_path: Path) -> None:
    """A static file from another region is refused, not silently misaligned."""
    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    other = TargetGrid.from_bounds(70.0, 72.0, 10.0, 12.0, 0.25)
    depths = [0.5, 5.0]
    ds = xr.Dataset(
        {
            "bathymetry": (("lat", "lon"), np.zeros((8, 8), dtype=np.float32)),
            "woa_clim_profile": (
                ("month", "depth", "lat", "lon"),
                np.zeros((12, 2, 8, 8), dtype=np.float32),
            ),
        },
        coords={"month": np.arange(1, 13), "depth": depths, "lat": other.lats, "lon": other.lons},
    )
    path = tmp_path / "static.nc"
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="grid does not match"):
        load_static_fields(path, grid, depths)


def test_stream_era5_daily_requests_month_by_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ERA5 is requested one month per CDS call (a full year exceeds the cost limit)."""
    import sys
    import types

    from oceanscope_ml.data.build_store import stream_era5_daily

    calls: list[dict[str, object]] = []

    class FakeClient:
        def retrieve(self, name: str, request: dict[str, object], target: str) -> None:
            calls.append(request)
            month = int(request["month"][0])  # type: ignore[index]
            days = pd.date_range(f"2017-{month:02d}-01", periods=2, freq="D")
            xr.Dataset(
                {"u10": (("valid_time", "latitude", "longitude"), np.full((2, 2, 2), month))},
                coords={"valid_time": days, "latitude": [10.0, 10.25], "longitude": [60.0, 60.25]},
            ).to_netcdf(target)

    monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=FakeClient))
    ds = stream_era5_daily(2017, (60.0, 62.0, 10.0, 12.0), tmp_path)
    assert len(calls) == 12
    assert all(len(c["month"]) == 1 for c in calls)  # type: ignore[arg-type]
    assert ds.sizes["valid_time"] == 24
    assert ds.attrs["is_daily_mean"]

    calls.clear()
    stream_era5_daily(2017, (60.0, 62.0, 10.0, 12.0), tmp_path)
    assert calls == []  # every month is cached


def test_hourly_to_daily_mean_matches_hourly_reducer() -> None:
    """ARCO hourly -> daily means gives the same fluxes as feeding hourly data to the reducer."""
    from oceanscope_ml.data.build_store import hourly_to_daily_mean
    from oceanscope_ml.data.reduce import reduce_era5_fluxes, reduce_era5_winds

    grid = TargetGrid.from_bounds(60.0, 61.0, 10.0, 11.0, 0.25)
    lats = np.arange(11.5, 9.4, -0.25)
    lons = np.arange(59.5, 61.6, 0.25)
    hours = pd.date_range("2017-01-01T00", "2017-01-04T00", freq="h")
    rng = np.random.default_rng(0)
    shape = (len(hours), len(lats), len(lons))
    data = {
        "u10": ("m s**-1", False, rng.normal(0, 5, shape)),
        "v10": ("m s**-1", False, rng.normal(0, 5, shape)),
        "ssr": ("J m**-2", True, rng.uniform(0, 3e6, shape)),
        "str": ("J m**-2", True, rng.uniform(-3e5, 0, shape)),
        "sshf": ("J m**-2", True, rng.uniform(-1e5, 0, shape)),
        "slhf": ("J m**-2", True, rng.uniform(-5e5, 0, shape)),
        "tp": ("m", True, rng.uniform(0, 1e-3, shape)),
    }
    hourly = xr.Dataset(
        {
            k: xr.DataArray(
                v.astype(np.float32),
                dims=("time", "latitude", "longitude"),
                attrs={"units": u, "is_accumulation": acc},
            )
            for k, (u, acc, v) in data.items()
        },
        coords={"time": hours, "latitude": lats, "longitude": lons},
    )
    days = pd.date_range("2017-01-01", "2017-01-03", freq="D")
    daily = hourly_to_daily_mean(hourly, days)
    assert daily.attrs["daily_statistic"] == "daily_mean"
    assert daily["ssr"].attrs["units"] == "J m**-2"

    ref_flux = reduce_era5_fluxes(hourly, grid, target_time=days.values)
    new_flux = reduce_era5_fluxes(daily, grid, target_time=days.values)
    for v in ("net_heat_flux", "precip"):
        np.testing.assert_allclose(new_flux[v].values, ref_flux[v].values, rtol=1e-4)
    ref_wind = reduce_era5_winds(hourly, grid, target_time=days.values)
    new_wind = reduce_era5_winds(daily, grid, target_time=days.values)
    np.testing.assert_allclose(new_wind["u_wind"].values, ref_wind["u_wind"].values, rtol=1e-4)
