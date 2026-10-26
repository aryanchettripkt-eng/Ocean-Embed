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
