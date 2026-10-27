"""Mocked remote products for offline streaming build tests (PRD_AI_ML §4.2, §8)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.data.download import PRODUCT_SPECS
from oceanscope_ml.data.reduce import TargetGrid


def install_stream_mocks(
    monkeypatch: pytest.MonkeyPatch,
    depths_m: list[float],
    sample_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Replace every remote reader in build_store with constant-valued mock products.

    Each mock honours the date window the builder asks for, and the asked windows are
    recorded in the returned dict so a test can check what was requested.
    """
    import oceanscope_ml.data.build_store as bs_mod

    grid = TargetGrid.from_bounds(60.0, 62.0, 10.0, 12.0, 0.25)
    calls: dict[str, Any] = {"copernicus": [], "era5_last_month": [], "oscar_end": []}

    lats = grid.lats
    lons = grid.lons
    n_t = len(sample_dates)
    n_d = len(depths_m)
    n_lat = len(lats)
    n_lon = len(lons)

    ds_mock_glorys = xr.Dataset(
        data_vars={
            "thetao": (
                ("time", "depth", "lat", "lon"),
                np.full((n_t, n_d, n_lat, n_lon), 24.0, dtype=np.float32),
            )
        },
        coords={"time": sample_dates, "depth": depths_m, "lat": lats, "lon": lons},
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
        coords={"time": sample_dates, "depth": depths_m, "lat": lats, "lon": lons},
    )
    ds_mock_armor = xr.Dataset(
        data_vars={
            "to": (
                ("time", "depth", "lat", "lon"),
                np.full((n_t, n_d, n_lat, n_lon), 24.2, dtype=np.float32),
            )
        },
        coords={"time": sample_dates, "depth": depths_m, "lat": lats, "lon": lons},
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
        coords={"month": np.arange(1, 13), "depth": depths_m, "lat": lats, "lon": lons},
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
        start, end = kwargs["dates"]
        calls["copernicus"].append((dataset_id, start, end))
        return expected_datasets[dataset_id].sel(time=slice(start, end))

    def mock_era5(year: int, *args: Any, last_month: int = 12, **kwargs: Any) -> xr.Dataset:
        calls["era5_last_month"].append(last_month)
        end = pd.Timestamp(f"{year}-{last_month:02d}-01") + pd.offsets.MonthEnd(0)
        return ds_mock_era5.sel(time=slice(f"{year}-01-01", end))

    def mock_oscar(year: int, *args: Any, end_date: str | None = None, **kwargs: Any) -> xr.Dataset:
        calls["oscar_end"].append(end_date)
        return ds_mock_oscar.sel(time=slice(f"{year}-01-01", end_date or f"{year}-12-31"))

    monkeypatch.setattr(bs_mod, "stream_copernicus", mock_stream_copernicus)
    monkeypatch.setattr(bs_mod, "stream_era5_daily", mock_era5)
    monkeypatch.setattr(bs_mod, "stream_era5_arco", mock_era5)
    monkeypatch.setattr(bs_mod, "stream_oscar", mock_oscar)
    monkeypatch.setattr(bs_mod, "stream_gebco", lambda *args, **kwargs: ds_mock_gebco)
    monkeypatch.setattr(bs_mod, "stream_woa", lambda *args, **kwargs: da_mock_woa)
    return calls
