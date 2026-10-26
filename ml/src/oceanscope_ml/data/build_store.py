"""Full-period streaming store builder for OceanScope AI.

Implements PRD_AI_ML §4.2: stream -> reduce -> store, one calendar year at a time.
No raw archive is kept on disk (target footprint <= 20 GB scratch, <= 20 GB store).
Calls the shared per-product reduce functions from oceanscope_ml.data.reduce to guarantee
exact parity with preprocess.py (Test 13).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from oceanscope_ml.data.climatology import fit_harmonic_climatology
from oceanscope_ml.data.download import PRODUCT_SPECS
from oceanscope_ml.data.preprocess import (
    compute_config_hash,
    find_product_files,
    load_config,
)
from oceanscope_ml.data.reduce import (
    TargetGrid,
    reduce_adt,
    reduce_armor3d,
    reduce_currents,
    reduce_era5_fluxes,
    reduce_era5_winds,
    reduce_gebco,
    reduce_glorys12,
    reduce_grep,
    reduce_sla,
    reduce_sss,
    reduce_sst,
    reduce_woa,
    standardize_coords,
)
from oceanscope_ml.data.transform import check_no_test_leakage, compute_norm_stats

logger = logging.getLogger("oceanscope_ml.data.build_store")


def check_budget(
    scratch_dir: Path | None,
    store_dir: Path,
    max_scratch_gb: float = 20.0,
    max_store_gb: float = 20.0,
) -> None:
    """Enforce scratch disk and store budget (PRD §4.2: <= 20 GB scratch, <= 20 GB store)."""
    if scratch_dir is not None and scratch_dir.exists():
        scratch_bytes = sum(f.stat().st_size for f in scratch_dir.rglob("*") if f.is_file())
        scratch_gb = scratch_bytes / (1024.0**3)
        if scratch_gb > max_scratch_gb:
            raise RuntimeError(
                f"Scratch disk budget exceeded: {scratch_gb:.2f} GB "
                f"> {max_scratch_gb:.1f} GB limit."
            )

    if store_dir.exists():
        store_bytes = sum(f.stat().st_size for f in store_dir.rglob("*") if f.is_file())
        store_gb = store_bytes / (1024.0**3)
        if store_gb > max_store_gb:
            raise RuntimeError(
                f"Store disk budget exceeded: {store_gb:.2f} GB > {max_store_gb:.1f} GB limit."
            )


def parse_year_spec(year_spec: str) -> list[int]:
    """Parse comma-separated or hyphenated year spec into sorted integer list."""
    years: set[int] = set()
    for part in year_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            years.update(range(int(start_s.strip()), int(end_s.strip()) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def plan_store_build(
    years: list[int],
    config: Any,
) -> dict[str, Any]:
    """Calculate planned remote requests, transfer sizes, and store growth for dry-run.

    Implements PRD §4.2 and REVIEW B7.
    """
    lon_min = config.region.lon_min
    lon_max = config.region.lon_max
    lat_min = config.region.lat_min
    lat_max = config.region.lat_max
    res = config.grid.resolution_deg
    depths = config.depths_m

    n_lat = int(round((lat_max - lat_min) / res))
    n_lon = int(round((lon_max - lon_min) / res))
    n_depths = len(depths)

    cells_2d = n_lat * n_lon
    cells_3d = n_depths * cells_2d

    # Remote daily requests derived from spatial and depth dimensions
    glorys_daily_mb = (380 * 740 * n_depths * 4) / (1024.0**2)
    sla_daily_mb = (250 * 490 * 2 * 4) / (1024.0**2)
    sst_daily_mb = (620 * 1220 * 1 * 4) / (1024.0**2)
    sss_daily_mb = (250 * 490 * 1 * 4) / (1024.0**2)
    oscar_daily_mb = (cells_2d * 2 * 4) / (1024.0**2)
    era5_daily_mb = (cells_2d * 5 * 4) / (1024.0**2)
    grep_daily_mb = (cells_3d * 4) / (1024.0**2)
    armor_daily_mb = (250 * 490 * n_depths * 4) / (1024.0**2)

    daily_transfer_mb = (
        glorys_daily_mb
        + sla_daily_mb
        + sst_daily_mb
        + sss_daily_mb
        + oscar_daily_mb
        + era5_daily_mb
        + grep_daily_mb
        + armor_daily_mb
    )

    daily_store_uncompressed_mb = (cells_3d * 4 * 4 + cells_2d * 11 * 4) / (1024.0**2)
    daily_store_compressed_mb = daily_store_uncompressed_mb / 3.0

    static_transfer_mb = 75.0
    static_store_mb = 25.0

    plan_per_year: list[dict[str, Any]] = []
    total_transfer_mb = static_transfer_mb
    total_store_growth_mb = static_store_mb

    for yr in years:
        days_in_yr = 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
        yr_transfer = days_in_yr * daily_transfer_mb
        yr_store = days_in_yr * daily_store_compressed_mb

        total_transfer_mb += yr_transfer
        total_store_growth_mb += yr_store

        plan_per_year.append(
            {
                "year": yr,
                "days": days_in_yr,
                "estimated_transfer_mb": round(yr_transfer, 1),
                "estimated_store_mb": round(yr_store, 1),
            }
        )

    requests_info = [
        {
            "product": "glorys12",
            "service": "Copernicus Marine",
            "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            "var": "thetao",
            "dims": f"{n_depths} depths x 1/12° bbox",
            "daily_mb": round(glorys_daily_mb, 2),
        },
        {
            "product": "sla_adt",
            "service": "Copernicus Marine",
            "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            "var": "sla, adt",
            "dims": "0.125° bbox",
            "daily_mb": round(sla_daily_mb, 2),
        },
        {
            "product": "sst",
            "service": "Copernicus Marine",
            "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
            "var": "analysed_sst",
            "dims": "0.05° bbox",
            "daily_mb": round(sst_daily_mb, 2),
        },
        {
            "product": "sss",
            "service": "Copernicus Marine",
            "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
            "var": "sos",
            "dims": "0.125° bbox",
            "daily_mb": round(sss_daily_mb, 2),
        },
        {
            "product": "oscar",
            "service": "NASA PO.DAAC / Harmony",
            "dataset_id": "oscar_currents",
            "var": "u, v",
            "dims": "0.25° bbox",
            "daily_mb": round(oscar_daily_mb, 2),
        },
        {
            "product": "era5",
            "service": "Copernicus CDS",
            "dataset_id": "derived-era5-single-levels-daily-statistics",
            "var": "u10, v10, fluxes, precip",
            "dims": "0.25° bbox",
            "daily_mb": round(era5_daily_mb, 2),
        },
        {
            "product": "grep",
            "service": "Copernicus Marine",
            "dataset_id": "cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m",
            "var": "thetao_mean",
            "dims": f"{n_depths} depths x 0.25° bbox",
            "daily_mb": round(grep_daily_mb, 2),
        },
        {
            "product": "armor3d",
            "service": "Copernicus Marine",
            "dataset_id": "cmems_obs-mob_glo_phy_my_0.125deg_P1D-m",
            "var": "to",
            "dims": f"{n_depths} depths x 0.125° bbox",
            "daily_mb": round(armor_daily_mb, 2),
        },
    ]

    return {
        "years": years,
        "n_years": len(years),
        "region_bbox": [lon_min, lon_max, lat_min, lat_max],
        "grid_resolution_deg": res,
        "grid_shape": [n_lat, n_lon],
        "depths_count": n_depths,
        "requests": requests_info,
        "static_transfer_mb": round(static_transfer_mb, 1),
        "total_transfer_mb": round(total_transfer_mb, 1),
        "total_store_growth_mb": round(total_store_growth_mb, 1),
        "years_detail": plan_per_year,
    }


def print_dry_run_plan(plan: dict[str, Any]) -> None:
    """Print formatted dry-run plan table to console."""
    print("\n" + "=" * 80)
    print("  OCEANSCOPE AI - BUILD STORE DRY-RUN PLAN (PRD_AI_ML §4.2)")
    print("=" * 80)
    print(f"Target Years       : {plan['years']}")
    print(f"Total Years Count  : {plan['n_years']}")
    dims_str = f"{plan['grid_shape'][0]}x{plan['grid_shape'][1]} ({plan['grid_resolution_deg']}°)"
    print(f"Grid Dimensions    : {dims_str}, {plan['depths_count']} depths")
    print(f"Static Transfer    : {plan['static_transfer_mb']:.1f} MB (GEBCO + WOA23)")
    total_tx_gb = plan["total_transfer_mb"] / 1024.0
    total_st_gb = plan["total_store_growth_mb"] / 1024.0
    print(f"Total Est. Transfer: {total_tx_gb:.2f} GB ({plan['total_transfer_mb']:.1f} MB)")
    print(f"Total Est. Storage : {total_st_gb:.2f} GB (target <= 20 GB master)")
    print("-" * 80)
    print(f"{'Product':<12} {'Service':<22} {'Dataset ID':<35} {'Est. MB/day':<10}")
    print("-" * 80)
    for r in plan["requests"]:
        print(f"{r['product']:<12} {r['service']:<22} {r['dataset_id']:<35} {r['daily_mb']:<10.2f}")
    print("-" * 80)
    print(f"{'Year':<8} {'Days':<6} {'Est. Transfer (MB)':<22} {'Est. Store Growth (MB)':<22}")
    print("-" * 80)
    for y_info in plan["years_detail"]:
        y_yr = y_info["year"]
        y_days = y_info["days"]
        y_tx = y_info["estimated_transfer_mb"]
        y_st = y_info["estimated_store_mb"]
        print(f"{y_yr:<8} {y_days:<6} {y_tx:<22.1f} {y_st:<22.1f}")
    print("=" * 80)
    print("Dry-run complete. No data was downloaded or written.\n")


def stream_copernicus(
    dataset_id: str,
    bbox: tuple[float, float, float, float],
    dates: tuple[str, str],
    variables: list[str] | None = None,
    depths: tuple[float, float] | None = None,
    depth_levels: list[float] | None = None,
) -> xr.Dataset:
    """Stream a remote Copernicus Marine dataset via copernicusmarine.open_dataset."""
    try:
        import copernicusmarine
    except ImportError as e:
        raise RuntimeError("copernicusmarine package is required for streaming data.") from e

    lon_min, lon_max, lat_min, lat_max = bbox
    kwargs: dict[str, Any] = {
        "dataset_id": dataset_id,
        "minimum_longitude": lon_min - 1.0,
        "maximum_longitude": lon_max + 1.0,
        "minimum_latitude": lat_min - 1.0,
        "maximum_latitude": lat_max + 1.0,
        "start_datetime": dates[0],
        "end_datetime": dates[1],
    }
    if variables:
        kwargs["variables"] = variables
    if depths:
        kwargs["minimum_depth"] = depths[0]
        kwargs["maximum_depth"] = depths[1]
    elif depth_levels:
        kwargs["minimum_depth"] = float(min(depth_levels))
        kwargs["maximum_depth"] = float(max(depth_levels))

    logger.info("Streaming remote dataset %s for %s..%s...", dataset_id, dates[0], dates[1])
    ds = copernicusmarine.open_dataset(**kwargs)
    if depth_levels is not None and "depth" in ds.coords:
        ds_depths = np.asarray(ds.coords["depth"].values)
        matched_idx: list[int] = []
        for dl in depth_levels:
            diffs = np.abs(ds_depths - dl)
            idx = int(np.argmin(diffs))
            if diffs[idx] <= 0.05 and idx not in matched_idx:
                matched_idx.append(idx)
        if len(matched_idx) == len(depth_levels):
            ds = ds.isel(depth=matched_idx)
    return ds


def stream_era5_daily(
    year: int,
    bbox: tuple[float, float, float, float],
    scratch_dir: Path,
) -> xr.Dataset:
    """Request and download ERA5 daily-statistics dataset from CDS into scratch_dir."""
    out_file = scratch_dir / f"era5_daily_{year}.nc"
    if out_file.exists() and out_file.stat().st_size > 0:
        ds = xr.open_dataset(out_file)
        ds.attrs["daily_statistic"] = "daily_mean"
        ds.attrs["is_daily_mean"] = True
        return ds

    try:
        import cdsapi
    except ImportError as e:
        raise RuntimeError("cdsapi package is required to stream ERA5 data from CDS.") from e

    lon_min, lon_max, lat_min, lat_max = bbox
    area = [lat_max + 1.0, lon_min - 1.0, lat_min - 1.0, lon_max + 1.0]

    c = cdsapi.Client()
    c.retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_net_solar_radiation",
                "surface_net_thermal_radiation",
                "surface_sensible_heat_flux",
                "surface_latent_heat_flux",
                "total_precipitation",
            ],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": area,
            "format": "netcdf",
        },
        str(out_file),
    )
    ds = xr.open_dataset(out_file)
    ds.attrs["daily_statistic"] = "daily_mean"
    ds.attrs["is_daily_mean"] = True
    return ds


def stream_oscar(
    year: int,
    bbox: tuple[float, float, float, float],
    scratch_dir: Path,
) -> xr.Dataset:
    """Download OSCAR ocean currents from NASA Earthdata into scratch_dir."""
    out_dir = scratch_dir / "oscar"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import earthaccess
    except ImportError as e:
        raise RuntimeError("earthaccess package is required to stream OSCAR currents.") from e

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name="OSCAR_L4_OC_FINAL_V2.0",
        temporal=(f"{year}-01-01", f"{year}-12-31"),
        # earthaccess expects (west, south, east, north); bbox is (lon_min, lon_max, lat_min, lat_max)
        bounding_box=(bbox[0], bbox[2], bbox[1], bbox[3]),
    )
    if not results:
        raise FileNotFoundError(f"No OSCAR granules found for year {year} in bbox {bbox}")
    downloaded = earthaccess.download(results, str(out_dir))
    if len(downloaded) == 1:
        return xr.open_dataset(downloaded[0])
    return xr.open_mfdataset(downloaded, combine="nested", concat_dim="time")


def stream_gebco(target_grid: TargetGrid, scratch_dir: Path) -> xr.Dataset:
    """Download regional GEBCO bathymetry into scratch_dir (cached across years)."""
    import urllib.request

    base_scratch = scratch_dir.parent if scratch_dir.name.isdigit() else scratch_dir
    base_scratch.mkdir(parents=True, exist_ok=True)
    out_file = base_scratch / "gebco_sub_ice_topo.nc"
    if out_file.exists() and out_file.stat().st_size > 0:
        return xr.open_dataset(out_file)

    ceda_url = (
        "https://data.ceda.ac.uk/bodc/gebco/global/gebco_2023/sub_ice_topo/netcdf/"
        "gebco_2023_sub_ice_topo.nc"
    )
    req = urllib.request.Request(ceda_url, headers={"User-Agent": "OceanScope-AI/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(out_file, "wb") as dst:
        shutil.copyfileobj(resp, dst, length=65536)
    return xr.open_dataset(out_file)


def stream_woa(
    target_grid: TargetGrid,
    depths_m: list[float],
    scratch_dir: Path | None = None,
) -> xr.DataArray:
    """Stream WOA23 0.25° monthly temperature climatology over OPeNDAP (cached across years)."""
    if scratch_dir is not None:
        base_scratch = scratch_dir.parent if scratch_dir.name.isdigit() else scratch_dir
        base_scratch.mkdir(parents=True, exist_ok=True)
        cache_file = base_scratch / "woa23_clim_profile.nc"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return xr.open_dataarray(cache_file)
    else:
        cache_file = None

    url_base = (
        "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/temperature/netcdf/decav/0.25"
    )
    woa_months = []
    for m in range(1, 13):
        url = f"{url_base}/woa23_decav_t{m:02d}_04.nc"
        ds_m_raw = xr.open_dataset(url, decode_times=False)
        ds_m = reduce_woa(ds_m_raw, target_grid, depths_m, month=m)
        woa_months.append(ds_m["woa_clim_profile"].values[0])
    woa_12m = np.stack(woa_months, axis=0)
    da_woa = xr.DataArray(
        woa_12m.astype(np.float32),
        dims=("month", "depth", "lat", "lon"),
        coords={
            "month": np.arange(1, 13),
            "depth": depths_m,
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )
    if cache_file is not None:
        try:
            da_woa.to_netcdf(cache_file)
        except Exception as e:
            logger.warning("Could not cache WOA23 DataArray to %s: %s", cache_file, e)
    return da_woa


def filter_files_by_year(files: list[Path], year: int) -> list[Path]:
    r"""Filter list of files to those matching the specified year in their filename or time range.

    Uses regex (?<!\d)(19|20)\d\d(?!\d) on f.name only. If a file has no year in its name,
    checks the file's time range coordinate. Returns [] when nothing matches.
    """
    import re

    pattern = re.compile(r"(?<!\d)(19|20)\d\d(?!\d)")
    yr_str = str(year)
    matched: list[Path] = []
    for f in files:
        name = f.name
        years_found = [m.group(0) for m in pattern.finditer(name)]
        if yr_str in years_found or re.search(r"(?<!\d)" + yr_str + r"(\d{2}|\d{4})?(?!\d)", name):
            matched.append(f)
        elif not years_found and not re.search(r"(?<!\d)(19|20)\d\d", name):
            # File has no 4-digit year in filename (e.g. data_stream-oper_stepType-instant.nc)
            # Check the file's time range coordinate after standardizing coordinates
            # (e.g. CDS ERA5 files often name time coordinate valid_time)
            try:
                with xr.open_dataset(f) as ds_raw:
                    ds = standardize_coords(ds_raw)
                    if "time" in ds.coords and len(ds.coords["time"]) > 0:
                        t_min = pd.Timestamp(ds.coords["time"].values[0])
                        t_max = pd.Timestamp(ds.coords["time"].values[-1])
                        if t_min.year <= year <= t_max.year:
                            matched.append(f)
            except Exception:
                pass
    return matched


def reduce_year_dataset(
    year: int,
    raw_path: Path,
    target_grid: TargetGrid,
    depths_m: list[float],
    manifest: dict[str, Any] | None = None,
    stream: bool = False,
    scratch_dir: Path | None = None,
    cached_bathymetry: xr.DataArray | None = None,
    cached_woa_profile: xr.DataArray | None = None,
) -> tuple[xr.Dataset, list[str]]:
    """Reduce all products for a given year using shared reduce functions from raw files.

    Strictly uses find_product_files (PRD §0 Rule 1, REVIEW A1); never synthesizes missing files.
    """
    source_files: list[str] = []
    processed_vars: dict[str, xr.DataArray] = {}

    bbox = (
        float(target_grid.lons.min()),
        float(target_grid.lons.max()),
        float(target_grid.lats.min()),
        float(target_grid.lats.max()),
    )
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # --- GLORYS12 Primary Target ---
    glorys_files = filter_files_by_year(find_product_files(raw_path, "glorys12", manifest), year)
    if glorys_files:
        source_files.append(
            str(
                glorys_files[0].relative_to(raw_path)
                if glorys_files[0].is_relative_to(raw_path)
                else glorys_files[0].name
            )
        )
        ds_raw_glorys = xr.open_dataset(glorys_files[0])
    elif stream:
        # Stream GLORYS12 via copernicusmarine.open_dataset (PRD §4.2, REVIEW_P3 R3)
        ds_raw_glorys = stream_copernicus(
            dataset_id=PRODUCT_SPECS["glorys12"]["dataset_id"],
            bbox=bbox,
            dates=(start_date, end_date),
            variables=["thetao"],
            depths=(float(min(depths_m)), float(max(depths_m))),
            depth_levels=depths_m,
        )
        source_files.append(f"copernicusmarine:{PRODUCT_SPECS['glorys12']['dataset_id']}:{year}")
    else:
        raise FileNotFoundError(f"Missing GLORYS12 raw data for year {year} in {raw_path}")

    ds_red_glorys = reduce_glorys12(ds_raw_glorys, target_grid, depths_m)
    processed_vars["thetao"] = ds_red_glorys["thetao"]
    processed_vars["ocean_mask"] = ds_red_glorys["ocean_mask"]
    processed_vars["target_valid"] = ds_red_glorys["target_valid"]

    common_time = ds_red_glorys.coords["time"]
    year_times = [t for t in common_time.values if pd.Timestamp(t).year == year]
    if year_times:
        target_dates = np.array(year_times)
        processed_vars["thetao"] = processed_vars["thetao"].sel(time=target_dates)
        processed_vars["target_valid"] = processed_vars["target_valid"].sel(time=target_dates)
    else:
        raise ValueError(
            f"GLORYS dataset does not contain dates for year {year}. Available years: "
            f"{sorted(list({pd.Timestamp(t).year for t in common_time.values}))}"
        )

    effective_scratch = scratch_dir or (raw_path / "scratch")
    effective_scratch.mkdir(parents=True, exist_ok=True)

    # --- SLA & ADT ---
    sla_files = filter_files_by_year(find_product_files(raw_path, "sla", manifest), year)
    if sla_files:
        source_files.append(
            str(
                sla_files[0].relative_to(raw_path)
                if sla_files[0].is_relative_to(raw_path)
                else sla_files[0].name
            )
        )
        ds_raw_sla = xr.open_dataset(sla_files[0])
    elif stream:
        ds_raw_sla = stream_copernicus(
            dataset_id=PRODUCT_SPECS["sla"]["dataset_id"],
            bbox=bbox,
            dates=(start_date, end_date),
            variables=["sla", "adt"],
        )
        source_files.append(f"copernicusmarine:{PRODUCT_SPECS['sla']['dataset_id']}:{year}")
    else:
        raise FileNotFoundError(f"Missing SLA/ADT raw data for year {year} in {raw_path}")

    ds_red_sla = reduce_sla(ds_raw_sla, target_grid, target_time=target_dates)
    processed_vars["sla"] = ds_red_sla["sla"]
    if "adt" in ds_raw_sla.data_vars:
        ds_red_adt = reduce_adt(ds_raw_sla, target_grid, target_time=target_dates)
        processed_vars["adt"] = ds_red_adt["adt"]

    # --- OSTIA SST ---
    sst_files = filter_files_by_year(find_product_files(raw_path, "sst", manifest), year)
    if sst_files:
        source_files.append(
            str(
                sst_files[0].relative_to(raw_path)
                if sst_files[0].is_relative_to(raw_path)
                else sst_files[0].name
            )
        )
        ds_raw_sst = xr.open_dataset(sst_files[0])
    elif stream:
        ds_raw_sst = stream_copernicus(
            dataset_id=PRODUCT_SPECS["sst"]["dataset_id"],
            bbox=bbox,
            dates=(start_date, end_date),
            variables=["analysed_sst"],
        )
        source_files.append(f"copernicusmarine:{PRODUCT_SPECS['sst']['dataset_id']}:{year}")
    else:
        raise FileNotFoundError(f"Missing OSTIA SST raw data for year {year} in {raw_path}")

    ds_red_sst = reduce_sst(ds_raw_sst, target_grid, target_time=target_dates)
    processed_vars["sst"] = ds_red_sst["sst"]

    # --- MULTIOBS SSS ---
    sss_files = filter_files_by_year(find_product_files(raw_path, "sss", manifest), year)
    if sss_files:
        source_files.append(
            str(
                sss_files[0].relative_to(raw_path)
                if sss_files[0].is_relative_to(raw_path)
                else sss_files[0].name
            )
        )
        ds_raw_sss = xr.open_dataset(sss_files[0])
    elif stream:
        ds_raw_sss = stream_copernicus(
            dataset_id=PRODUCT_SPECS["sss"]["dataset_id"],
            bbox=bbox,
            dates=(start_date, end_date),
            variables=["sos"],
        )
        source_files.append(f"copernicusmarine:{PRODUCT_SPECS['sss']['dataset_id']}:{year}")
    else:
        raise FileNotFoundError(f"Missing MULTIOBS SSS raw data for year {year} in {raw_path}")

    ds_red_sss = reduce_sss(ds_raw_sss, target_grid, target_time=target_dates)
    processed_vars["sss"] = ds_red_sss["sss"]

    # --- OSCAR Currents ---
    oscar_files = filter_files_by_year(find_product_files(raw_path, "oscar", manifest), year)
    if oscar_files:
        source_files.append(
            str(
                oscar_files[0].relative_to(raw_path)
                if oscar_files[0].is_relative_to(raw_path)
                else oscar_files[0].name
            )
        )
        if len(oscar_files) > 1:
            try:
                ds_raw_curr = xr.open_mfdataset(oscar_files, combine="nested", concat_dim="time")
            except Exception:
                ds_raw_curr = xr.open_dataset(oscar_files[0])
        else:
            ds_raw_curr = xr.open_dataset(oscar_files[0])
    elif stream:
        ds_raw_curr = stream_oscar(year, bbox, effective_scratch)
        source_files.append(f"podaac:OSCAR_L4_OC_FINAL_V2.0:{year}")
    else:
        raise FileNotFoundError(f"Missing OSCAR currents raw data for year {year} in {raw_path}")

    ds_red_curr = reduce_currents(ds_raw_curr, target_grid, target_time=target_dates)
    processed_vars["u_current"] = ds_red_curr["u_current"]
    processed_vars["v_current"] = ds_red_curr["v_current"]

    # --- ERA5 Winds & Fluxes ---
    winds_files = filter_files_by_year(find_product_files(raw_path, "era5_instant", manifest), year)
    flux_files = filter_files_by_year(find_product_files(raw_path, "era5_accum", manifest), year)
    if winds_files and flux_files:
        source_files.append(
            str(
                winds_files[0].relative_to(raw_path)
                if winds_files[0].is_relative_to(raw_path)
                else winds_files[0].name
            )
        )
        source_files.append(
            str(
                flux_files[0].relative_to(raw_path)
                if flux_files[0].is_relative_to(raw_path)
                else flux_files[0].name
            )
        )
        ds_raw_winds = xr.open_dataset(winds_files[0])
        ds_raw_flux = xr.open_dataset(flux_files[0])
    elif stream:
        ds_raw_era5 = stream_era5_daily(year, bbox, effective_scratch)
        source_files.append(f"cds:derived-era5-single-levels-daily-statistics:{year}")
        ds_raw_winds = ds_raw_era5
        ds_raw_flux = ds_raw_era5
    else:
        if not winds_files:
            raise FileNotFoundError(
                f"Missing ERA5 instant winds raw data for year {year} in {raw_path}"
            )
        raise FileNotFoundError(f"Missing ERA5 accum fluxes raw data for year {year} in {raw_path}")

    ds_red_winds = reduce_era5_winds(
        ds_raw_winds,
        target_grid,
        target_time=target_dates,
        ocean_mask=np.asarray(processed_vars["ocean_mask"].values, dtype=bool),
    )
    processed_vars["u_wind"] = ds_red_winds["u_wind"]
    processed_vars["v_wind"] = ds_red_winds["v_wind"]
    processed_vars["wind_stress_curl"] = ds_red_winds["wind_stress_curl"]

    ds_red_flux = reduce_era5_fluxes(ds_raw_flux, target_grid, target_time=target_dates)
    processed_vars["net_heat_flux"] = ds_red_flux["net_heat_flux"]
    processed_vars["precip"] = ds_red_flux["precip"]

    # --- GEBCO Bathymetry (Static, PRD §3) ---
    if cached_bathymetry is not None:
        processed_vars["bathymetry"] = cached_bathymetry
    else:
        gebco_files = find_product_files(raw_path, "gebco", manifest)
        if gebco_files:
            source_files.append(
                str(
                    gebco_files[0].relative_to(raw_path)
                    if gebco_files[0].is_relative_to(raw_path)
                    else gebco_files[0].name
                )
            )
            ds_red_gebco = reduce_gebco(xr.open_dataset(gebco_files[0]), target_grid)
            processed_vars["bathymetry"] = ds_red_gebco["bathymetry"]
        elif stream:
            try:
                ds_raw_gebco = stream_gebco(target_grid, effective_scratch)
                ds_red_gebco = reduce_gebco(ds_raw_gebco, target_grid)
                processed_vars["bathymetry"] = ds_red_gebco["bathymetry"]
                source_files.append("bodc:gebco_2023_sub_ice_topo")
            except Exception as e:
                raise RuntimeError(f"Failed to stream GEBCO bathymetry: {e}") from e
        else:
            raise FileNotFoundError(f"Missing GEBCO bathymetry raw data in {raw_path}")

    # --- WOA23 Climatology (Static by month, PRD §3, REVIEW B5) ---
    if cached_woa_profile is not None:
        processed_vars["woa_clim_profile"] = cached_woa_profile
    else:
        woa_files = find_product_files(raw_path, "woa23", manifest)
        if woa_files:
            source_files.append(
                str(
                    woa_files[0].relative_to(raw_path)
                    if woa_files[0].is_relative_to(raw_path)
                    else woa_files[0].name
                )
            )
            if len(woa_files) == 12:
                woa_months = []
                for m_idx, wf in enumerate(woa_files, start=1):
                    ds_m = reduce_woa(
                        xr.open_dataset(wf, decode_times=False),
                        target_grid,
                        depths_m,
                        month=m_idx,
                    )
                    woa_months.append(ds_m["woa_clim_profile"].values[0])
                woa_12m = np.stack(woa_months, axis=0)  # (12, depth, lat, lon)
            else:
                ds_red_woa = reduce_woa(
                    xr.open_dataset(woa_files[0], decode_times=False),
                    target_grid,
                    depths_m,
                    month=1,
                )
                woa_12m = np.repeat(ds_red_woa["woa_clim_profile"].values, 12, axis=0)

            processed_vars["woa_clim_profile"] = xr.DataArray(
                woa_12m.astype(np.float32),
                dims=("month", "depth", "lat", "lon"),
                coords={
                    "month": np.arange(1, 13),
                    "depth": depths_m,
                    "lat": target_grid.lats,
                    "lon": target_grid.lons,
                },
            )
        elif stream:
            try:
                processed_vars["woa_clim_profile"] = stream_woa(
                    target_grid, depths_m, effective_scratch
                )
                source_files.append("opendap:ncei.noaa.gov:woa23_decav_0.25")
            except Exception as e:
                raise RuntimeError(f"Failed to stream WOA23 dataset over OPeNDAP: {e}") from e
        else:
            raise FileNotFoundError(f"Missing WOA23 climatology raw data in {raw_path}")

    # --- GREP Ensemble Mean ---
    is_after_grep_cutoff = any(pd.Timestamp(t) > pd.Timestamp("2020-12-15") for t in target_dates)
    grep_files = filter_files_by_year(find_product_files(raw_path, "grep", manifest), year)
    if grep_files:
        source_files.append(
            str(
                grep_files[0].relative_to(raw_path)
                if grep_files[0].is_relative_to(raw_path)
                else grep_files[0].name
            )
        )
        ds_raw_grep = xr.open_dataset(grep_files[0])
    elif stream:
        try:
            # Request GREP down to >= 1250 m so 1062 m level is covered (PRD §4.2)
            ds_raw_grep = stream_copernicus(
                dataset_id=PRODUCT_SPECS["grep"]["dataset_id"],
                bbox=bbox,
                dates=(start_date, end_date),
                variables=["thetao_mean"],
                depths=(0.0, 1250.0),
            )
            source_files.append(f"copernicusmarine:{PRODUCT_SPECS['grep']['dataset_id']}:{year}")
        except Exception as e:
            if is_after_grep_cutoff:
                logger.warning("GREP unavailable after 2020-12-15 cutoff: %s", e)
                ds_raw_grep = None
            else:
                raise RuntimeError(f"Failed to stream GREP dataset for year {year}: {e}") from e
    else:
        if is_after_grep_cutoff:
            ds_raw_grep = None
        else:
            raise FileNotFoundError(f"Missing GREP raw data for year {year} in {raw_path}")

    if ds_raw_grep is not None:
        ds_red_grep = reduce_grep(ds_raw_grep, target_grid, depths_m)
        if "time" in ds_red_grep.dims:
            # Reindex to target_dates so partial-year coverage (e.g. GREP ending on 2020-12-15)
            # fills missing days with NaN without raising KeyError
            if len(ds_red_grep.coords["time"]) > 0:
                grep_dt = pd.to_datetime(ds_red_grep.coords["time"].values).floor("D")
                target_dt = pd.to_datetime(target_dates).floor("D")
                ds_red_grep = (
                    ds_red_grep.assign_coords(time=grep_dt)
                    .reindex(time=target_dt)
                    .assign_coords(time=target_dates)
                )
            else:
                ds_red_grep = ds_red_grep.reindex(time=target_dates)
        processed_vars["grep_mean"] = ds_red_grep["grep_mean"]

    # --- ARMOR3D 3D Temperature ---
    armor_files = filter_files_by_year(find_product_files(raw_path, "armor3d", manifest), year)
    if armor_files:
        source_files.append(
            str(
                armor_files[0].relative_to(raw_path)
                if armor_files[0].is_relative_to(raw_path)
                else armor_files[0].name
            )
        )
        ds_raw_armor = xr.open_dataset(armor_files[0])
    elif stream:
        try:
            ds_raw_armor = stream_copernicus(
                dataset_id=PRODUCT_SPECS["armor3d"]["dataset_id"],
                bbox=bbox,
                dates=(start_date, end_date),
                variables=["to"],
                depths=(0.0, 1100.0),
            )
            source_files.append(f"copernicusmarine:{PRODUCT_SPECS['armor3d']['dataset_id']}:{year}")
        except Exception as e:
            raise RuntimeError(f"Failed to stream ARMOR3D dataset for year {year}: {e}") from e
    else:
        raise FileNotFoundError(f"Missing ARMOR3D raw data for year {year} in {raw_path}")

    ds_red_armor = reduce_armor3d(ds_raw_armor, target_grid, depths_m)
    if "time" in ds_red_armor.dims:
        if len(ds_red_armor.coords["time"]) > 0:
            armor_dt = pd.to_datetime(ds_red_armor.coords["time"].values).floor("D")
            target_dt = pd.to_datetime(target_dates).floor("D")
            ds_red_armor = (
                ds_red_armor.assign_coords(time=armor_dt)
                .reindex(time=target_dt)
                .assign_coords(time=target_dates)
            )
        else:
            ds_red_armor = ds_red_armor.reindex(time=target_dates)
    processed_vars["armor3d"] = ds_red_armor["armor3d"]

    ds_year = xr.Dataset(data_vars=processed_vars)
    return ds_year, source_files


def export_f16_store(
    src_zarr: Path,
    dst_zarr: Path,
) -> Path:
    """Export memory-efficient float16 copy of processed store for training (PRD §6.1)."""
    logger.info("Opening master Zarr store from %s for float16 export...", src_zarr)
    ds = xr.open_zarr(src_zarr)

    dst_zarr.parent.mkdir(parents=True, exist_ok=True)
    f16_vars: dict[str, xr.DataArray] = {}

    for var_name, da in ds.data_vars.items():
        if da.dtype in (np.float32, np.float64):
            f16_vars[var_name] = da.astype(np.float16)
        else:
            f16_vars[var_name] = da

    ds_f16 = xr.Dataset(data_vars=f16_vars, coords=ds.coords, attrs=ds.attrs)
    ds_f16.attrs["dtype"] = "float16"
    logger.info("Writing float16 Zarr store to %s...", dst_zarr)
    ds_f16.to_zarr(dst_zarr, mode="w")
    return dst_zarr


def build_store(
    config: Any,
    years: list[int] | None = None,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    scratch_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    export_f16: bool = False,
    finalize: bool = False,
    stream: bool = False,
) -> Path:
    """Stream, reduce, and store multi-year ocean dataset one year at a time.

    Uses pre-allocated region writes and persists built_years directly on disk
    (PRD §4.2, REVIEW A3).
    """
    start_year = int(pd.Timestamp(config.time.start).year)
    end_year = int(pd.Timestamp(config.time.end).year)
    years_to_process = years if years is not None else list(range(start_year, end_year + 1))

    if dry_run:
        plan = plan_store_build(years_to_process, config)
        print_dry_run_plan(plan)
        return Path(out_dir or config.paths.processed) / f"oceanscope_{config.dataset_version}.zarr"

    out_base = Path(out_dir or config.paths.processed)
    out_base.mkdir(parents=True, exist_ok=True)
    zarr_dest = out_base / f"oceanscope_{config.dataset_version}.zarr"

    scratch_base = Path(scratch_dir or (out_base / "scratch"))
    scratch_base.mkdir(parents=True, exist_ok=True)

    target_grid = TargetGrid.from_bounds(
        lon_min=config.region.lon_min,
        lon_max=config.region.lon_max,
        lat_min=config.region.lat_min,
        lat_max=config.region.lat_max,
        resolution_deg=config.grid.resolution_deg,
    )
    depths_m: list[float] = [float(d) for d in config.depths_m]

    built_years: list[int] = []
    accumulated_sources: list[str] = []

    # Enforce budget before starting
    check_budget(scratch_base, zarr_dest)

    # 1. Determine all target dates across full period from config and requested years (PRD §4.2)
    start_time = pd.Timestamp(config.time.start)
    end_time = pd.Timestamp(config.time.end)
    min_year = min(years_to_process)
    max_year = max(years_to_process)
    store_start = min(start_time, pd.Timestamp(f"{min_year}-01-01"))
    store_end = max(end_time, pd.Timestamp(f"{max_year}-12-31"))
    all_dates = pd.date_range(store_start, store_end, freq="D")

    # 2. Open existing store or preallocate new store
    if zarr_dest.exists() and not force:
        try:
            root = zarr.open_group(str(zarr_dest), mode="r+")
            raw_built = root.attrs.get("built_years", [])
            if isinstance(raw_built, str):
                built_years = [int(y) for y in json.loads(raw_built)]
            elif isinstance(raw_built, (list, tuple)):
                built_years = [int(y) for y in raw_built]
            else:
                built_years = []

            raw_src = root.attrs.get("source_files", [])
            if isinstance(raw_src, str):
                accumulated_sources = [str(s) for s in json.loads(raw_src)]
            elif isinstance(raw_src, (list, tuple)):
                accumulated_sources = [str(s) for s in raw_src]
            else:
                accumulated_sources = []
            logger.info("Found existing store at %s with built_years=%s", zarr_dest, built_years)
            ds_existing = xr.open_zarr(zarr_dest)
            all_dates = pd.to_datetime(ds_existing.coords["time"].values)
        except Exception as e:
            logger.warning("Could not read attributes from existing store %s: %s", zarr_dest, e)

    raw_base = Path(raw_dir or config.paths.raw)
    manifest_path = raw_base / "MANIFEST.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning("Could not read MANIFEST.json: %s", e)

    # 3. Process each year
    cached_bathymetry: xr.DataArray | None = None
    cached_woa_profile: xr.DataArray | None = None

    for yr in years_to_process:
        if yr in built_years and not force:
            logger.info(
                "Year %d is already in built_years attribute. Skipping (use --force to rebuild).",
                yr,
            )
            continue

        yr_scratch = scratch_base / str(yr)
        yr_scratch.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Starting processing for year %d...", yr)
        logger.info("=" * 60)

        # In testing/local mode: reduce from raw_dir; in streaming mode: copernicusmarine
        ds_yr, src_files = reduce_year_dataset(
            yr,
            raw_base,
            target_grid,
            depths_m,
            manifest,
            stream=stream,
            scratch_dir=yr_scratch,
            cached_bathymetry=cached_bathymetry,
            cached_woa_profile=cached_woa_profile,
        )
        if cached_bathymetry is None and "bathymetry" in ds_yr.data_vars:
            cached_bathymetry = ds_yr["bathymetry"]
        if cached_woa_profile is None and "woa_clim_profile" in ds_yr.data_vars:
            cached_woa_profile = ds_yr["woa_clim_profile"]
        accumulated_sources.extend(src_files)

        yr_dates = pd.to_datetime(ds_yr.coords["time"].values)

        # If store does not exist yet or force requested, initialize full preallocated store
        if not zarr_dest.exists() or (force and len(built_years) == 0):
            logger.info(
                "Pre-allocating master Zarr store at %s across %d dates...",
                zarr_dest,
                len(all_dates),
            )

            import dask.array as da

            t_chunk = min(32, len(all_dates))
            n_t = len(all_dates)
            n_d = len(depths_m)
            n_lat = len(target_grid.lats)
            n_lon = len(target_grid.lons)

            data_vars: dict[str, Any] = {}
            # 2D surface variables (lazy dask arrays: 0 bytes RAM, REVIEW_P3 R3)
            for v in [
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
            ]:
                data_vars[v] = xr.DataArray(
                    da.full(
                        (n_t, n_lat, n_lon),
                        np.nan,
                        dtype=np.float32,
                        chunks=(t_chunk, n_lat, n_lon),
                    ),
                    dims=("time", "lat", "lon"),
                )

            # 4D subsurface variables (lazy dask arrays: 0 bytes RAM, REVIEW_P3 R3)
            for v in ["thetao", "grep_mean", "armor3d"]:
                data_vars[v] = xr.DataArray(
                    da.full(
                        (n_t, n_d, n_lat, n_lon),
                        np.nan,
                        dtype=np.float32,
                        chunks=(t_chunk, n_d, n_lat, n_lon),
                    ),
                    dims=("time", "depth", "lat", "lon"),
                )
            data_vars["target_valid"] = xr.DataArray(
                da.zeros(
                    (n_t, n_d, n_lat, n_lon),
                    dtype=bool,
                    chunks=(t_chunk, n_d, n_lat, n_lon),
                ),
                dims=("time", "depth", "lat", "lon"),
            )

            # Static variables
            if "ocean_mask" in ds_yr:
                data_vars["ocean_mask"] = ds_yr["ocean_mask"]
            if "bathymetry" in ds_yr:
                data_vars["bathymetry"] = ds_yr["bathymetry"]
            if "woa_clim_profile" in ds_yr:
                data_vars["woa_clim_profile"] = ds_yr["woa_clim_profile"]

            template_ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": all_dates,
                    "depth": depths_m,
                    "lat": target_grid.lats,
                    "lon": target_grid.lons,
                },
            )
            template_ds.attrs["dataset_version"] = str(config.dataset_version)
            template_ds.attrs["config_hash"] = compute_config_hash(config.raw_dict)
            template_ds.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            template_ds.attrs["built_years"] = []
            template_ds.attrs["source_files"] = []

            # Write empty Zarr template metadata without computing large in-memory arrays
            template_ds.to_zarr(zarr_dest, mode="w", compute=False)

        # Find time slice index in all_dates
        t_mask = np.isin(all_dates, yr_dates)
        indices = np.where(t_mask)[0]
        if len(indices) == 0:
            raise ValueError(f"Year {yr} dates do not overlap preallocated store dates.")
        i0, i1 = int(indices[0]), int(indices[-1]) + 1

        # Drop non-time coordinates from region dataset per xarray region write specification
        vars_to_write = {
            k: v.reset_coords(drop=True) for k, v in ds_yr.data_vars.items() if "time" in v.dims
        }
        ds_region = xr.Dataset(data_vars=vars_to_write, coords={"time": yr_dates})
        non_time_coords = [c for c in ds_region.coords if c != "time"]
        if non_time_coords:
            ds_region = ds_region.drop_vars(non_time_coords)
        ds_region.to_zarr(zarr_dest, region={"time": slice(i0, i1)})

        # Immediately update root attributes on disk so interruptions preserve progress (A3)
        built_years.append(yr)
        root = zarr.open_group(str(zarr_dest), mode="r+")
        root.attrs["built_years"] = sorted(list(set(built_years)))
        root.attrs["source_files"] = sorted(list(set(accumulated_sources)))
        logger.info(
            "Persisted year %d to built_years. Total built: %s", yr, root.attrs["built_years"]
        )

        # Check budget and clean scratch
        if yr_scratch.exists():
            shutil.rmtree(yr_scratch, ignore_errors=True)
        check_budget(scratch_base, zarr_dest)

    # 4. Finalize step: Climatology and Norm Stats (A4)
    # Only run if finalize=True or if train split is covered
    if finalize or set(built_years) == set(years_to_process):
        ds_master = xr.open_zarr(zarr_dest)
        master_dates = pd.to_datetime(ds_master.coords["time"].values)
        train_start = pd.Timestamp(config.splits.train[0])
        train_end = pd.Timestamp(config.splits.train[1])
        has_train_dates = np.any((master_dates >= train_start) & (master_dates <= train_end))

        if has_train_dates:
            logger.info(
                "Computing harmonic climatology and normalization statistics on training split..."
            )
            check_no_test_leakage(config.splits.train, config.splits.test)

            t_3d = ds_master["thetao"].sel(time=slice(train_start, train_end))
            train_times = pd.to_datetime(t_3d.coords["time"].values)

            clim_366 = fit_harmonic_climatology(
                t_3d,
                train_times,
                n_harmonics=3,
                train_split=config.splits.train,
                test_split=config.splits.test,
            )

            ds_clim_da = xr.DataArray(
                clim_366.astype(np.float32),
                dims=("dayofyear", "depth", "lat", "lon"),
                coords={
                    "dayofyear": np.arange(1, 367),
                    "depth": depths_m,
                    "lat": target_grid.lats,
                    "lon": target_grid.lons,
                },
            )
            root = zarr.open_group(str(zarr_dest), mode="r+")
            existing_attrs = dict(root.attrs)

            ds_clim_ds = ds_clim_da.to_dataset(name="clim")
            ds_clim_ds.attrs = existing_attrs
            ds_clim_ds.to_zarr(zarr_dest, mode="a")

            norm_stats = compute_norm_stats(
                ds_master,
                train_split=config.splits.train,
                test_split=config.splits.test,
            )
            root = zarr.open_group(str(zarr_dest), mode="r+")
            root.attrs.update(existing_attrs)
            root.attrs["built_years"] = sorted(list(set(built_years)))
            root.attrs["norm_stats"] = norm_stats
            root.attrs["split_dates"] = {
                "train": config.splits.train,
                "val": config.splits.val,
                "test": config.splits.test,
            }
            logger.info("Climatology and normalization statistics successfully computed and saved.")
        else:
            logger.info(
                "Store contains no training dates; marking sample_only=True "
                "and skipping train climatology."
            )
            root = zarr.open_group(str(zarr_dest), mode="r+")
            root.attrs["sample_only"] = True
            root.attrs["built_years"] = sorted(list(set(built_years)))

    if export_f16:
        f16_dest = out_base / f"oceanscope_{config.dataset_version}_f16.zarr"
        export_f16_store(zarr_dest, f16_dest)

    return zarr_dest


def main() -> None:
    """CLI entry point for oceanscope_ml.data.build_store."""
    parser = argparse.ArgumentParser(
        description="Stream, reduce, and build multi-year OceanScope Zarr store (PRD_AI_ML §4.2)"
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated or hyphenated years (e.g. '2018,2019' or '1993-2021')",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional local raw data directory for development/testing",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output destination directory for processed Zarr store",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Temporary scratch directory (disposable, <= 20 GB)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned requests and size estimates without downloading or writing data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild years even if already present in built_years attribute",
    )
    parser.add_argument(
        "--f16",
        "--fp16",
        dest="f16",
        action="store_true",
        help="Export additional float16 store (oceanscope_<v>_f16.zarr) for training upload",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Compute train-period climatology and normalization statistics",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream remote products (Copernicus, CDS, etc.) instead of reading local raw files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    cfg = load_config(args.config)
    target_years = parse_year_spec(args.years) if args.years else None

    build_store(
        config=cfg,
        years=target_years,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        scratch_dir=Path(args.scratch_dir) if args.scratch_dir else None,
        dry_run=args.dry_run,
        force=args.force,
        export_f16=args.f16,
        finalize=args.finalize,
        stream=args.stream,
    )
    if not args.dry_run:
        print("\n[OK] build_store complete. Processed store ready.")


if __name__ == "__main__":
    main()
