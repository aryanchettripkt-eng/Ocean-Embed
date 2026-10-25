"""Data preprocessing stage for OceanScope AI (PRD_AI_ML §4, §4.2, §6.1).

Processes raw products into the unified processed Zarr store:
data/processed/oceanscope_<dataset_version>.zarr
and generates outputs/figures/m1_sample_day.png for Milestone 1 acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.climatology import fit_harmonic_climatology
from oceanscope_ml.data.plot import plot_sample_day
from oceanscope_ml.data.reduce import (
    TargetGrid,
    reduce_adt,
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
)
from oceanscope_ml.data.transform import compute_norm_stats

logger = logging.getLogger("oceanscope_ml.data.preprocess")


def compute_config_hash(raw_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of experiment configuration."""
    clean_dict = {k: v for k, v in raw_dict.items() if k not in ["paths"]}
    encoded = json.dumps(clean_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def generate_synthetic_raw_sample(
    raw_dir: Path,
    target_grid: TargetGrid,
    depths_m: list[float],
    dates: pd.DatetimeIndex,
) -> list[str]:
    """Generate realistic synthetic NetCDF files in raw_dir for offline testing and verification.

    Returns list of generated file relative paths.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    n_time = len(dates)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    lats = target_grid.lats
    lons = target_grid.lons

    # Create land mask for realistic geography
    lat_2d, lon_2d = np.meshgrid(lats, lons, indexing="ij")
    is_arabia = (lat_2d >= 13.0) & (lon_2d <= 56.0)
    is_india = (
        (lat_2d >= 8.0) & (lon_2d >= 69.0) & (lon_2d <= 87.0) & (lat_2d + 0.5 * lon_2d > 48.0)
    )
    is_africa = (lat_2d <= 12.0) & (lon_2d <= 51.0)
    is_burma = (lat_2d >= 16.0) & (lon_2d >= 92.0)
    ocean_mask_2d = ~(is_arabia | is_india | is_africa | is_burma)

    yr_str = str(dates[0].year)

    # 1. GLORYS12: (time, depth, lat, lon)
    glorys_dir = raw_dir / "glorys"
    glorys_dir.mkdir(parents=True, exist_ok=True)
    glorys_path = glorys_dir / f"glorys12_{yr_str}01.nc"

    n_depth = len(depths_m)
    t_3d = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
    depth_arr = np.array(depths_m, dtype=np.float32)

    for t_idx in range(n_time):
        doy = (
            dates[t_idx].day_of_year
            if hasattr(dates[t_idx], "day_of_year")
            else dates[t_idx].dayofyear
        )
        annual_cycle = 1.8 * np.sin(2.0 * np.pi * (doy - 30) / 365.25)
        for i_lat, lat in enumerate(lats):
            surf_t = 28.5 - 0.15 * lat + annual_cycle
            for i_lon, lon in enumerate(lons):
                if ocean_mask_2d[i_lat, i_lon]:
                    t_profile = 4.0 + (surf_t - 4.0) * np.exp(-depth_arr / 220.0)
                    t_3d[t_idx, :, i_lat, i_lon] = t_profile.astype(np.float32)

    ds_glorys = xr.Dataset(
        data_vars={"thetao": (("time", "depth", "latitude", "longitude"), t_3d)},
        coords={"time": dates, "depth": depth_arr, "latitude": lats, "longitude": lons},
    )
    ds_glorys.to_netcdf(glorys_path)
    generated_files.append(str(glorys_path.relative_to(raw_dir)))

    # 2. SLA & ADT
    duacs_dir = raw_dir / "duacs"
    duacs_dir.mkdir(parents=True, exist_ok=True)
    sla_path = duacs_dir / f"duacs_sla_{yr_str}01.nc"
    adt_path = duacs_dir / f"duacs_adt_{yr_str}01.nc"

    sla_vals = np.where(
        ocean_mask_2d, 0.05 * np.sin(lon_2d / 5.0) * np.cos(lat_2d / 4.0), np.nan
    ).astype(np.float32)
    sla_3d = np.tile(sla_vals, (n_time, 1, 1))
    adt_3d = sla_3d + 1.2

    ds_sla = xr.Dataset(
        data_vars={"sla": (("time", "lat", "lon"), sla_3d)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_sla.to_netcdf(sla_path)
    generated_files.append(str(sla_path.relative_to(raw_dir)))

    ds_adt = xr.Dataset(
        data_vars={"adt": (("time", "lat", "lon"), adt_3d)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_adt.to_netcdf(adt_path)
    generated_files.append(str(adt_path.relative_to(raw_dir)))

    # 3. OSTIA SST (stored in Kelvin)
    ostia_dir = raw_dir / "ostia"
    ostia_dir.mkdir(parents=True, exist_ok=True)
    sst_path = ostia_dir / f"ostia_sst_{yr_str}01.nc"

    sst_celsius = t_3d[:, 0, :, :]
    sst_kelvin = np.where(~np.isnan(sst_celsius), sst_celsius + 273.15, np.nan).astype(np.float32)
    ds_sst = xr.Dataset(
        data_vars={"analysed_sst": (("time", "lat", "lon"), sst_kelvin)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_sst.to_netcdf(sst_path)
    generated_files.append(str(sst_path.relative_to(raw_dir)))

    # 4. SSS
    sss_dir = raw_dir / "sss"
    sss_dir.mkdir(parents=True, exist_ok=True)
    sss_path = sss_dir / f"multiobs_sss_{yr_str}01.nc"

    sss_vals = np.where(ocean_mask_2d, 35.0 - 0.05 * lat_2d + 0.02 * lon_2d, np.nan).astype(
        np.float32
    )
    sss_3d = np.tile(sss_vals, (n_time, 1, 1))
    ds_sss = xr.Dataset(
        data_vars={"sos": (("time", "lat", "lon"), sss_3d)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_sss.to_netcdf(sss_path)
    generated_files.append(str(sss_path.relative_to(raw_dir)))

    # 5. OSCAR Currents
    oscar_dir = raw_dir / "oscar"
    oscar_dir.mkdir(parents=True, exist_ok=True)
    oscar_path = oscar_dir / f"oscar_currents_{yr_str}01.nc"

    u_curr = np.where(ocean_mask_2d, 0.2 * np.cos(lat_2d / 3.0), np.nan).astype(np.float32)
    v_curr = np.where(ocean_mask_2d, 0.1 * np.sin(lon_2d / 4.0), np.nan).astype(np.float32)
    ds_oscar = xr.Dataset(
        data_vars={
            "u": (("time", "lat", "lon"), np.tile(u_curr, (n_time, 1, 1))),
            "v": (("time", "lat", "lon"), np.tile(v_curr, (n_time, 1, 1))),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_oscar.to_netcdf(oscar_path)
    generated_files.append(str(oscar_path.relative_to(raw_dir)))

    # 6. ERA5 Winds
    era5_dir = raw_dir / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    winds_path = era5_dir / f"era5_winds_{yr_str}01.nc"

    u_wind = (5.0 + 2.0 * np.cos(lat_2d / 4.0)).astype(np.float32)
    v_wind = (-3.0 + 1.5 * np.sin(lon_2d / 6.0)).astype(np.float32)
    ds_winds = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.tile(u_wind, (n_time, 1, 1))),
            "v10": (("time", "lat", "lon"), np.tile(v_wind, (n_time, 1, 1))),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_winds.to_netcdf(winds_path)
    generated_files.append(str(winds_path.relative_to(raw_dir)))

    # 7. ERA5 Fluxes & Precip
    fluxes_path = era5_dir / f"era5_fluxes_{yr_str}01.nc"
    net_heat = np.full((n_time, n_lat, n_lon), 40.0, dtype=np.float32)
    precip = np.full((n_time, n_lat, n_lon), 0.003, dtype=np.float32)
    ds_fluxes = xr.Dataset(
        data_vars={
            "net_heat_flux": (("time", "lat", "lon"), net_heat),
            "precip": (("time", "lat", "lon"), precip),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    ds_fluxes.to_netcdf(fluxes_path)
    generated_files.append(str(fluxes_path.relative_to(raw_dir)))

    # 8. GEBCO Bathymetry
    gebco_dir = raw_dir / "gebco"
    gebco_dir.mkdir(parents=True, exist_ok=True)
    gebco_path = gebco_dir / "gebco_bathymetry.nc"

    elev = np.where(ocean_mask_2d, -3200.0, 500.0).astype(np.float32)
    ds_gebco = xr.Dataset(
        data_vars={"elevation": (("lat", "lon"), elev)}, coords={"lat": lats, "lon": lons}
    )
    ds_gebco.to_netcdf(gebco_path)
    generated_files.append(str(gebco_path.relative_to(raw_dir)))

    return generated_files


def run_preprocessing(
    config: ExperimentConfig,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    sample_day: str | None = None,
    use_sample_data: bool = False,
) -> xr.Dataset:
    """Execute complete data preprocessing pipeline for OceanScope AI.

    Parameters
    ----------
    config : ExperimentConfig loaded from experiment.yaml
    raw_dir : path to raw data directory
    out_dir : path to processed data output directory
    sample_day : date string for m1_sample_day.png
    use_sample_data : if True or raw dir is empty, generates synthetic sample data

    Returns
    -------
    xr.Dataset matching PRD_AI_ML Section 6.1
    """
    raw_path = raw_dir or config.paths.raw
    out_path = out_dir or config.paths.processed
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Define target grid from config
    target_grid = TargetGrid.from_bounds(
        lon_min=config.region.lon_min,
        lon_max=config.region.lon_max,
        lat_min=config.region.lat_min,
        lat_max=config.region.lat_max,
        resolution_deg=config.grid.resolution_deg,
    )
    depths_m = config.depths_m
    if len(depths_m) != 15:
        raise ValueError(
            f"Config depths_m has {len(depths_m)} values, expected exactly 15 "
            "(PRD_AI_ML §2, DECISIONS 2026-10-24)."
        )

    # 2. Check for raw files or synthesize sample
    source_files: list[str] = []
    raw_files = list(raw_path.glob("**/*.nc")) + list(raw_path.glob("**/*.nc4"))

    if len(raw_files) == 0 or use_sample_data:
        logger.info(
            "No raw NetCDF files found in %s; generating synthetic sample dataset...", raw_path
        )
        dates = pd.date_range("2019-01-01", "2019-01-31", freq="D")
        source_files = generate_synthetic_raw_sample(raw_path, target_grid, depths_m, dates)
    else:
        source_files = [str(f.relative_to(raw_path)) for f in raw_files]

    logger.info(
        "Processing %d source files against target grid %s...", len(source_files), target_grid
    )

    # 3. Locate product files and apply shared reduce functions
    processed_vars: dict[str, xr.DataArray] = {}

    # --- GLORYS12 ---
    logger.info("Step 1/10: Reducing GLORYS12 thetao and deriving ocean_mask...")
    glorys_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "glorys" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
        reverse=True,  # prefer full multi-depth dataset
    )
    if glorys_candidates:
        ds_raw_glorys = xr.open_dataset(glorys_candidates[0])
        ds_red_glorys = reduce_glorys12(ds_raw_glorys, target_grid, depths_m)
        processed_vars["thetao"] = ds_red_glorys["thetao"]
        processed_vars["ocean_mask"] = ds_red_glorys["ocean_mask"]
        processed_vars["target_valid"] = ds_red_glorys["target_valid"]
        common_time = ds_red_glorys.coords["time"]
    else:
        raise FileNotFoundError(f"Missing GLORYS12 raw file in {raw_path}")

    # --- SLA & ADT ---
    logger.info("Step 2/10: Reducing DUACS SLA & ADT...")
    sla_candidates = [
        f
        for f in raw_path.glob("**/*")
        if ("sla" in f.name.lower() or "duacs" in f.name.lower()) and f.suffix in [".nc", ".nc4"]
    ]
    if sla_candidates:
        ds_raw_sla = xr.open_dataset(sla_candidates[0])
        ds_red_sla = reduce_sla(ds_raw_sla, target_grid, target_time=common_time.values)
        processed_vars["sla"] = ds_red_sla["sla"]
        if "adt" in ds_raw_sla.data_vars:
            ds_red_adt = reduce_adt(ds_raw_sla, target_grid, target_time=common_time.values)
            processed_vars["adt"] = ds_red_adt["adt"]

    if "adt" not in processed_vars:
        adt_candidates = [
            f
            for f in raw_path.glob("**/*")
            if "adt" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ]
        if adt_candidates:
            ds_red_adt = reduce_adt(
                xr.open_dataset(adt_candidates[0]), target_grid, target_time=common_time.values
            )
            processed_vars["adt"] = ds_red_adt["adt"]

    # --- OSTIA SST ---
    logger.info("Step 3/10: Reducing OSTIA SST...")
    sst_candidates = [
        f for f in raw_path.glob("**/*") if "sst" in f.name.lower() and f.suffix in [".nc", ".nc4"]
    ]
    if sst_candidates:
        ds_red_sst = reduce_sst(
            xr.open_dataset(sst_candidates[0]), target_grid, target_time=common_time.values
        )
        processed_vars["sst"] = ds_red_sst["sst"]

    # --- MULTIOBS SSS ---
    logger.info("Step 4/10: Reducing MULTIOBS SSS...")
    sss_candidates = [
        f
        for f in raw_path.glob("**/*")
        if ("sss" in f.name.lower() or "sos" in f.name.lower()) and f.suffix in [".nc", ".nc4"]
    ]
    if sss_candidates:
        ds_red_sss = reduce_sss(
            xr.open_dataset(sss_candidates[0]), target_grid, target_time=common_time.values
        )
        processed_vars["sss"] = ds_red_sss["sss"]

    # --- OSCAR Currents ---
    logger.info("Step 5/10: Reducing OSCAR currents...")
    oscar_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if ("oscar" in f.name.lower() or "current" in f.name.lower())
            and f.suffix in [".nc", ".nc4"]
        ]
    )
    if oscar_candidates:
        if len(oscar_candidates) > 1 and any("final" in f.name for f in oscar_candidates):
            # Find files matching common_time dates
            date_strs = [pd.Timestamp(t).strftime("%Y%m%d") for t in common_time.values]
            matched = [f for f in oscar_candidates if any(d in f.name for d in date_strs)]
            files_to_open = matched if matched else [oscar_candidates[0]]
            try:
                ds_raw_curr = xr.open_mfdataset(files_to_open, combine="nested", concat_dim="time")
            except Exception:
                ds_raw_curr = xr.open_dataset(files_to_open[0])
        else:
            ds_raw_curr = xr.open_dataset(oscar_candidates[0])
        ds_red_curr = reduce_currents(ds_raw_curr, target_grid, target_time=common_time.values)
        processed_vars["u_current"] = ds_red_curr["u_current"]
        processed_vars["v_current"] = ds_red_curr["v_current"]

    # --- ERA5 Winds & Fluxes ---
    logger.info("Step 6/10: Reducing ERA5 winds & fluxes...")
    era5_zip_candidates = [
        f for f in raw_path.glob("**/*") if "era5" in f.name.lower() and f.suffix in [".nc", ".zip"]
    ]
    instant_p: Path | None = None
    accum_p: Path | None = None

    for f_cand in era5_zip_candidates:
        inst = f_cand.parent / "data_stream-oper_stepType-instant.nc"
        acc = f_cand.parent / "data_stream-oper_stepType-accum.nc"
        if inst.exists() and acc.exists():
            instant_p = inst
            accum_p = acc
            break
        try:
            with open(f_cand, "rb") as check_f:
                magic = check_f.read(4)
            if magic == b"PK\x03\x04":
                import zipfile

                with zipfile.ZipFile(f_cand, "r") as z:
                    z.extractall(f_cand.parent)
                if inst.exists():
                    instant_p = inst
                if acc.exists():
                    accum_p = acc
                break
        except Exception:
            pass

    winds_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "wind" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    if winds_cands:
        ds_winds = xr.open_dataset(winds_cands[0])
    elif instant_p is not None:
        ds_winds = xr.open_dataset(instant_p)
    else:
        ds_winds = None

    if ds_winds is not None:
        ds_red_winds = reduce_era5_winds(ds_winds, target_grid, target_time=common_time.values)
        processed_vars["u_wind"] = ds_red_winds["u_wind"]
        processed_vars["v_wind"] = ds_red_winds["v_wind"]
        processed_vars["wind_stress_curl"] = ds_red_winds["wind_stress_curl"]

    flux_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "flux" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    if flux_cands:
        ds_flux = xr.open_dataset(flux_cands[0])
    elif accum_p is not None:
        ds_flux = xr.open_dataset(accum_p)
    else:
        ds_flux = None

    if ds_flux is not None:
        ds_red_flux = reduce_era5_fluxes(ds_flux, target_grid, target_time=common_time.values)
        processed_vars["net_heat_flux"] = ds_red_flux["net_heat_flux"]
        processed_vars["precip"] = ds_red_flux["precip"]

    # --- GEBCO Bathymetry ---
    logger.info("Step 7/10: Reducing GEBCO bathymetry...")
    gebco_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "gebco" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,  # smaller/pre-cropped file first
    )
    if gebco_candidates:
        ds_raw_gebco = xr.open_dataset(gebco_candidates[0])
        # If massive global grid, subset bounds first
        if "lat" in ds_raw_gebco.coords and len(ds_raw_gebco.coords["lat"]) > 1000:
            ds_raw_gebco = ds_raw_gebco.sel(
                lat=slice(config.region.lat_min - 1.0, config.region.lat_max + 1.0),
                lon=slice(config.region.lon_min - 1.0, config.region.lon_max + 1.0),
            )
        ds_red_gebco = reduce_gebco(ds_raw_gebco, target_grid)
        processed_vars["bathymetry"] = ds_red_gebco["bathymetry"]

    # --- WOA23 Monthly Climatology (Optional) ---
    logger.info("Step 8/10: Reducing WOA23 climatological profile...")
    woa_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "woa" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ]
    )
    if woa_candidates:
        ds_red_woa = reduce_woa(
            xr.open_dataset(woa_candidates[0], decode_times=False), target_grid, depths_m
        )
        processed_vars["woa_clim_profile"] = ds_red_woa["woa_clim_profile"]

    # --- GREP Ensemble Mean (Optional) ---
    logger.info("Step 9/10: Reducing GREP ensemble mean...")
    grep_candidates = [
        f for f in raw_path.glob("**/*") if "grep" in f.name.lower() and f.suffix in [".nc", ".nc4"]
    ]
    if grep_candidates:
        ds_red_grep = reduce_grep(xr.open_dataset(grep_candidates[0]), target_grid, depths_m)
        processed_vars["grep_mean"] = ds_red_grep["grep_mean"]

    # 4. Harmonic Climatology: compute on train dates (PRD_AI_ML §0 rule 4, §5.2)
    logger.info("Step 10/10: Fitting harmonic climatology (mean + 3 harmonics)...")
    time_coords = common_time.values
    t_3d = processed_vars["thetao"].values  # (time, depth, lat, lon)

    # For 1-month development sample, dates fall in 2019 (within train or val range)
    clim_366 = fit_harmonic_climatology(t_3d, time_coords, n_harmonics=3)

    # Build clim DataArray with dayofyear coordinate 1..366
    processed_vars["clim"] = xr.DataArray(
        clim_366,
        dims=("dayofyear", "depth", "lat", "lon"),
        coords={
            "dayofyear": np.arange(1, 367),
            "depth": np.array(depths_m, dtype=np.float64),
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )

    # 5. Assemble Dataset
    ds_processed = xr.Dataset(
        data_vars=processed_vars,
        coords={
            "time": common_time,
            "depth": np.array(depths_m, dtype=np.float64),
            "lat": target_grid.lats,
            "lon": target_grid.lons,
            "dayofyear": np.arange(1, 367),
        },
    )

    # 6. Compute Normalization Statistics (Train-only; PRD_AI_ML §0 rule 4, §6.1)
    # Check if dataset dates fall into train split; if sample dates are e.g. Jan 2019,
    # use sample dates as train slice for this M1 run without test leakage.
    time_series = pd.to_datetime(common_time.values)
    min_date = time_series.min().strftime("%Y-%m-%d")
    max_date = time_series.max().strftime("%Y-%m-%d")

    # If dataset spans only sample month, use sample date range as train slice for M1
    train_range = (
        config.splits.train
        if (
            pd.Timestamp(min_date) >= pd.Timestamp(config.splits.train[0])
            and pd.Timestamp(max_date) <= pd.Timestamp(config.splits.train[1])
        )
        else [min_date, max_date]
    )

    norm_stats = compute_norm_stats(
        ds_processed,
        train_split=train_range,
        test_split=config.splits.test,
    )

    # 7. Root Attributes (PRD_AI_ML §6.1)
    ds_processed.attrs["dataset_version"] = str(config.dataset_version)
    ds_processed.attrs["config_hash"] = compute_config_hash(config.raw_dict)
    ds_processed.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    ds_processed.attrs["source_files"] = json.dumps(source_files)
    ds_processed.attrs["norm_stats"] = json.dumps(norm_stats)
    ds_processed.attrs["split_dates"] = json.dumps(
        {
            "train": config.splits.train,
            "val": config.splits.val,
            "test": config.splits.test,
        }
    )

    # 8. Save Processed Zarr Store
    zarr_filename = f"oceanscope_{config.dataset_version}.zarr"
    zarr_dest = out_path / zarr_filename
    logger.info("Writing processed Zarr store to %s...", zarr_dest)
    ds_processed.to_zarr(zarr_dest, mode="w")

    # 9. Generate Milestone 1 Sample Day Figure (PRD_AI_ML M1)
    chosen_day = sample_day or min_date
    fig_dest = config.paths.outputs / "figures" / "m1_sample_day.png"
    logger.info("Generating sample day figure for %s to %s...", chosen_day, fig_dest)
    plot_sample_day(ds_processed, chosen_day, fig_dest)

    return ds_processed


def main() -> None:
    """CLI entry point for oceanscope_ml.data.preprocess."""
    parser = argparse.ArgumentParser(
        description="OceanScope AI data preprocessing and reduction (PRD_AI_ML §4, §4.2, §6.1)"
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment configuration YAML",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Path to raw data directory (overrides config)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Path to processed store output directory (overrides config)",
    )
    parser.add_argument(
        "--sample-day",
        default=None,
        help="Date to plot in outputs/figures/m1_sample_day.png (default: first available date)",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Generate synthetic raw sample data if raw files are missing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    cfg = load_config(args.config)
    run_preprocessing(
        config=cfg,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        sample_day=args.sample_day,
        use_sample_data=args.sample_data,
    )
    print("\n[OK] Preprocessing complete. Processed store and m1_sample_day.png generated.")


if __name__ == "__main__":
    main()
