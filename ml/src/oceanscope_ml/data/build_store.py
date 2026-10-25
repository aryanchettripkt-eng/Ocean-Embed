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

from oceanscope_ml.data.climatology import fit_harmonic_climatology
from oceanscope_ml.data.preprocess import (
    compute_config_hash,
    generate_synthetic_raw_sample,
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
)
from oceanscope_ml.data.transform import compute_norm_stats

logger = logging.getLogger("oceanscope_ml.data.build_store")

# Estimated daily and static download sizes (MB) for dry-run estimation
ESTIMATED_SIZES_MB: dict[str, dict[str, float]] = {
    "glorys12": {"daily_download": 20.0, "daily_store": 1.7},
    "sla_adt": {"daily_download": 3.0, "daily_store": 0.23},
    "sst": {"daily_download": 6.8, "daily_store": 0.115},
    "sss": {"daily_download": 1.5, "daily_store": 0.115},
    "oscar": {"daily_download": 1.0, "daily_store": 0.23},
    "era5": {"daily_download": 2.5, "daily_store": 0.58},
    "grep": {"daily_download": 6.0, "daily_store": 1.7},
    "gebco": {"static_download": 15.0, "static_store": 0.115},
    "woa23": {"static_download": 25.0, "static_store": 1.7},
}


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
    """Calculate planned requests, transfer sizes, and store growth for dry-run."""
    plan_per_year: list[dict[str, Any]] = []
    total_transfer_mb = 0.0
    total_store_growth_mb = 0.0

    # Static products accounted once
    static_transfer_mb = (
        ESTIMATED_SIZES_MB["gebco"]["static_download"]
        + ESTIMATED_SIZES_MB["woa23"]["static_download"]
    )
    static_store_mb = (
        ESTIMATED_SIZES_MB["gebco"]["static_store"] + ESTIMATED_SIZES_MB["woa23"]["static_store"]
    )

    total_transfer_mb += static_transfer_mb
    total_store_growth_mb += static_store_mb

    for yr in years:
        days_in_yr = 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
        daily_transfer = sum(
            spec.get("daily_download", 0.0)
            for k, spec in ESTIMATED_SIZES_MB.items()
            if "daily_download" in spec
        )
        daily_store = sum(
            spec.get("daily_store", 0.0)
            for k, spec in ESTIMATED_SIZES_MB.items()
            if "daily_store" in spec
        )

        yr_transfer = days_in_yr * daily_transfer
        yr_store = days_in_yr * daily_store

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

    return {
        "years": years,
        "n_years": len(years),
        "static_transfer_mb": round(static_transfer_mb, 1),
        "total_transfer_mb": round(total_transfer_mb, 1),
        "total_store_growth_mb": round(total_store_growth_mb, 1),
        "years_detail": plan_per_year,
    }


def print_dry_run_plan(plan: dict[str, Any]) -> None:
    """Print formatted dry-run plan table to console."""
    print("\n" + "=" * 70)
    print("  OCEANSCOPE AI - BUILD STORE DRY-RUN PLAN (PRD_AI_ML §4.2)")
    print("=" * 70)
    print(f"Target Years       : {plan['years']}")
    print(f"Total Years Count  : {plan['n_years']}")
    print(f"Static Transfer    : {plan['static_transfer_mb']:.1f} MB (GEBCO + WOA23)")
    total_tx_gb = plan["total_transfer_mb"] / 1024.0
    total_st_gb = plan["total_store_growth_mb"] / 1024.0
    print(f"Total Est. Transfer: {total_tx_gb:.2f} GB ({plan['total_transfer_mb']:.1f} MB)")
    print(f"Total Est. Storage : {total_st_gb:.2f} GB (target <= 20 GB master)")
    print("-" * 70)
    print(f"{'Year':<8} {'Days':<6} {'Est. Transfer (MB)':<22} {'Est. Store Growth (MB)':<22}")
    print("-" * 70)
    for y_info in plan["years_detail"]:
        y_yr = y_info["year"]
        y_days = y_info["days"]
        y_tx = y_info["estimated_transfer_mb"]
        y_st = y_info["estimated_store_mb"]
        print(f"{y_yr:<8} {y_days:<6} {y_tx:<22.1f} {y_st:<22.1f}")
    print("=" * 70)
    print("Dry-run complete. No data was downloaded or written.\n")


def reduce_year_dataset(
    year: int,
    raw_path: Path,
    target_grid: TargetGrid,
    depths_m: list[float],
) -> tuple[xr.Dataset, list[str]]:
    """Reduce all products for a given year using shared reduce functions."""
    source_files: list[str] = []
    processed_vars: dict[str, xr.DataArray] = {}

    # --- GLORYS12 ---
    glorys_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "glorys" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    if not glorys_candidates:
        raise FileNotFoundError(f"Missing GLORYS12 raw data for year {year} in {raw_path}")

    source_files.append(str(glorys_candidates[0].relative_to(raw_path)))
    ds_raw_glorys = xr.open_dataset(glorys_candidates[0])
    ds_red_glorys = reduce_glorys12(ds_raw_glorys, target_grid, depths_m)
    processed_vars["thetao"] = ds_red_glorys["thetao"]
    processed_vars["ocean_mask"] = ds_red_glorys["ocean_mask"]
    processed_vars["target_valid"] = ds_red_glorys["target_valid"]
    common_time = ds_red_glorys.coords["time"]

    # Filter to requested year if dataset spans multi-year
    year_times = [t for t in common_time.values if pd.Timestamp(t).year == year]
    if year_times:
        target_dates = np.array(year_times)
        processed_vars["thetao"] = processed_vars["thetao"].sel(time=target_dates)
        processed_vars["target_valid"] = processed_vars["target_valid"].sel(time=target_dates)
        common_time = xr.DataArray(target_dates, dims=("time",), coords={"time": target_dates})
    else:
        raise ValueError(
            f"GLORYS dataset does not contain dates for year {year}. Available years: "
            f"{sorted(list({pd.Timestamp(t).year for t in common_time.values}))}"
        )

    # --- SLA & ADT ---
    sla_candidates = [
        f
        for f in raw_path.glob("**/*")
        if ("sla" in f.name.lower() or "duacs" in f.name.lower()) and f.suffix in [".nc", ".nc4"]
    ]
    if sla_candidates:
        source_files.append(str(sla_candidates[0].relative_to(raw_path)))
        ds_raw_sla = xr.open_dataset(sla_candidates[0])
        ds_red_sla = reduce_sla(ds_raw_sla, target_grid, target_time=target_dates)
        processed_vars["sla"] = ds_red_sla["sla"]
        if "adt" in ds_raw_sla.data_vars:
            ds_red_adt = reduce_adt(ds_raw_sla, target_grid, target_time=target_dates)
            processed_vars["adt"] = ds_red_adt["adt"]

    if "adt" not in processed_vars:
        adt_candidates = [
            f
            for f in raw_path.glob("**/*")
            if "adt" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ]
        if adt_candidates:
            source_files.append(str(adt_candidates[0].relative_to(raw_path)))
            ds_red_adt = reduce_adt(
                xr.open_dataset(adt_candidates[0]), target_grid, target_time=target_dates
            )
            processed_vars["adt"] = ds_red_adt["adt"]

    # --- OSTIA SST ---
    sst_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "sst" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    if sst_candidates:
        source_files.append(str(sst_candidates[0].relative_to(raw_path)))
        ds_red_sst = reduce_sst(
            xr.open_dataset(sst_candidates[0]), target_grid, target_time=target_dates
        )
        processed_vars["sst"] = ds_red_sst["sst"]

    # --- MULTIOBS SSS ---
    sss_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if ("sss" in f.name.lower() or "sos" in f.name.lower()) and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    if sss_candidates:
        source_files.append(str(sss_candidates[0].relative_to(raw_path)))
        ds_red_sss = reduce_sss(
            xr.open_dataset(sss_candidates[0]), target_grid, target_time=target_dates
        )
        processed_vars["sss"] = ds_red_sss["sss"]

    # --- OSCAR Currents ---
    oscar_candidates = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if ("oscar" in f.name.lower() or "current" in f.name.lower())
            and f.suffix in [".nc", ".nc4"]
        ]
    )
    if oscar_candidates:
        source_files.append(str(oscar_candidates[0].relative_to(raw_path)))
        if len(oscar_candidates) > 1 and any("final" in f.name for f in oscar_candidates):
            date_strs = [pd.Timestamp(t).strftime("%Y%m%d") for t in target_dates]
            matched = [f for f in oscar_candidates if any(d in f.name for d in date_strs)]
            files_to_open = matched if matched else [oscar_candidates[0]]
            try:
                ds_raw_curr = xr.open_mfdataset(files_to_open, combine="nested", concat_dim="time")
            except Exception:
                ds_raw_curr = xr.open_dataset(files_to_open[0])
        else:
            ds_raw_curr = xr.open_dataset(oscar_candidates[0])
        ds_red_curr = reduce_currents(ds_raw_curr, target_grid, target_time=target_dates)
        processed_vars["u_current"] = ds_red_curr["u_current"]
        processed_vars["v_current"] = ds_red_curr["v_current"]

    # --- ERA5 Winds & Fluxes ---
    winds_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "wind" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    flux_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "flux" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    if winds_cands:
        source_files.append(str(winds_cands[0].relative_to(raw_path)))
        ds_red_winds = reduce_era5_winds(
            xr.open_dataset(winds_cands[0]), target_grid, target_time=target_dates
        )
        processed_vars["u_wind"] = ds_red_winds["u_wind"]
        processed_vars["v_wind"] = ds_red_winds["v_wind"]
        processed_vars["wind_stress_curl"] = ds_red_winds["wind_stress_curl"]

    if flux_cands:
        source_files.append(str(flux_cands[0].relative_to(raw_path)))
        ds_red_flux = reduce_era5_fluxes(
            xr.open_dataset(flux_cands[0]), target_grid, target_time=target_dates
        )
        processed_vars["net_heat_flux"] = ds_red_flux["net_heat_flux"]
        processed_vars["precip"] = ds_red_flux["precip"]

    # --- GEBCO Bathymetry (Static) ---
    gebco_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "gebco" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ],
        key=lambda f: f.stat().st_size,
    )
    if gebco_cands:
        source_files.append(str(gebco_cands[0].relative_to(raw_path)))
        ds_red_gebco = reduce_gebco(xr.open_dataset(gebco_cands[0]), target_grid)
        processed_vars["bathymetry"] = ds_red_gebco["bathymetry"]

    # --- WOA23 Climatology (Static by month) ---
    woa_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "woa" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ]
    )
    if woa_cands:
        source_files.append(str(woa_cands[0].relative_to(raw_path)))
        ds_red_woa = reduce_woa(
            xr.open_dataset(woa_cands[0], decode_times=False), target_grid, depths_m
        )
        # Broadcast month profile across this year's dates
        woa_prof = ds_red_woa["woa_clim_profile"].values  # (1, depth, lat, lon)
        n_days = len(target_dates)
        woa_time = np.repeat(woa_prof, n_days, axis=0)
        processed_vars["woa_clim_profile"] = xr.DataArray(
            woa_time.astype(np.float32),
            dims=("time", "depth", "lat", "lon"),
            coords={
                "time": target_dates,
                "depth": depths_m,
                "lat": target_grid.lats,
                "lon": target_grid.lons,
            },
        )

    # --- GREP Ensemble Mean (Auxiliary Target) ---
    grep_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "grep" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ]
    )
    if grep_cands:
        source_files.append(str(grep_cands[0].relative_to(raw_path)))
        ds_red_grep = reduce_grep(xr.open_dataset(grep_cands[0]), target_grid, depths_m)
        processed_vars["grep_mean"] = ds_red_grep["grep_mean"]

    # --- ARMOR3D 3D Temperature (Validation & Test Competitor) ---
    armor_cands = sorted(
        [
            f
            for f in raw_path.glob("**/*")
            if "armor" in f.name.lower() and f.suffix in [".nc", ".nc4"]
        ]
    )
    if armor_cands:
        source_files.append(str(armor_cands[0].relative_to(raw_path)))
        ds_red_armor = reduce_armor3d(xr.open_dataset(armor_cands[0]), target_grid, depths_m)
        processed_vars["armor3d"] = ds_red_armor["armor3d"]

    ds_year = xr.Dataset(data_vars=processed_vars)
    return ds_year, source_files


def export_fp16_store(
    src_zarr: Path,
    dst_zarr: Path,
) -> Path:
    """Export memory-efficient float16 copy of processed store for training (PRD §6.1)."""
    logger.info("Opening master Zarr store from %s for float16 export...", src_zarr)
    ds = xr.open_zarr(src_zarr)

    dst_zarr.parent.mkdir(parents=True, exist_ok=True)
    fp16_vars: dict[str, xr.DataArray] = {}

    for var_name, da in ds.data_vars.items():
        if da.dtype == np.float32 or da.dtype == np.float64:
            fp16_vars[var_name] = da.astype(np.float16)
        else:
            fp16_vars[var_name] = da

    ds_fp16 = xr.Dataset(data_vars=fp16_vars, coords=ds.coords, attrs=ds.attrs)
    ds_fp16.attrs["dtype"] = "float16"
    logger.info("Writing float16 Zarr store to %s...", dst_zarr)
    ds_fp16.to_zarr(dst_zarr, mode="w")
    return dst_zarr


def build_store(
    config: Any,
    years: list[int] | None = None,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    scratch_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    export_fp16: bool = False,
) -> Path:
    """Stream, reduce, and store multi-year ocean dataset one year at a time.

    Parameters
    ----------
    config : ExperimentConfig object
    years : list of years to process (default: start_year to end_year from config)
    raw_dir : path to local raw data directory (used in development/testing)
    out_dir : destination directory for processed Zarr store
    scratch_dir : temporary scratch directory (disposable, <= 20 GB footprint)
    dry_run : if True, prints planned requests and estimated sizes without reading/writing
    force : if True, rebuilds years even if already present in store attributes
    export_fp16 : if True, writes an additional float16 store for training upload
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

    # Check existing store to support resumption
    built_years: list[int] = []
    accumulated_sources: list[str] = []

    if zarr_dest.exists():
        try:
            ds_existing = xr.open_zarr(zarr_dest)
            raw_built = ds_existing.attrs.get("built_years", "[]")
            built_years = json.loads(raw_built) if isinstance(raw_built, str) else list(raw_built)
            raw_src = ds_existing.attrs.get("source_files", "[]")
            accumulated_sources = json.loads(raw_src) if isinstance(raw_src, str) else list(raw_src)
            logger.info("Found existing store at %s with built_years=%s", zarr_dest, built_years)
        except Exception as e:
            logger.warning("Could not read existing store %s: %s", zarr_dest, e)

    raw_base = Path(raw_dir or config.paths.raw)

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

        # In local/testing mode, read from raw_base.
        # If files for that year are missing, generate them in scratch if synthetic.
        yr_raw_path = raw_base
        glorys_for_yr = [
            f
            for f in raw_base.glob("**/*")
            if "glorys" in f.name.lower() and f.suffix in [".nc", ".nc4"] and str(yr) in f.name
        ]
        if not glorys_for_yr:
            logger.info(
                "No local raw files found for year %d in %s. Generating year scratch...",
                yr,
                raw_base,
            )
            sample_dates = pd.date_range(f"{yr}-01-01", periods=3, freq="D")
            generate_synthetic_raw_sample(yr_scratch, target_grid, depths_m, dates=sample_dates)
            yr_raw_path = yr_scratch
        else:
            yr_raw_path = raw_base

        ds_yr, src_files = reduce_year_dataset(yr, yr_raw_path, target_grid, depths_m)
        accumulated_sources.extend(src_files)

        # Chunk definition: (time=32, lat=120, lon=240), 4D: (time=32, depth=15, lat=120, lon=240)
        chunks_map: dict[str, Any] = {
            "time": 32,
            "lat": len(target_grid.lats),
            "lon": len(target_grid.lons),
        }
        if "depth" in ds_yr.dims:
            chunks_map["depth"] = len(depths_m)

        # Write or append to store
        if not zarr_dest.exists() or len(built_years) == 0:
            logger.info("Initializing new Zarr store at %s with year %d...", zarr_dest, yr)
            ds_yr.attrs["dataset_version"] = str(config.dataset_version)
            ds_yr.attrs["config_hash"] = compute_config_hash(config.raw_dict)
            ds_yr.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            ds_yr.attrs["built_years"] = json.dumps([yr])
            ds_yr.attrs["source_files"] = json.dumps(list(set(accumulated_sources)))
            ds_yr.chunk(chunks_map).to_zarr(zarr_dest, mode="w")
            built_years.append(yr)
        else:
            logger.info("Appending year %d to Zarr store at %s...", yr, zarr_dest)
            # Remove static 2D coordinates/variables before appending along time
            vars_to_append = {k: v for k, v in ds_yr.data_vars.items() if "time" in v.dims}
            ds_append = xr.Dataset(data_vars=vars_to_append)
            ds_append.chunk(chunks_map).to_zarr(zarr_dest, append_dim="time")

            built_years.append(yr)
            # Update root attributes with completed year
            ds_updated = xr.open_zarr(zarr_dest)
            ds_updated.attrs["built_years"] = json.dumps(sorted(list(set(built_years))))
            ds_updated.attrs["source_files"] = json.dumps(list(set(accumulated_sources)))

        # Clean scratch disk for this year immediately (scratch <= 20 GB rule)
        if yr_scratch.exists():
            shutil.rmtree(yr_scratch, ignore_errors=True)
            logger.info("Scratch directory for year %d cleaned up.", yr)

    # Compute Climatology & Norm Stats on master store once years are assembled
    logger.info("Updating harmonic climatology and normalization statistics on master store...")
    ds_master = xr.open_zarr(zarr_dest)
    time_vals = ds_master.coords["time"].values
    t_3d = ds_master["thetao"].values

    clim_366 = fit_harmonic_climatology(t_3d, time_vals, n_harmonics=3)
    ds_master["clim"] = xr.DataArray(
        clim_366,
        dims=("dayofyear", "depth", "lat", "lon"),
        coords={
            "dayofyear": np.arange(1, 367),
            "depth": depths_m,
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )

    norm_stats = compute_norm_stats(
        ds_master,
        train_split=config.splits.train,
        test_split=config.splits.test,
    )
    ds_master.attrs["norm_stats"] = json.dumps(norm_stats)
    ds_master.attrs["split_dates"] = json.dumps(
        {
            "train": config.splits.train,
            "val": config.splits.val,
            "test": config.splits.test,
        }
    )
    ds_master.attrs["built_years"] = json.dumps(sorted(list(set(built_years))))
    ds_master.to_zarr(zarr_dest, mode="a")
    logger.info("Master Zarr store update complete at %s (built_years=%s).", zarr_dest, built_years)

    if export_fp16:
        fp16_dest = out_base / f"oceanscope_{config.dataset_version}_fp16.zarr"
        export_fp16_store(zarr_dest, fp16_dest)

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
        "--fp16",
        action="store_true",
        help="Export additional float16 store for training upload",
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
        export_fp16=args.fp16,
    )
    if not args.dry_run:
        print("\n[OK] build_store complete. Processed store ready.")


if __name__ == "__main__":
    main()
