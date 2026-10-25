"""Data preprocessing and reduction pipeline for OceanScope AI.

Implements PRD_AI_ML §3, §4, §4.2, and §6.1.
Applies shared reduce functions to raw NetCDF/HDF5 data, fits harmonic climatology
over training dates, computes training normalization statistics, and saves
the processed Zarr store (oceanscope_v1.zarr) and sample-day inspection figure.
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

logger = logging.getLogger("oceanscope_ml.data.preprocess")


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of configuration dictionary."""
    cfg_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:16]


def find_product_files(
    raw_path: Path,
    product: str,
    manifest: dict[str, Any] | None = None,
) -> list[Path]:
    """Find local NetCDF files for a product using MANIFEST.json first, then directory patterns.

    Never matches loose substrings or sorts smallest file first.
    """
    # 1. Try MANIFEST.json if available
    if manifest and "products" in manifest and product in manifest["products"]:
        p_info = manifest["products"][product]
        rel_files = [f["rel_path"] for f in p_info.get("files", []) if "rel_path" in f]
        resolved = [(raw_path / rf) for rf in rel_files if (raw_path / rf).exists()]
        if resolved:
            return sorted(resolved)

    # 2. Directory and naming patterns
    patterns: dict[str, list[str]] = {
        "glorys12": ["glorys12/*.nc*", "*cmems_mod_glo_phy_my_0.083deg*.nc*"],
        "sla": ["sla/*.nc*", "*cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs*.nc*"],
        "sst": ["sst/*.nc*", "*METOFFICE-GLO-SST-L4-REP-OBS-SST*.nc*"],
        "sss": ["sss/*.nc*", "*cmems_obs-mob_glo_phy-sss_my_multi*.nc*"],
        "oscar": ["oscar/*.nc*"],
        "era5_instant": ["era5/*instant*.nc*", "era5/data_stream-oper_stepType-instant.nc"],
        "era5_accum": ["era5/*accum*.nc*", "era5/data_stream-oper_stepType-accum.nc"],
        "gebco": ["gebco/*sub_ice_topo*.nc*", "gebco/*.nc*"],
        "woa23": ["woa23/*.nc*"],
        "grep": ["grep/*.nc*", "*cmems_mod_glo_phy-mnstd_my_0.25deg*.nc*"],
        "armor3d": ["armor3d/*.nc*", "*cmems_obs-mob_glo_phy_my_0.125deg*.nc*"],
    }

    matches: list[Path] = []
    for pat in patterns.get(product, []):
        hits = list(raw_path.glob(pat))
        if hits:
            matches.extend(hits)
            break

    # Exclude files with 'synthetic' in name if real files exist
    clean_matches = [
        f
        for f in matches
        if not f.name.startswith("synthetic_") and not f.name.endswith("_synthetic.nc")
    ]
    return sorted(clean_matches)


def run_preprocessing(
    config: ExperimentConfig,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    sample_day: str | None = None,
) -> xr.Dataset:
    """Execute complete data preprocessing pipeline for OceanScope AI.

    Parameters
    ----------
    config : ExperimentConfig loaded from experiment.yaml
    raw_dir : path to raw data directory
    out_dir : path to processed data output directory
    sample_day : date string for m1_sample_day.png

    Returns
    -------
    xr.Dataset matching PRD_AI_ML Section 6.1
    """
    raw_path = raw_dir or config.paths.raw
    out_path = out_dir or config.paths.processed
    out_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw directory {raw_path} does not exist. "
            "PRD §0 rule 1 requires stopping and reporting."
        )

    manifest: dict[str, Any] | None = None
    manifest_file = raw_path / "MANIFEST.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception as e:
            logger.warning("Could not read MANIFEST.json: %s", e)

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

    processed_vars: dict[str, xr.DataArray] = {}
    opened_source_files: dict[str, list[str]] = {}

    # --- GLORYS12 ---
    logger.info("Step 1/11: Reducing GLORYS12 thetao and deriving ocean_mask...")
    glorys_files = find_product_files(raw_path, "glorys12", manifest)
    if not glorys_files:
        raise FileNotFoundError(
            f"Missing GLORYS12 raw file in {raw_path}. PRD §0 rule 1 forbids synthetic stand-ins."
        )
    glorys_file = glorys_files[0]
    logger.info("Opening GLORYS12 from %s...", glorys_file.name)
    ds_raw_glorys = xr.open_dataset(glorys_file)
    ds_red_glorys = reduce_glorys12(ds_raw_glorys, target_grid, depths_m)
    processed_vars["thetao"] = ds_red_glorys["thetao"]
    processed_vars["ocean_mask"] = ds_red_glorys["ocean_mask"]
    processed_vars["target_valid"] = ds_red_glorys["target_valid"]
    common_time = ds_red_glorys.coords["time"]
    ocean_mask_arr = ds_red_glorys["ocean_mask"].values.astype(bool)

    opened_source_files["thetao"] = [str(glorys_file.relative_to(raw_path))]
    opened_source_files["ocean_mask"] = [str(glorys_file.relative_to(raw_path))]
    opened_source_files["target_valid"] = [str(glorys_file.relative_to(raw_path))]

    # --- SLA & ADT ---
    logger.info("Step 2/11: Reducing DUACS SLA & ADT...")
    sla_files = find_product_files(raw_path, "sla", manifest)
    if not sla_files:
        raise FileNotFoundError(
            f"Missing DUACS SLA raw file in {raw_path}. PRD §0 rule 1 forbids synthetic stand-ins."
        )
    sla_file = sla_files[0]
    logger.info("Opening SLA from %s...", sla_file.name)
    ds_raw_sla = xr.open_dataset(sla_file)
    ds_red_sla = reduce_sla(ds_raw_sla, target_grid, target_time=common_time.values)
    processed_vars["sla"] = ds_red_sla["sla"]
    opened_source_files["sla"] = [str(sla_file.relative_to(raw_path))]

    if "adt" in ds_raw_sla.data_vars:
        ds_red_adt = reduce_adt(ds_raw_sla, target_grid, target_time=common_time.values)
        processed_vars["adt"] = ds_red_adt["adt"]
        opened_source_files["adt"] = [str(sla_file.relative_to(raw_path))]
    else:
        adt_files = list(raw_path.glob("adt/*.nc*"))
        if adt_files:
            ds_raw_adt = xr.open_dataset(adt_files[0])
            ds_red_adt = reduce_adt(ds_raw_adt, target_grid, target_time=common_time.values)
            processed_vars["adt"] = ds_red_adt["adt"]
            opened_source_files["adt"] = [str(adt_files[0].relative_to(raw_path))]
        else:
            raise FileNotFoundError(f"Missing DUACS ADT in {raw_path}")

    # --- OSTIA SST ---
    logger.info("Step 3/11: Reducing OSTIA SST...")
    sst_files = find_product_files(raw_path, "sst", manifest)
    if not sst_files:
        raise FileNotFoundError(f"Missing OSTIA SST raw file in {raw_path}")
    sst_file = sst_files[0]
    logger.info("Opening SST from %s...", sst_file.name)
    ds_raw_sst = xr.open_dataset(sst_file)
    ds_red_sst = reduce_sst(ds_raw_sst, target_grid, target_time=common_time.values)
    processed_vars["sst"] = ds_red_sst["sst"]
    opened_source_files["sst"] = [str(sst_file.relative_to(raw_path))]

    # --- MULTIOBS SSS ---
    logger.info("Step 4/11: Reducing MULTIOBS SSS...")
    sss_files = find_product_files(raw_path, "sss", manifest)
    if not sss_files:
        raise FileNotFoundError(f"Missing MULTIOBS SSS raw file in {raw_path}")
    sss_file = sss_files[0]
    logger.info("Opening SSS from %s...", sss_file.name)
    ds_raw_sss = xr.open_dataset(sss_file)
    ds_red_sss = reduce_sss(ds_raw_sss, target_grid, target_time=common_time.values)
    processed_vars["sss"] = ds_red_sss["sss"]
    opened_source_files["sss"] = [str(sss_file.relative_to(raw_path))]

    # --- OSCAR Currents ---
    logger.info("Step 5/11: Reducing OSCAR currents...")
    oscar_files = find_product_files(raw_path, "oscar", manifest)
    if not oscar_files:
        raise FileNotFoundError(f"Missing OSCAR currents raw files in {raw_path}")

    # Match files covering common_time dates if daily files
    date_strs = [pd.Timestamp(t).strftime("%Y%m%d") for t in common_time.values]
    matched_oscar = [f for f in oscar_files if any(d in f.name for d in date_strs)]
    files_to_open = matched_oscar if matched_oscar else oscar_files
    logger.info("Opening %d OSCAR daily files...", len(files_to_open))
    if len(files_to_open) > 1:
        ds_raw_curr = xr.open_mfdataset(files_to_open, combine="nested", concat_dim="time")
    else:
        ds_raw_curr = xr.open_dataset(files_to_open[0])

    ds_red_curr = reduce_currents(ds_raw_curr, target_grid, target_time=common_time.values)
    processed_vars["u_current"] = ds_red_curr["u_current"]
    processed_vars["v_current"] = ds_red_curr["v_current"]
    opened_source_files["u_current"] = [str(f.relative_to(raw_path)) for f in files_to_open]
    opened_source_files["v_current"] = [str(f.relative_to(raw_path)) for f in files_to_open]

    # --- ERA5 Winds & Fluxes ---
    logger.info("Step 6/11: Reducing ERA5 winds & fluxes...")
    era5_inst_files = find_product_files(raw_path, "era5_instant", manifest)
    if not era5_inst_files:
        # Fallback to merged era5 file if present
        era5_merged = list(raw_path.glob("era5/*era5*.nc*"))
        if era5_merged:
            era5_inst_files = era5_merged
        else:
            raise FileNotFoundError(f"Missing ERA5 instant wind file in {raw_path}")

    inst_file = era5_inst_files[0]
    logger.info("Opening ERA5 winds from %s...", inst_file.name)
    ds_winds = xr.open_dataset(inst_file)
    ds_red_winds = reduce_era5_winds(
        ds_winds, target_grid, target_time=common_time.values, ocean_mask=ocean_mask_arr
    )
    processed_vars["u_wind"] = ds_red_winds["u_wind"]
    processed_vars["v_wind"] = ds_red_winds["v_wind"]
    processed_vars["wind_stress_curl"] = ds_red_winds["wind_stress_curl"]
    opened_source_files["u_wind"] = [str(inst_file.relative_to(raw_path))]
    opened_source_files["v_wind"] = [str(inst_file.relative_to(raw_path))]
    opened_source_files["wind_stress_curl"] = [str(inst_file.relative_to(raw_path))]

    era5_acc_files = find_product_files(raw_path, "era5_accum", manifest)
    if not era5_acc_files:
        era5_merged = list(raw_path.glob("era5/*era5*.nc*"))
        if era5_merged:
            era5_acc_files = era5_merged
        else:
            raise FileNotFoundError(f"Missing ERA5 accum fluxes file in {raw_path}")

    acc_file = era5_acc_files[0]
    logger.info("Opening ERA5 fluxes from %s...", acc_file.name)
    ds_flux = xr.open_dataset(acc_file)
    ds_red_flux = reduce_era5_fluxes(ds_flux, target_grid, target_time=common_time.values)
    processed_vars["net_heat_flux"] = ds_red_flux["net_heat_flux"]
    processed_vars["precip"] = ds_red_flux["precip"]
    opened_source_files["net_heat_flux"] = [str(acc_file.relative_to(raw_path))]
    opened_source_files["precip"] = [str(acc_file.relative_to(raw_path))]

    # --- GEBCO Bathymetry ---
    logger.info("Step 7/11: Reducing GEBCO bathymetry...")
    gebco_files = find_product_files(raw_path, "gebco", manifest)
    if not gebco_files:
        raise FileNotFoundError(f"Missing GEBCO bathymetry file in {raw_path}")
    gebco_file = gebco_files[0]
    logger.info("Opening GEBCO from %s...", gebco_file.name)
    ds_raw_gebco = xr.open_dataset(gebco_file)
    ds_red_gebco = reduce_gebco(ds_raw_gebco, target_grid)
    processed_vars["bathymetry"] = ds_red_gebco["bathymetry"]
    opened_source_files["bathymetry"] = [str(gebco_file.relative_to(raw_path))]

    # --- WOA23 Monthly Climatology ---
    logger.info("Step 8/11: Reducing WOA23 climatological profile...")
    woa_files = find_product_files(raw_path, "woa23", manifest)
    if woa_files:
        logger.info("Found %d WOA23 monthly files...", len(woa_files))
        woa_monthly_ds: list[xr.Dataset] = []
        for m_idx, woa_file in enumerate(sorted(woa_files), start=1):
            ds_woa_single = xr.open_dataset(woa_file, decode_times=False)
            woa_monthly_ds.append(reduce_woa(ds_woa_single, target_grid, depths_m, month=m_idx))
        if len(woa_monthly_ds) == 1:
            processed_vars["woa_clim_profile"] = woa_monthly_ds[0]["woa_clim_profile"]
        else:
            ds_woa_all = xr.concat([d["woa_clim_profile"] for d in woa_monthly_ds], dim="month")
            processed_vars["woa_clim_profile"] = ds_woa_all
        opened_source_files["woa_clim_profile"] = [str(f.relative_to(raw_path)) for f in woa_files]

    # --- GREP Ensemble Mean ---
    logger.info("Step 9/11: Reducing GREP ensemble mean...")
    grep_files = find_product_files(raw_path, "grep", manifest)
    if grep_files:
        grep_file = grep_files[0]
        logger.info("Opening GREP from %s...", grep_file.name)
        ds_raw_grep = xr.open_dataset(grep_file)
        ds_red_grep = reduce_grep(ds_raw_grep, target_grid, depths_m)
        processed_vars["grep_mean"] = ds_red_grep["grep_mean"]
        opened_source_files["grep_mean"] = [str(grep_file.relative_to(raw_path))]

    # --- ARMOR3D Baseline Competitor ---
    logger.info("Step 10/11: Reducing ARMOR3D competitor...")
    armor_files = find_product_files(raw_path, "armor3d", manifest)
    if armor_files:
        armor_file = armor_files[0]
        logger.info("Opening ARMOR3D from %s...", armor_file.name)
        ds_raw_armor = xr.open_dataset(armor_file)
        ds_red_armor = reduce_armor3d(ds_raw_armor, target_grid, depths_m)
        processed_vars["armor3d"] = ds_red_armor["armor3d"]
        opened_source_files["armor3d"] = [str(armor_file.relative_to(raw_path))]

    # --- Harmonic Climatology ---
    logger.info("Step 11/11: Fitting harmonic climatology (mean + 3 harmonics)...")
    time_coords = common_time.values
    t_3d = processed_vars["thetao"].values

    time_series = pd.to_datetime(common_time.values)
    min_date = time_series.min().strftime("%Y-%m-%d")
    max_date = time_series.max().strftime("%Y-%m-%d")

    # Check whether store dates fall in train split
    is_in_train = pd.Timestamp(min_date) >= pd.Timestamp(config.splits.train[0]) and pd.Timestamp(
        max_date
    ) <= pd.Timestamp(config.splits.train[1])

    if is_in_train:
        clim_366 = fit_harmonic_climatology(
            t_3d,
            time_coords,
            n_harmonics=3,
            train_split=config.splits.train,
            test_split=config.splits.test,
        )
        sample_only = False
    else:
        # Development sample month outside training split (e.g. January 2019 in validation)
        # Labelled clearly as sample_only (REVIEW_P3 A4)
        clim_366 = fit_harmonic_climatology(
            t_3d,
            time_coords,
            n_harmonics=3,
            test_split=config.splits.test,
        )
        sample_only = True

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

    # Assemble Dataset
    coords_dict: dict[str, Any] = {
        "time": common_time,
        "depth": np.array(depths_m, dtype=np.float64),
        "lat": target_grid.lats,
        "lon": target_grid.lons,
        "dayofyear": np.arange(1, 367),
    }
    if (
        "woa_clim_profile" in processed_vars
        and "month" in processed_vars["woa_clim_profile"].coords
    ):
        coords_dict["month"] = processed_vars["woa_clim_profile"].coords["month"].values

    ds_processed = xr.Dataset(data_vars=processed_vars, coords=coords_dict)

    # Compute Normalization Statistics (Train-only or sample-labelled; PRD_AI_ML §0 rule 4, §6.1)
    if is_in_train:
        norm_stats = compute_norm_stats(
            ds_processed,
            train_split=config.splits.train,
            test_split=config.splits.test,
        )
    else:
        # Compute sample stats for inspection, labelled as sample_only
        norm_stats = compute_norm_stats(
            ds_processed,
            train_split=[min_date, max_date],
            test_split=config.splits.test,
        )

    # Root Attributes (PRD_AI_ML §6.1, REVIEW_P3 A1, A4)
    all_unique_files = sorted(list({f for flist in opened_source_files.values() for f in flist}))
    ds_processed.attrs["dataset_version"] = str(config.dataset_version)
    ds_processed.attrs["config_hash"] = compute_config_hash(config.raw_dict)
    ds_processed.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    ds_processed.attrs["source_files"] = all_unique_files
    ds_processed.attrs["source_files_by_var"] = opened_source_files
    ds_processed.attrs["norm_stats"] = norm_stats
    ds_processed.attrs["sample_only"] = sample_only
    ds_processed.attrs["synthetic"] = False
    if sample_only:
        ds_processed.attrs["norm_stats_sample_only"] = True
    ds_processed.attrs["split_dates"] = {
        "train": config.splits.train,
        "val": config.splits.val,
        "test": config.splits.test,
    }

    # Save Processed Zarr Store
    zarr_filename = f"oceanscope_{config.dataset_version}.zarr"
    zarr_dest = out_path / zarr_filename
    logger.info("Writing processed Zarr store to %s...", zarr_dest)
    ds_processed.to_zarr(zarr_dest, mode="w")

    # Generate Milestone 1 Sample Day Figure (PRD_AI_ML M1)
    chosen_day = sample_day or min_date
    fig_dest = config.paths.outputs / "figures" / "m1_sample_day.png"
    fig_dest.parent.mkdir(parents=True, exist_ok=True)
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
    )
    print("\n[OK] Preprocessing complete. Processed store and m1_sample_day.png generated.")


if __name__ == "__main__":
    main()
