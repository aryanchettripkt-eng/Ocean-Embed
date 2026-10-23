"""Raw ocean data inspection stage (PRD_AI_ML §0 rules 1-2, §2, §4, §7).

Analyzes downloaded raw datasets using xarray to extract exact variable names,
coordinates, dimensions, units, native depth levels, and time coverage.
Produces data_inspection.md and data_inspection.json in outputs/reports/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.manifest import load_manifest

logger = logging.getLogger("oceanscope_ml.data.inspect")

# Target depth tiers (meters) for finding nearest native levels
TARGET_DEPTH_TIERS_M = [
    0.5,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    35.0,
    55.0,
    90.0,
    155.0,
    260.0,
    450.0,
    750.0,
    1000.0,
    1450.0,
]


def find_raw_data_files(raw_dir: Path) -> list[Path]:
    """Find all raw NetCDF, Zarr, CSV, or Parquet files in raw_dir."""
    if not raw_dir.exists():
        return []
    valid_suffixes = {".nc", ".nc4", ".csv", ".parquet"}
    files = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in valid_suffixes
        and not p.name.startswith(".")
        and "MANIFEST.json" not in p.name
    ]
    # Also include directory zarr stores
    zarr_dirs = [p for p in raw_dir.rglob("*.zarr") if p.is_dir()]
    return sorted(files + zarr_dirs)


def inspect_file(file_path: Path) -> dict[str, Any]:
    """Open NetCDF or Zarr file with xarray and inspect coordinates, variables, and depths."""
    try:
        if file_path.is_dir() and file_path.suffix == ".zarr":
            ds = xr.open_zarr(file_path)
        elif file_path.suffix in [".nc", ".nc4"]:
            ds = xr.open_dataset(file_path)
        else:
            return {
                "file": file_path.name,
                "path": str(file_path),
                "type": file_path.suffix.lstrip("."),
            }
    except Exception as exc:
        logger.warning("Failed to open %s with xarray: %s", file_path, exc)
        return {"file": file_path.name, "path": str(file_path), "error": str(exc)}

    coords_info: dict[str, Any] = {}
    for c_name, c_da in ds.coords.items():
        coords_info[str(c_name)] = {
            "dtype": str(c_da.dtype),
            "shape": list(c_da.shape),
            "attrs": {k: str(v) for k, v in c_da.attrs.items()},
        }

    variables_info: dict[str, Any] = {}
    for v_name, v_da in ds.data_vars.items():
        variables_info[str(v_name)] = {
            "dims": list(v_da.dims),
            "dtype": str(v_da.dtype),
            "shape": list(v_da.shape),
            "units": str(v_da.attrs.get("units", "unknown")),
            "standard_name": str(v_da.attrs.get("standard_name", "")),
            "fill_value": str(v_da.encoding.get("_FillValue", v_da.attrs.get("_FillValue", ""))),
        }

    # Detect native depth coordinate
    depth_coords: list[float] = []
    depth_coord_name = None
    for candidate in ["depth", "deptht", "lev", "level", "z", "depth_m"]:
        if candidate in ds.coords:
            depth_coord_name = candidate
            vals = ds[candidate].values
            depth_coords = [float(d) for d in np.ravel(vals) if not np.isnan(d)]
            break

    # Time coverage
    time_info: dict[str, Any] = {}
    for candidate in ["time", "date"]:
        if candidate in ds.coords:
            t_vals = ds[candidate].values
            if len(t_vals) > 0:
                time_info["start"] = str(t_vals[0])
                time_info["end"] = str(t_vals[-1])
                time_info["steps"] = len(t_vals)
            break

    sizes_dict = dict(ds.sizes)
    ds.close()

    return {
        "file": file_path.name,
        "path": str(file_path),
        "dimensions": sizes_dict,
        "coordinates": coords_info,
        "variables": variables_info,
        "depth_coord_name": depth_coord_name,
        "native_depths": depth_coords,
        "time_coverage": time_info,
    }


def propose_15_depths_from_native(native_depths: list[float]) -> list[float]:
    """Given GLORYS native depth levels, select 15 levels closest to target oceanographic tiers."""
    if not native_depths:
        return []
    sorted_native = sorted(native_depths)
    proposed: list[float] = []
    used_indices: set[int] = set()

    for target in TARGET_DEPTH_TIERS_M:
        best_idx = -1
        best_diff = float("inf")
        for i, d in enumerate(sorted_native):
            diff = abs(d - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx != -1 and best_idx not in used_indices:
            used_indices.add(best_idx)
            proposed.append(sorted_native[best_idx])

    if len(proposed) < 15:
        for i, d in enumerate(sorted_native):
            if i not in used_indices:
                proposed.append(d)
                used_indices.add(i)
                if len(proposed) == 15:
                    break

    return sorted(proposed)


def classify_argo_region(lat: float, lon: float) -> str:
    """Classify ARGO profile location into Indian Ocean subregions (PRD_AI_ML §6.3)."""
    if lat >= 5.0 and lon >= 78.0:
        return "bay_of_bengal"
    elif lat >= 5.0 and lon < 78.0:
        return "arabian_sea"
    else:
        return "equatorial"


def build_real_argo_holdout(
    argo_files: list[Path],
    seed: int = 42,
    holdout_fraction: float = 0.20,
) -> dict[str, Any]:
    """Build stratified holdout from real ARGO platform numbers extracted from downloaded files.

    PRD_AI_ML §7:
    - 20% float hold-out.
    - Stratified by region (Bay of Bengal, Arabian Sea, Equatorial).
    - Bay of Bengal has held-out floats across every year present in data.
    - Reports counts per region.
    """
    float_regions: dict[str, list[str]] = {}
    float_bob_years: dict[str, set[int]] = {}

    for p in argo_files:
        try:
            if p.suffix in [".nc", ".nc4"]:
                with xr.open_dataset(p) as ds:
                    if "platform_number" in ds.variables:
                        pns = ds["platform_number"].values
                        lats = ds["LATITUDE"].values if "LATITUDE" in ds.variables else None
                        lons = ds["LONGITUDE"].values if "LONGITUDE" in ds.variables else None
                        times = (
                            ds["JULD"].values
                            if "JULD" in ds.variables
                            else (ds["TIME"].values if "TIME" in ds.variables else None)
                        )

                        flat_pns = np.ravel(pns)
                        flat_lats = np.ravel(lats) if lats is not None else [None] * len(flat_pns)
                        flat_lons = np.ravel(lons) if lons is not None else [None] * len(flat_pns)
                        flat_times = (
                            np.ravel(times) if times is not None else [None] * len(flat_pns)
                        )

                        for pn, lat_val, lon_val, t_val in zip(
                            flat_pns, flat_lats, flat_lons, flat_times
                        ):
                            s = (
                                pn.decode("utf-8").strip()
                                if isinstance(pn, bytes)
                                else str(pn).strip()
                            )
                            if not s:
                                continue
                            if s not in float_regions:
                                float_regions[s] = []
                                float_bob_years[s] = set()

                            if (
                                lat_val is not None
                                and lon_val is not None
                                and not np.isnan(lat_val)
                                and not np.isnan(lon_val)
                            ):
                                reg = classify_argo_region(float(lat_val), float(lon_val))
                                float_regions[s].append(reg)
                                if reg == "bay_of_bengal" and t_val is not None:
                                    try:
                                        yr = pd.to_datetime(t_val).year
                                        float_bob_years[s].add(int(yr))
                                    except Exception:
                                        pass
            elif p.suffix == ".csv":
                df = pd.read_csv(p)
                if "platform_number" in df.columns:
                    for _, row in df.iterrows():
                        pn_val = row["platform_number"]
                        if pd.isna(pn_val):
                            continue
                        s = str(pn_val).strip()
                        if not s:
                            continue
                        if s not in float_regions:
                            float_regions[s] = []
                            float_bob_years[s] = set()

                        lat_val = (
                            row.get("latitude") if "latitude" in df.columns else row.get("lat")
                        )
                        lon_val = (
                            row.get("longitude") if "longitude" in df.columns else row.get("lon")
                        )
                        if (
                            lat_val is not None
                            and lon_val is not None
                            and not pd.isna(lat_val)
                            and not pd.isna(lon_val)
                        ):
                            reg = classify_argo_region(float(lat_val), float(lon_val))
                            float_regions[s].append(reg)
                            if reg == "bay_of_bengal":
                                date_val = row.get("date") or row.get("time")
                                if date_val and not pd.isna(date_val):
                                    try:
                                        yr = pd.to_datetime(date_val).year
                                        float_bob_years[s].add(int(yr))
                                    except Exception:
                                        pass
        except Exception as exc:
            logger.warning("Error reading ARGO platforms from %s: %s", p, exc)

    if not float_regions:
        return {
            "status": "pending_download",
            "message": "No ARGO profiles found in raw data to extract platform numbers",
            "total_floats": 0,
            "held_out_floats": [],
            "counts_by_region": {},
            "held_out_by_region": {},
        }

    # Assign primary region to each float (majority of observations,
    # fallback to bay_of_bengal if tied/present)
    floats_by_region: dict[str, list[str]] = {
        "bay_of_bengal": [],
        "arabian_sea": [],
        "equatorial": [],
    }

    for s, reg_list in float_regions.items():
        if not reg_list:
            # Fallback if no coordinates available
            floats_by_region["equatorial"].append(s)
        elif "bay_of_bengal" in reg_list:
            floats_by_region["bay_of_bengal"].append(s)
        else:
            primary_reg = max(set(reg_list), key=reg_list.count)
            floats_by_region[primary_reg].append(s)

    rng = np.random.default_rng(seed)
    all_held_out: set[str] = set()
    held_out_by_reg: dict[str, list[str]] = {}
    counts_by_reg: dict[str, int] = {}

    # 1. Stratified selection for Bay of Bengal (ensuring float held out in every year)
    bob_floats = sorted(floats_by_region["bay_of_bengal"])
    counts_by_reg["bay_of_bengal"] = len(bob_floats)
    n_bob_target = max(1, int(round(len(bob_floats) * holdout_fraction))) if bob_floats else 0

    bob_held: set[str] = set()
    if bob_floats:
        # Group BoB floats by years they cover
        all_bob_years: set[int] = set()
        for f_id in bob_floats:
            all_bob_years.update(float_bob_years.get(f_id, set()))

        # For every year, pick at least one float if possible
        for yr in sorted(all_bob_years):
            candidate_floats = [
                f for f in bob_floats if yr in float_bob_years.get(f, set()) and f not in bob_held
            ]
            if candidate_floats:
                shuffled_candidates = rng.permutation(candidate_floats).tolist()
                bob_held.add(shuffled_candidates[0])

        # If we need more floats to reach 20%, pick remaining at random
        remaining_bob = [f for f in bob_floats if f not in bob_held]
        if len(bob_held) < n_bob_target and remaining_bob:
            shuffled_rem = rng.permutation(remaining_bob).tolist()
            bob_held.update(shuffled_rem[: n_bob_target - len(bob_held)])

    held_out_by_reg["bay_of_bengal"] = sorted(bob_held)
    all_held_out.update(bob_held)

    # 2. Stratified selection for Arabian Sea and Equatorial
    for reg in ["arabian_sea", "equatorial"]:
        r_floats = sorted(floats_by_region[reg])
        counts_by_reg[reg] = len(r_floats)
        n_target = max(1, int(round(len(r_floats) * holdout_fraction))) if r_floats else 0
        if r_floats:
            shuffled_r = rng.permutation(r_floats).tolist()
            reg_held = sorted(shuffled_r[:n_target])
        else:
            reg_held = []
        held_out_by_reg[reg] = reg_held
        all_held_out.update(reg_held)

    return {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "total_floats": len(float_regions),
        "total_held_out": len(all_held_out),
        "counts_by_region": counts_by_reg,
        "held_out_by_region": {k: len(v) for k, v in held_out_by_reg.items()},
        "held_out_floats": sorted(all_held_out),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def is_glorys_file(path: Path) -> bool:
    """Identify if a file belongs to the GLORYS12 physical reanalysis.

    Copernicus Marine typically names files like:
    `cmems_mod_glo_phy_my_0.083deg_P1D-m_...nc` or stores them in a folder
    named `glorys` or `glorys12` or `001_030`.
    Must strictly exclude competitor products (GREP 001_031 and ARMOR3D 015_004).
    """
    name = path.name.lower()
    folder = str(path.parent).lower()

    # Strictly reject competitor products
    if any(k in name or k in folder for k in ["grep", "001_031", "armor3d", "015_004", "armor-3d"]):
        return False

    # Check filename and folder hierarchy
    if "glorys" in name or "glorys" in folder or "001_030" in name or "001_030" in folder:
        return True
    if "cmems_mod_glo_phy" in name and "0.083" in name:
        return True
    if "cmems_mod_glo_phy_my" in name and "p1d-m" in name:
        return True
    return False


def inspect_raw_data(
    config: ExperimentConfig,
    raw_dir: Path,
) -> dict[str, Any]:
    """Inspect raw files in raw_dir using xarray. Fails if no data files found."""
    files = find_raw_data_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"no raw data found in {raw_dir.resolve()}")

    inspected_files: list[dict[str, Any]] = []
    glorys_native_depths: list[float] = []

    for f in files:
        f_info = inspect_file(f)
        inspected_files.append(f_info)
        # GLORYS12 is the authoritative source for the native 15 depths (PRD_AI_ML §2, §4.1)
        if is_glorys_file(f) and f_info.get("native_depths"):
            glorys_native_depths = f_info["native_depths"]

    # PRD_AI_ML §2: Depths must come exclusively from GLORYS12. Do NOT fall back to GREP or ARMOR3D.
    if glorys_native_depths:
        proposed_15 = propose_15_depths_from_native(glorys_native_depths)
        glorys_status = "extracted_from_glorys"
    else:
        proposed_15 = []
        glorys_status = "missing_glorys_file"

    argo_files = [f for f in files if "argo" in f.name.lower() or "argo" in str(f.parent).lower()]
    argo_holdout = build_real_argo_holdout(
        argo_files,
        seed=config.train.get("seed", 42),
    )

    manifest = load_manifest(raw_dir)

    return {
        "inspected_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir.resolve()),
        "glorys_status": glorys_status,
        "proposed_15_depths": proposed_15,
        "argo_holdout": argo_holdout,
        "manifest": manifest,
        "files": inspected_files,
        "file_count": len(files),
        "glorys_native_depths": glorys_native_depths,
    }


def generate_markdown_report(
    inspection: dict[str, Any],
    config: ExperimentConfig,
) -> str:
    """Format data inspection report in GitHub-flavored markdown based on real inspected data."""
    report = []
    report.append("# OceanScope AI — Raw Data Inspection Report")
    report.append("")
    report.append(f"**Generated:** {inspection['inspected_utc']}  ")
    report.append(f"**Raw Directory:** `{inspection['raw_dir']}`  ")
    report.append(f"**Files Inspected:** {inspection['file_count']}  ")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 1. Owner Confirmation Table (`[CONFIRM]` Items for P1-M3)")
    report.append("")
    report.append(
        "The owner (Aryan) must confirm these values in `docs/DECISIONS.md` "
        "before P3 baseline and training runs begin:"
    )
    report.append("")
    report.append("| Parameter | Proposed Value | PRD Ref | Status | Notes |")
    report.append("|---|---|---|---|---|")

    bbox_str = (
        f"{config.region.lat_min}°–{config.region.lat_max}°N, "
        f"{config.region.lon_min}°–{config.region.lon_max}°E"
    )
    report.append(
        f"| `region.bbox` | {bbox_str} | PRD §2 | `[CONFIRM]` | "
        "Bounding box covering Arabian Sea, Bay of Bengal, Equatorial IO |"
    )
    report.append(
        f"| `time.start` | {config.time.start} | PRD §2 | `[CONFIRM]` | "
        "Common intersection of altimetry, SST, currents, winds |"
    )
    report.append(
        f"| `time.end` | {config.time.end} | PRD §2 | `[CONFIRM]` | "
        "Full period ending with 2021 test split |"
    )
    train_str = f"{config.splits.train[0]} to {config.splits.train[1]}"
    report.append(f"| `splits.train` | {train_str} | PRD §2 | `[CONFIRM]` | Training period |")
    val_str = f"{config.splits.val[0]} to {config.splits.val[1]}"
    report.append(f"| `splits.val` | {val_str} | PRD §2 | `[CONFIRM]` | Validation period |")
    test_str = f"{config.splits.test[0]} to {config.splits.test[1]}"
    report.append(
        f"| `splits.test` | {test_str} | PRD §2 | `[CONFIRM]` | "
        "Sealed test split (evaluated once in P8) |"
    )

    prop_depths = inspection.get("proposed_15_depths", [])
    if prop_depths:
        d_str = f"15 levels: {prop_depths[0]:.3f}m to {prop_depths[-1]:.3f}m"
        status_d = "`[CONFIRM]`"
        notes_d = "Derived from inspected native depth coordinates"
    else:
        d_str = "Pending GLORYS data download"
        status_d = "`[BLOCKED]`"
        notes_d = "Cannot extract native depths until GLORYS is downloaded"

    report.append(f"| `depths_m` | {d_str} | PRD §2, §6.1 | {status_d} | {notes_d} |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 2. Inspected Native Depth Coordinate")
    report.append("")
    if prop_depths:
        report.append("Proposed 15 depth tiers selected from GLORYS12 native coordinates:")
        report.append("")
        report.append("| Level # | Native Depth (m) | Tier Target (m) |")
        report.append("|---|---|---|")
        for i, d in enumerate(prop_depths, 1):
            target = TARGET_DEPTH_TIERS_M[i - 1] if i - 1 < len(TARGET_DEPTH_TIERS_M) else "N/A"
            report.append(f"| {i} | {d:.4f} m | ~{target} m |")
    else:
        report.append(
            "*No GLORYS reanalysis file found in raw directory. Under PRD_AI_ML §2, "
            "the 15 depth tiers must be extracted directly from GLORYS12 native depth "
            "coordinates and cannot be derived from GREP or ARMOR3D. "
            "Proposing depths is deferred until GLORYS download completes.*"
        )
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 3. ARGO Float Holdout Status (Stratified by Region)")
    report.append("")
    argo_status = inspection.get("argo_holdout", {})
    if argo_status.get("status") == "pending_download":
        report.append(
            "*ARGO hold-out generation is pending real ARGO data download. "
            "No synthetic or invented float numbers are used (PRD_AI_ML §0 rule 1).*"
        )
    else:
        tot = argo_status.get("total_floats", 0)
        held = argo_status.get("total_held_out", 0)
        report.append(
            f"Extracted {tot} unique ARGO platforms from downloaded data. "
            f"Held out {held} floats (20%) using seed={argo_status.get('seed', 42)}."
        )
        report.append("")
        report.append("| Region | Total Floats | Held Out (20%) |")
        report.append("|---|---|---|")
        counts_reg = argo_status.get("counts_by_region", {})
        held_reg = argo_status.get("held_out_by_region", {})
        for reg_key in ["bay_of_bengal", "arabian_sea", "equatorial"]:
            tot_r = counts_reg.get(reg_key, 0)
            h_r = held_reg.get(reg_key, 0)
            report.append(f"| {reg_key.replace('_', ' ').title()} | {tot_r} | {h_r} |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 4. Inspected Files & Variables")
    report.append("")
    for f in inspection.get("files", []):
        report.append(f"### `{f.get('file')}`")
        if f.get("dimensions"):
            report.append(f"- **Dimensions:** `{f.get('dimensions')}`")
        if f.get("time_coverage"):
            report.append(f"- **Time Coverage:** `{f.get('time_coverage')}`")
        if f.get("variables"):
            report.append("- **Variables:**")
            for v_name, v_info in f.get("variables", {}).items():
                units = v_info.get("units", "")
                report.append(f"  - `{v_name}`: units={units}, dims={v_info.get('dims')}")
        report.append("")

    return "\n".join(report)


def main() -> None:
    """CLI entry point for oceanscope_ml.data.inspect."""
    parser = argparse.ArgumentParser(description="Inspect raw data and generate inspection report.")
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config YAML."
    )
    parser.add_argument("--raw-dir", default=None, help="Directory containing raw data.")
    parser.add_argument("--out-dir", default=None, help="Directory to save report outputs.")

    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.paths.raw
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.outputs / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting raw data in {raw_dir.resolve()}...")

    try:
        inspection = inspect_raw_data(config=cfg, raw_dir=raw_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    json_file = out_dir / "data_inspection.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)
    print(f"Saved inspection parameters to {json_file}")

    report_md = generate_markdown_report(inspection, cfg)
    report_file = out_dir / "data_inspection.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved data inspection report to {report_file}")

    # If real argo holdout was generated, save argo_eval_floats.json
    argo_holdout = inspection.get("argo_holdout", {})
    if argo_holdout.get("total_held_out", 0) > 0:
        holdout_file = out_dir / "argo_eval_floats.json"
        with open(holdout_file, "w", encoding="utf-8") as f:
            json.dump(argo_holdout, f, indent=2)
        print(f"Saved real ARGO holdout floats to {holdout_file}")

    print(f"\nInspection complete! Report available at {report_file}")


if __name__ == "__main__":
    main()
