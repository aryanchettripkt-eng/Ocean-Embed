"""Generate tiny, contract-exact synthetic sample outputs for OceanScope AI.

Follows specifications in docs/PRD_AI_ML.md Section 6 and Section 8.1.
Outputs are written with synthetic: true and must never be presented as real results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr


def generate_sample_outputs(out_dir: Path) -> None:
    """Generate contract-exact sample outputs in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    reports_dir = out_dir / "reports"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = "synthetic_v1"
    model_name = "SYNTHETIC_DEMO"
    dataset_version = "v1"

    # Region and coordinates
    lon_min, lon_max = 40.0, 100.0
    lat_min, lat_max = 0.0, 30.0
    res = 1.0

    lons = np.arange(lon_min, lon_max + res / 2.0, res, dtype=np.float64)
    lats = np.arange(lat_min, lat_max + res / 2.0, res, dtype=np.float64)
    n_lon = len(lons)
    n_lat = len(lats)

    depths_m = [
        0.5,
        5.0,
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        125.0,
        150.0,
        200.0,
        300.0,
        500.0,
        750.0,
        1000.0,
    ]
    n_depth = len(depths_m)

    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    n_time = len(dates)
    times = dates.values.astype("datetime64[ns]")

    # Build land mask: 0 = land/unsupported, 1 = ocean valid, 2 = insufficient inputs
    # Indian subcontinent rough bounds: lat 8..28, lon 68..88
    # Arabian peninsula: lat 12..30, lon 40..58
    coverage = np.ones((n_time, n_lat, n_lon), dtype=np.int8)

    for i_lat, lat in enumerate(lats):
        for i_lon, lon in enumerate(lons):
            # Land condition
            is_arabia = (lat >= 13.0) and (lon <= 56.0)
            is_india = (lat >= 8.0) and (lon >= 69.0) and (lon <= 87.0) and (lat + 0.5 * lon > 48.0)
            is_africa = (lat <= 12.0) and (lon <= 51.0)
            is_burma = (lat >= 16.0) and (lon >= 92.0)
            if is_arabia or is_india or is_africa or is_burma:
                coverage[:, i_lat, i_lon] = 0
            # Some insufficient inputs along coastal fringe or edge
            elif (lat == 0.0 or lon == 40.0) and ((i_lat + i_lon) % 3 == 0):
                coverage[:, i_lat, i_lon] = 2

    # Synthesize physically plausible temperatures
    # T(z) = T_deep + (T_surf - T_deep) * exp(-z / 200)
    # T_surf ~ 29°C equator, cooling to ~24°C at 30°N
    temp_glorys = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
    temp_pred = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    depth_arr = np.array(depths_m, dtype=np.float32)

    for t_idx in range(n_time):
        day_factor = 0.2 * np.sin(2.0 * np.pi * t_idx / 365.0)
        for i_lat, lat in enumerate(lats):
            lat_factor = 29.0 - 0.18 * lat + day_factor
            for i_lon, lon in enumerate(lons):
                cov = coverage[t_idx, i_lat, i_lon]
                if cov == 1:
                    surf_t = lat_factor + 0.5 * np.cos(np.radians(lon))
                    # depth profile
                    vert = 4.0 + (surf_t - 4.0) * np.exp(-depth_arr / 220.0)
                    temp_glorys[t_idx, :, i_lat, i_lon] = vert
                    # prediction has small synthetic residual error
                    err = 0.25 * np.sin(depth_arr / 80.0) + 0.1 * np.cos(np.radians(lat * 2))
                    temp_pred[t_idx, :, i_lat, i_lon] = vert + err

    # 1. Write Zarr Store
    zarr_path = predictions_dir / f"predictions_{run_id}.zarr"
    if zarr_path.exists():
        import shutil

        shutil.rmtree(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)

    # Coordinates
    root.create_array("time", data=times.astype("int64"))
    root["time"].attrs["units"] = "nanoseconds since 1970-01-01"
    root["time"].attrs["calendar"] = "proleptic_gregorian"

    root.create_array("depth", data=depth_arr.astype(np.float64))
    root["depth"].attrs["units"] = "m"
    root["depth"].attrs["positive"] = "down"

    root.create_array("lat", data=lats.astype(np.float64))
    root["lat"].attrs["units"] = "degrees_north"

    root.create_array("lon", data=lons.astype(np.float64))
    root["lon"].attrs["units"] = "degrees_east"

    # Data variables (Chunking: 1 day per chunk = (1, n_depth, n_lat, n_lon))
    chunk_4d = (1, n_depth, n_lat, n_lon)
    chunk_3d = (1, n_lat, n_lon)

    ds_pred = root.create_array(
        "temp_pred",
        data=temp_pred,
        chunks=chunk_4d,
        fill_value=np.nan,
    )
    ds_pred.attrs["units"] = "degC"
    ds_pred.attrs["long_name"] = "Reconstructed subsurface temperature"

    ds_glorys = root.create_array(
        "temp_glorys",
        data=temp_glorys,
        chunks=chunk_4d,
        fill_value=np.nan,
    )
    ds_glorys.attrs["units"] = "degC"
    ds_glorys.attrs["long_name"] = "GLORYS reference subsurface temperature"

    ds_cov = root.create_array(
        "coverage",
        data=coverage,
        chunks=chunk_3d,
        fill_value=0,
    )
    ds_cov.attrs["flag_values"] = [0, 1, 2]
    ds_cov.attrs["flag_meanings"] = "land_unsupported valid_prediction insufficient_inputs"

    root.attrs["run_id"] = run_id
    root.attrs["dataset_version"] = dataset_version
    root.attrs["model_name"] = model_name
    root.attrs["created_utc"] = "2026-10-22T12:00:00Z"
    root.attrs["split_of_each_date"] = json.dumps({str(d.date()): "val" for d in dates})
    root.attrs["depths_m"] = depths_m

    # 2. Write Validation Metrics CSVs
    # Overall val metrics
    val_rows: list[dict[str, Any]] = []
    for d in depths_m:
        val_rows.append(
            {
                "model": model_name,
                "split": "val",
                "depth_m": d,
                "n": 10000,
                "mae_c": 0.28,
                "rmse_c": 0.38,
                "bias_c": 0.04,
                "skill_vs_clim_pct": 34.5,
            }
        )
    val_rows.append(
        {
            "model": model_name,
            "split": "val",
            "depth_m": "all",
            "n": 150000,
            "mae_c": 0.29,
            "rmse_c": 0.40,
            "bias_c": 0.03,
            "skill_vs_clim_pct": 32.8,
        }
    )
    pd.DataFrame(val_rows).to_csv(metrics_dir / f"{run_id}_val.csv", index=False)

    # Seasonal metrics
    season_rows: list[dict[str, Any]] = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        for d in depths_m:
            season_rows.append(
                {
                    "model": model_name,
                    "split": "val",
                    "season": season,
                    "depth_m": d,
                    "n": 2500,
                    "mae_c": 0.27,
                    "rmse_c": 0.37,
                    "bias_c": 0.03,
                    "skill_vs_clim_pct": 35.1,
                }
            )
        season_rows.append(
            {
                "model": model_name,
                "split": "val",
                "season": season,
                "depth_m": "all",
                "n": 37500,
                "mae_c": 0.28,
                "rmse_c": 0.39,
                "bias_c": 0.03,
                "skill_vs_clim_pct": 33.2,
            }
        )
    pd.DataFrame(season_rows).to_csv(metrics_dir / f"{run_id}_val_by_season.csv", index=False)

    # Regional metrics
    region_rows: list[dict[str, Any]] = []
    for reg in ["arabian_sea", "bay_of_bengal", "equatorial"]:
        for d in depths_m:
            region_rows.append(
                {
                    "model": model_name,
                    "split": "val",
                    "region": reg,
                    "depth_m": d,
                    "n": 3300,
                    "mae_c": 0.29,
                    "rmse_c": 0.39,
                    "bias_c": 0.04,
                    "skill_vs_clim_pct": 33.0,
                }
            )
        region_rows.append(
            {
                "model": model_name,
                "split": "val",
                "region": reg,
                "depth_m": "all",
                "n": 50000,
                "mae_c": 0.30,
                "rmse_c": 0.41,
                "bias_c": 0.04,
                "skill_vs_clim_pct": 31.5,
            }
        )
    pd.DataFrame(region_rows).to_csv(metrics_dir / f"{run_id}_val_by_region.csv", index=False)

    # Baseline Climatology CSV
    clim_rows: list[dict[str, Any]] = []
    for d in depths_m:
        clim_rows.append(
            {
                "model": "climatology",
                "split": "val",
                "depth_m": d,
                "n": 10000,
                "mae_c": 0.44,
                "rmse_c": 0.58,
                "bias_c": 0.00,
                "skill_vs_clim_pct": 0.0,
            }
        )
    clim_rows.append(
        {
            "model": "climatology",
            "split": "val",
            "depth_m": "all",
            "n": 150000,
            "mae_c": 0.45,
            "rmse_c": 0.60,
            "bias_c": 0.00,
            "skill_vs_clim_pct": 0.0,
        }
    )
    pd.DataFrame(clim_rows).to_csv(metrics_dir / "baseline_climatology_val.csv", index=False)

    # Baseline Ridge CSV
    ridge_rows: list[dict[str, Any]] = []
    for d in depths_m:
        ridge_rows.append(
            {
                "model": "ridge",
                "split": "val",
                "depth_m": d,
                "n": 10000,
                "mae_c": 0.36,
                "rmse_c": 0.48,
                "bias_c": 0.02,
                "skill_vs_clim_pct": 17.2,
            }
        )
    ridge_rows.append(
        {
            "model": "ridge",
            "split": "val",
            "depth_m": "all",
            "n": 150000,
            "mae_c": 0.37,
            "rmse_c": 0.50,
            "bias_c": 0.02,
            "skill_vs_clim_pct": 16.7,
        }
    )
    pd.DataFrame(ridge_rows).to_csv(metrics_dir / "baseline_ridge_val.csv", index=False)

    # ARGO metrics CSV
    argo_rows: list[dict[str, Any]] = []
    for d in depths_m:
        argo_rows.append(
            {
                "depth_m": d,
                "n_profiles": 45,
                "n_obs": 45,
                "mae_c": 0.32,
                "rmse_c": 0.43,
                "bias_c": 0.05,
                "glorys_mae_c": 0.28,
                "glorys_rmse_c": 0.38,
            }
        )
    pd.DataFrame(argo_rows).to_csv(metrics_dir / f"argo_{run_id}.csv", index=False)

    # ARGO matches Parquet
    argo_matches = pd.DataFrame(
        [
            {
                "platform_number": "2902123",
                "cycle": 14,
                "time": "2020-01-05T06:00:00Z",
                "lat": 12.0,
                "lon": 65.0,
                "depth_m": 10.0,
                "temp_obs_c": 27.85,
                "temp_pred_c": 27.92,
                "temp_glorys_c": 27.88,
            },
            {
                "platform_number": "2902124",
                "cycle": 8,
                "time": "2020-01-07T12:00:00Z",
                "lat": 8.0,
                "lon": 85.0,
                "depth_m": 100.0,
                "temp_obs_c": 22.10,
                "temp_pred_c": 22.25,
                "temp_glorys_c": 22.15,
            },
        ]
    )
    table = pa.Table.from_pandas(argo_matches)
    pq.write_table(table, metrics_dir / f"argo_matches_{run_id}.parquet")

    # 3. Write evaluation report
    report_content = (
        f"# Evaluation Report: {run_id} (SYNTHETIC)\n\n"
        "> **Important:** This report contains synthetic fixture data generated for\n"
        "> development and pipeline verification. It must never be presented as real results.\n\n"
        f"## Model\n- **Model Name:** {model_name}\n"
        f"- **Run ID:** {run_id}\n- **Dataset Version:** {dataset_version}\n\n"
        "## Scientific Caveats\n"
        "- Predictions are trained to reproduce GLORYS reanalysis.\n"
        "- GLORYS assimilates satellite altimetry and ARGO profiles, so skill vs GLORYS "
        "is not independent skill vs the real ocean.\n"
    )
    (reports_dir / f"evaluation_{run_id}.md").write_text(report_content, encoding="utf-8")

    # 4. Write manifest.json
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "active_run_id": run_id,
        "dataset_version": dataset_version,
        "model_name": model_name,
        "created_utc": "2026-10-22T12:00:00Z",
        "region": {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        },
        "grid_resolution_deg": res,
        "depths_m": depths_m,
        "available_dates": {
            "start": "2020-01-01",
            "end": "2020-01-10",
        },
        "splits": {
            "train": ["2010-01-01", "2018-12-01"],
            "val": ["2019-01-01", "2020-12-01"],
            "test": ["2021-01-01", "2021-12-31"],
        },
        "test_evaluated": False,
        "files": {
            "predictions": f"predictions/predictions_{run_id}.zarr",
            "metrics_val": f"metrics/{run_id}_val.csv",
            "metrics_val_by_season": f"metrics/{run_id}_val_by_season.csv",
            "metrics_val_by_region": f"metrics/{run_id}_val_by_region.csv",
            "metrics_test": None,
            "baselines_val": [
                "metrics/baseline_climatology_val.csv",
                "metrics/baseline_ridge_val.csv",
            ],
            "argo": f"metrics/argo_{run_id}.csv",
            "argo_matches": f"metrics/argo_matches_{run_id}.parquet",
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated contract-exact sample outputs in {out_dir}")


def main() -> None:
    """CLI entry point for make_sample_outputs."""
    parser = argparse.ArgumentParser(description="Generate synthetic sample outputs.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../fixtures/sample_outputs"),
        help="Target output directory (default: ../fixtures/sample_outputs).",
    )
    args = parser.parse_args()
    generate_sample_outputs(args.out)


if __name__ == "__main__":
    main()
