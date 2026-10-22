"""Generate contract-exact synthetic sample outputs for OceanScope AI.

Follows specifications in docs/PRD_AI_ML.md Section 6 and Section 8.1.
Outputs are written with synthetic: true and must never be presented as real results.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr


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

    # Region and coordinates (using cell centres: Review A8)
    # Resolution = 1.0 deg. Centres: lon 40.5..99.5 (60), lat 0.5..29.5 (30)
    lon_min, lon_max = 40.0, 100.0
    lat_min, lat_max = 0.0, 30.0
    res = 1.0

    lons = np.arange(lon_min + res / 2.0, lon_max, res, dtype=np.float64)
    lats = np.arange(lat_min + res / 2.0, lat_max, res, dtype=np.float64)
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

    # Build land mask: 0 = land/unsupported, 1 = ocean valid, 2 = insufficient inputs
    coverage = np.ones((n_time, n_lat, n_lon), dtype=np.int8)

    for i_lat, lat in enumerate(lats):
        for i_lon, lon in enumerate(lons):
            is_arabia = (lat >= 13.0) and (lon <= 56.0)
            is_india = (lat >= 8.0) and (lon >= 69.0) and (lon <= 87.0) and (lat + 0.5 * lon > 48.0)
            is_africa = (lat <= 12.0) and (lon <= 51.0)
            is_burma = (lat >= 16.0) and (lon >= 92.0)
            if is_arabia or is_india or is_africa or is_burma:
                coverage[:, i_lat, i_lon] = 0
            elif (i_lat == 0 or i_lon == 0) and ((i_lat + i_lon) % 3 == 0):
                coverage[:, i_lat, i_lon] = 2

    # Synthesize physically plausible temperatures
    # T(z) = T_deep + (T_surf - T_deep) * exp(-z / 220)
    temp_glorys = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
    temp_pred = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    depth_arr = np.array(depths_m, dtype=np.float32)

    for t_idx in range(n_time):
        day_factor = 0.2 * np.sin(2.0 * np.pi * t_idx / 365.0)
        is_live = t_idx >= 8  # Last 2 days (2020-01-09, 2020-01-10) are live
        for i_lat, lat in enumerate(lats):
            lat_factor = 29.0 - 0.18 * lat + day_factor
            for i_lon, lon in enumerate(lons):
                cov = coverage[t_idx, i_lat, i_lon]
                if cov == 1:
                    surf_t = lat_factor + 0.5 * np.cos(np.radians(lon))
                    vert = 4.0 + (surf_t - 4.0) * np.exp(-depth_arr / 220.0)
                    if not is_live:
                        temp_glorys[t_idx, :, i_lat, i_lon] = vert
                    # Deterministic, depth-varying synthetic anomaly error
                    err = 0.22 * np.sin(depth_arr / 70.0) + 0.08 * np.cos(np.radians(lat * 2))
                    temp_pred[t_idx, :, i_lat, i_lon] = vert + err

    # 1. Build and write xarray Dataset with zarr_format=2 (Review A1)
    ds = xr.Dataset(
        data_vars={
            "temp_pred": (
                ("time", "depth", "lat", "lon"),
                temp_pred,
                {"units": "degC", "long_name": "Reconstructed subsurface temperature"},
            ),
            "temp_glorys": (
                ("time", "depth", "lat", "lon"),
                temp_glorys,
                {"units": "degC", "long_name": "GLORYS reference subsurface temperature"},
            ),
            "coverage": (
                ("time", "lat", "lon"),
                coverage,
                {
                    "flag_values": [0, 1, 2],
                    "flag_meanings": "land_unsupported valid_prediction insufficient_inputs",
                },
            ),
        },
        coords={
            "time": dates.values,
            "depth": depth_arr.astype(np.float64),
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "run_id": run_id,
            "dataset_version": dataset_version,
            "model_name": model_name,
            "created_utc": "2026-10-22T12:00:00Z",
            "split_of_each_date": json.dumps(
                {
                    "2020-01-01/2020-01-08": "val",
                    "2020-01-09/2020-01-10": "live",
                }
            ),
            "depths_m": depths_m,
        },
    )

    ds["depth"].attrs["units"] = "m"
    ds["depth"].attrs["positive"] = "down"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"

    zarr_path = predictions_dir / f"predictions_{run_id}.zarr"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    encoding = {
        "temp_pred": {"chunks": (1, n_depth, n_lat, n_lon), "dtype": "float32"},
        "temp_glorys": {"chunks": (1, n_depth, n_lat, n_lon), "dtype": "float32"},
        "coverage": {"chunks": (1, n_lat, n_lon), "dtype": "int8"},
        "time": {"chunks": (n_time,)},
        "depth": {"chunks": (n_depth,)},
        "lat": {"chunks": (n_lat,)},
        "lon": {"chunks": (n_lon,)},
    }

    ds.to_zarr(str(zarr_path), mode="w", consolidated=True, zarr_format=2, encoding=encoding)

    # 2. Compute Self-Consistent Validation Metrics (Review B2)
    # Validation period is days 0..7 (first 8 days)
    val_slice = slice(0, 8)
    val_valid_mask = coverage[val_slice] == 1
    # Climatology baseline reference error (synthetic ~ 0.58 °C)
    clim_rmse_ref = 0.58

    def calc_stats(pred: np.ndarray, glorys: np.ndarray) -> tuple[int, float, float, float, float]:
        diff = pred - glorys
        n = int(np.count_nonzero(~np.isnan(diff)))
        if n == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        mae = float(np.round(np.nanmean(np.abs(diff)), 2))
        rmse = float(np.round(np.sqrt(np.nanmean(diff**2)), 2))
        bias = float(np.round(np.nanmean(diff), 2))
        skill = float(np.round(100.0 * (1.0 - rmse / clim_rmse_ref), 1))
        return n, mae, rmse, bias, skill

    # Overall validation metrics
    val_rows: list[dict[str, Any]] = []
    total_n_val = 0
    all_pred_list: list[np.ndarray] = []
    all_glorys_list: list[np.ndarray] = []

    for d_idx, d_m in enumerate(depths_m):
        p_d = temp_pred[val_slice, d_idx, :, :][val_valid_mask]
        g_d = temp_glorys[val_slice, d_idx, :, :][val_valid_mask]
        all_pred_list.append(p_d)
        all_glorys_list.append(g_d)
        n_d, mae_d, rmse_d, bias_d, skill_d = calc_stats(p_d, g_d)
        total_n_val += n_d
        val_rows.append(
            {
                "model": model_name,
                "split": "val",
                "depth_m": d_m,
                "n": n_d,
                "mae_c": mae_d,
                "rmse_c": rmse_d,
                "bias_c": bias_d,
                "skill_vs_clim_pct": skill_d,
            }
        )

    all_p = np.concatenate(all_pred_list)
    all_g = np.concatenate(all_glorys_list)
    _, mae_all, rmse_all, bias_all, skill_all = calc_stats(all_p, all_g)
    val_rows.append(
        {
            "model": model_name,
            "split": "val",
            "depth_m": "all",
            "n": total_n_val,
            "mae_c": mae_all,
            "rmse_c": rmse_all,
            "bias_c": bias_all,
            "skill_vs_clim_pct": skill_all,
        }
    )
    pd.DataFrame(val_rows).to_csv(metrics_dir / f"{run_id}_val.csv", index=False)

    # Seasonal metrics
    season_rows: list[dict[str, Any]] = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        season_n = 0
        s_p_list: list[np.ndarray] = []
        s_g_list: list[np.ndarray] = []
        # Multiplier to provide subtle plausible variation across seasons
        factor = {"DJF": 1.0, "MAM": 1.05, "JJA": 0.95, "SON": 1.02}[season]
        for d_idx, d_m in enumerate(depths_m):
            p_d = temp_pred[val_slice, d_idx, :, :][val_valid_mask]
            g_d = temp_glorys[val_slice, d_idx, :, :][val_valid_mask]
            err = (p_d - g_d) * factor
            p_synth = g_d + err
            s_p_list.append(p_synth)
            s_g_list.append(g_d)
            n_d, mae_d, rmse_d, bias_d, skill_d = calc_stats(p_synth, g_d)
            season_n += n_d
            season_rows.append(
                {
                    "model": model_name,
                    "split": "val",
                    "season": season,
                    "depth_m": d_m,
                    "n": n_d,
                    "mae_c": mae_d,
                    "rmse_c": rmse_d,
                    "bias_c": bias_d,
                    "skill_vs_clim_pct": skill_d,
                }
            )
        s_all_p = np.concatenate(s_p_list)
        s_all_g = np.concatenate(s_g_list)
        _, mae_s, rmse_s, bias_s, skill_s = calc_stats(s_all_p, s_all_g)
        season_rows.append(
            {
                "model": model_name,
                "split": "val",
                "season": season,
                "depth_m": "all",
                "n": season_n,
                "mae_c": mae_s,
                "rmse_c": rmse_s,
                "bias_c": bias_s,
                "skill_vs_clim_pct": skill_s,
            }
        )
    pd.DataFrame(season_rows).to_csv(metrics_dir / f"{run_id}_val_by_season.csv", index=False)

    # Regional metrics (definitions from PRD_AI_ML §6.3)
    region_rows: list[dict[str, Any]] = []
    lat_grid = np.broadcast_to(lats[np.newaxis, :, np.newaxis], (8, n_lat, n_lon))
    lon_grid = np.broadcast_to(lons[np.newaxis, np.newaxis, :], (8, n_lat, n_lon))

    regions_def = {
        "arabian_sea": val_valid_mask & (lon_grid < 78.0),
        "bay_of_bengal": val_valid_mask & (lon_grid >= 78.0) & (lat_grid >= 5.0),
        "equatorial": val_valid_mask & (lat_grid < 5.0),
    }

    for reg_name, reg_mask in regions_def.items():
        reg_total_n = 0
        r_p_list: list[np.ndarray] = []
        r_g_list: list[np.ndarray] = []
        for d_idx, d_m in enumerate(depths_m):
            p_d = temp_pred[val_slice, d_idx, :, :][reg_mask]
            g_d = temp_glorys[val_slice, d_idx, :, :][reg_mask]
            r_p_list.append(p_d)
            r_g_list.append(g_d)
            n_d, mae_d, rmse_d, bias_d, skill_d = calc_stats(p_d, g_d)
            reg_total_n += n_d
            region_rows.append(
                {
                    "model": model_name,
                    "split": "val",
                    "region": reg_name,
                    "depth_m": d_m,
                    "n": n_d,
                    "mae_c": mae_d,
                    "rmse_c": rmse_d,
                    "bias_c": bias_d,
                    "skill_vs_clim_pct": skill_d,
                }
            )
        r_all_p = np.concatenate(r_p_list)
        r_all_g = np.concatenate(r_g_list)
        _, mae_r, rmse_r, bias_r, skill_r = calc_stats(r_all_p, r_all_g)
        region_rows.append(
            {
                "model": model_name,
                "split": "val",
                "region": reg_name,
                "depth_m": "all",
                "n": reg_total_n,
                "mae_c": mae_r,
                "rmse_c": rmse_r,
                "bias_c": bias_r,
                "skill_vs_clim_pct": skill_r,
            }
        )
    pd.DataFrame(region_rows).to_csv(metrics_dir / f"{run_id}_val_by_region.csv", index=False)

    # Climatology baseline CSV (skill = 0.0)
    clim_rows: list[dict[str, Any]] = []
    total_clim_n = 0
    for d_idx, d_m in enumerate(depths_m):
        n_d = int(np.count_nonzero(val_valid_mask))
        total_clim_n += n_d
        clim_rows.append(
            {
                "model": "climatology",
                "split": "val",
                "depth_m": d_m,
                "n": n_d,
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
            "n": total_clim_n,
            "mae_c": 0.44,
            "rmse_c": 0.58,
            "bias_c": 0.00,
            "skill_vs_clim_pct": 0.0,
        }
    )
    pd.DataFrame(clim_rows).to_csv(metrics_dir / "baseline_climatology_val.csv", index=False)

    # Ridge baseline CSV
    ridge_rows: list[dict[str, Any]] = []
    total_ridge_n = 0
    for d_idx, d_m in enumerate(depths_m):
        n_d = int(np.count_nonzero(val_valid_mask))
        total_ridge_n += n_d
        ridge_rows.append(
            {
                "model": "ridge",
                "split": "val",
                "depth_m": d_m,
                "n": n_d,
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
            "n": total_ridge_n,
            "mae_c": 0.36,
            "rmse_c": 0.48,
            "bias_c": 0.02,
            "skill_vs_clim_pct": 17.2,
        }
    )
    pd.DataFrame(ridge_rows).to_csv(metrics_dir / "baseline_ridge_val.csv", index=False)

    # ARGO matchups Parquet and metrics CSV (including GREP and ARMOR3D competitors)
    argo_records: list[dict[str, Any]] = []
    for platform_idx, (p_num, cyc, d_str, p_lat, p_lon) in enumerate(
        [
            ("2902123", 14, "2020-01-03T06:00:00Z", 12.5, 65.5),
            ("2902124", 8, "2020-01-05T12:00:00Z", 8.5, 85.5),
            ("2902125", 22, "2020-01-07T18:00:00Z", 4.5, 75.5),
        ]
    ):
        for d_m in depths_m:
            g_t = float(4.0 + (28.0 - 4.0) * np.exp(-d_m / 220.0))
            p_t = float(g_t + 0.15 * np.sin(d_m / 60.0))
            o_t = float(g_t + 0.08 * np.cos(d_m / 50.0))
            grep_t = float(g_t + 0.18 * np.sin(d_m / 45.0))
            armor_t = float(g_t + 0.22 * np.cos(d_m / 40.0))
            argo_records.append(
                {
                    "platform_number": p_num,
                    "cycle": cyc,
                    "time": d_str,
                    "lat": p_lat,
                    "lon": p_lon,
                    "depth_m": float(d_m),
                    "temp_obs_c": round(o_t, 2),
                    "temp_pred_c": round(p_t, 2),
                    "temp_glorys_c": round(g_t, 2),
                    "temp_grep_c": round(grep_t, 2),
                    "temp_armor3d_c": round(armor_t, 2),
                }
            )

    argo_matches_df = pd.DataFrame(argo_records)
    table = pa.Table.from_pandas(argo_matches_df)
    pq.write_table(table, metrics_dir / f"argo_matches_{run_id}.parquet")

    argo_metric_rows: list[dict[str, Any]] = []
    for d_m in depths_m:
        sub = argo_matches_df[argo_matches_df["depth_m"] == d_m]
        p_diff = sub["temp_pred_c"] - sub["temp_obs_c"]
        g_diff = sub["temp_glorys_c"] - sub["temp_obs_c"]
        grep_diff = sub["temp_grep_c"] - sub["temp_obs_c"]
        armor_diff = sub["temp_armor3d_c"] - sub["temp_obs_c"]
        argo_metric_rows.append(
            {
                "depth_m": d_m,
                "n_profiles": len(sub),
                "n_obs": len(sub),
                "mae_c": round(float(np.mean(np.abs(p_diff))), 2),
                "rmse_c": round(float(np.sqrt(np.mean(p_diff**2))), 2),
                "bias_c": round(float(np.mean(p_diff)), 2),
                "glorys_mae_c": round(float(np.mean(np.abs(g_diff))), 2),
                "glorys_rmse_c": round(float(np.sqrt(np.mean(g_diff**2))), 2),
                "grep_mae_c": round(float(np.mean(np.abs(grep_diff))), 2),
                "grep_rmse_c": round(float(np.sqrt(np.mean(grep_diff**2))), 2),
                "armor3d_mae_c": round(float(np.mean(np.abs(armor_diff))), 2),
                "armor3d_rmse_c": round(float(np.sqrt(np.mean(armor_diff**2))), 2),
            }
        )
    pd.DataFrame(argo_metric_rows).to_csv(metrics_dir / f"argo_{run_id}.csv", index=False)

    # Headline CSV (Bay of Bengal 50-300m vs held-out ARGO, §1.1 table)
    headline_rows = [
        {
            "period": "val",
            "product": "model",
            "n_floats": 3,
            "n_obs": 15,
            "rmse_c": 0.28,
            "ci_low_c": 0.24,
            "ci_high_c": 0.32,
            "diff_vs_model_c": 0.00,
            "diff_ci_low_c": 0.00,
            "diff_ci_high_c": 0.00,
        },
        {
            "period": "val",
            "product": "glorys12",
            "n_floats": 3,
            "n_obs": 15,
            "rmse_c": 0.36,
            "ci_low_c": 0.31,
            "ci_high_c": 0.41,
            "diff_vs_model_c": -0.08,
            "diff_ci_low_c": -0.14,
            "diff_ci_high_c": -0.02,
        },
        {
            "period": "val",
            "product": "grep",
            "n_floats": 3,
            "n_obs": 15,
            "rmse_c": 0.39,
            "ci_low_c": 0.34,
            "ci_high_c": 0.45,
            "diff_vs_model_c": -0.11,
            "diff_ci_low_c": -0.17,
            "diff_ci_high_c": -0.04,
        },
        {
            "period": "val",
            "product": "armor3d",
            "n_floats": 3,
            "n_obs": 15,
            "rmse_c": 0.43,
            "ci_low_c": 0.37,
            "ci_high_c": 0.49,
            "diff_vs_model_c": -0.15,
            "diff_ci_low_c": -0.21,
            "diff_ci_high_c": -0.06,
        },
        {
            "period": "test",
            "product": "model",
            "n_floats": 0,
            "n_obs": 0,
            "rmse_c": None,
            "ci_low_c": None,
            "ci_high_c": None,
            "diff_vs_model_c": None,
            "diff_ci_low_c": None,
            "diff_ci_high_c": None,
        },
        {
            "period": "test",
            "product": "grep",
            "n_floats": 0,
            "n_obs": 0,
            "rmse_c": None,
            "ci_low_c": None,
            "ci_high_c": None,
            "diff_vs_model_c": None,
            "diff_ci_low_c": None,
            "diff_ci_high_c": None,
        },
    ]
    pd.DataFrame(headline_rows).to_csv(metrics_dir / f"{run_id}_headline.csv", index=False)

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
            "train": ["1993-03-02", "2017-12-01"],
            "val": ["2018-01-01", "2020-12-01"],
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
            "headline": f"metrics/{run_id}_headline.csv",
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
