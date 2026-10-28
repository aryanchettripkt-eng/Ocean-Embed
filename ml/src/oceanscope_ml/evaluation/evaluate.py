"""Validation metrics from the prediction store, and the manifest (PRD_AI_ML §4 stage 5, §6.3).

Usage (from ml/):

    python -m oceanscope_ml.evaluation.evaluate --config configs/experiment.yaml \
        --predictions outputs/predictions/predictions_<run_id>.zarr \
        --store <parts...> --stats <stats.zarr> [--out-dir outputs]

Writes ``<out>/metrics/<run_id>_val.csv``, ``_val_by_season.csv``, ``_val_by_region.csv`` and
``<out>/manifest.json``. Errors are model minus GLORYS on cells where the prediction is valid
(coverage 1) and the GLORYS target is valid; ``skill_vs_clim_pct`` compares with the
climatology on exactly the same cells. Only validation dates are read.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import check_no_test_leakage

logger = logging.getLogger("oceanscope_ml.evaluation.evaluate")

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM"}
SEASONS.update({6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
# Region definitions are part of the §6.3 contract, not tunables.
BAY_OF_BENGAL_MIN_LON = 78.0
EQUATORIAL_MAX_LAT = 5.0
BLOCK_DAYS = 16


def region_masks(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """[lat, lon] masks for the §6.3 regions (arabian_sea and equatorial overlap by design)."""
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    return {
        "arabian_sea": lon2 < BAY_OF_BENGAL_MIN_LON,
        "bay_of_bengal": (lon2 >= BAY_OF_BENGAL_MIN_LON) & (lat2 >= EQUATORIAL_MAX_LAT),
        "equatorial": lat2 < EQUATORIAL_MAX_LAT,
    }


class _Acc:
    """Running sums per depth for the model error and the climatology error."""

    def __init__(self, n_depth: int) -> None:
        self.n = np.zeros(n_depth)
        self.err = np.zeros(n_depth)
        self.abs = np.zeros(n_depth)
        self.sq = np.zeros(n_depth)
        self.clim_sq = np.zeros(n_depth)

    def add(self, err: np.ndarray, clim_err: np.ndarray, valid: np.ndarray) -> None:
        """err, clim_err, valid: [days, depth, cells]."""
        e = np.where(valid, err, 0.0)
        c = np.where(valid, clim_err, 0.0)
        self.n += valid.sum(axis=(0, 2))
        self.err += e.sum(axis=(0, 2))
        self.abs += np.abs(e).sum(axis=(0, 2))
        self.sq += (e**2).sum(axis=(0, 2))
        self.clim_sq += (c**2).sum(axis=(0, 2))

    def rows(self, depths: list[float], base: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        idx: list[tuple[Any, Any]] = [(d, i) for i, d in enumerate(depths)]
        for depth, sel in [*idx, ("all", slice(None))]:
            n = float(np.sum(self.n[sel]))
            if n > 0:
                rmse = float(np.sqrt(np.sum(self.sq[sel]) / n))
                rmse_clim = float(np.sqrt(np.sum(self.clim_sq[sel]) / n))
                row = {
                    "depth_m": depth,
                    "n": int(n),
                    "mae_c": round(float(np.sum(self.abs[sel]) / n), 3),
                    "rmse_c": round(rmse, 3),
                    "bias_c": round(float(np.sum(self.err[sel]) / n), 3),
                    "skill_vs_clim_pct": (
                        round(100.0 * (1.0 - rmse / rmse_clim), 2) if rmse_clim > 1e-6 else np.nan
                    ),
                }
            else:
                row = {"depth_m": depth, "n": 0, "mae_c": np.nan, "rmse_c": np.nan}
                row.update({"bias_c": np.nan, "skill_vs_clim_pct": np.nan})
            out.append({**base, **row})
        return out


def evaluate_predictions(
    config: ExperimentConfig,
    predictions: Path,
    store_path: Path | list[Path],
    stats_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Write the three validation metric CSVs; return their paths."""
    pred = xr.open_zarr(predictions)
    run_id = str(pred.attrs["run_id"])
    model_name = str(pred.attrs.get("model_name", "unet"))
    dates = pd.to_datetime(pred["time"].values)
    check_no_test_leakage(dates, config.splits.test)
    val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
    if len(dates) == 0 or dates.min() < val_start or dates.max() > val_end:
        raise ValueError("The prediction store must hold validation dates only")

    ds = open_parts(store_path if isinstance(store_path, list) else [store_path], stats_path)
    ds = ds.sel(time=dates)
    depths = [float(d) for d in pred["depth"].values]
    n_d = len(depths)
    regions = region_masks(np.asarray(pred["lat"].values), np.asarray(pred["lon"].values))
    clim = np.asarray(ds["clim"].values, dtype=np.float32)
    has_valid = "target_valid" in ds.data_vars

    total = _Acc(n_d)
    by_season = {s: _Acc(n_d) for s in SEASON_ORDER}
    by_region = {r: _Acc(n_d) for r in regions}
    for b0 in range(0, len(dates), BLOCK_DAYS):
        b1 = min(b0 + BLOCK_DAYS, len(dates))
        p = np.asarray(pred["temp_pred"].isel(time=slice(b0, b1)).values, dtype=np.float32)
        cov = np.asarray(pred["coverage"].isel(time=slice(b0, b1)).values)
        g = np.asarray(ds["thetao"].isel(time=slice(b0, b1)).values, dtype=np.float32)
        c = clim[dates[b0:b1].day_of_year.values - 1]
        valid = np.isfinite(p) & np.isfinite(g) & np.isfinite(c) & (cov == 1)[:, None]
        if has_valid:
            valid &= np.asarray(ds["target_valid"].isel(time=slice(b0, b1)).values, dtype=bool)
        err, clim_err = p - g, c - g
        flat = (err.shape[0], n_d, -1)
        total.add(err.reshape(flat), clim_err.reshape(flat), valid.reshape(flat))
        months = dates[b0:b1].month
        for season, acc in by_season.items():
            day_sel = np.array([SEASONS[m] == season for m in months])
            if day_sel.any():
                acc.add(
                    err[day_sel].reshape(int(day_sel.sum()), n_d, -1),
                    clim_err[day_sel].reshape(int(day_sel.sum()), n_d, -1),
                    valid[day_sel].reshape(int(day_sel.sum()), n_d, -1),
                )
        for name, mask in regions.items():
            by_region[name].add(err[..., mask], clim_err[..., mask], valid[..., mask])

    metrics_dir = Path(out_dir or config.paths.outputs) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    base = {"model": model_name, "split": "val"}
    cols = ["model", "split", "depth_m", "n", "mae_c", "rmse_c", "bias_c", "skill_vs_clim_pct"]
    paths = {
        "metrics_val": metrics_dir / f"{run_id}_val.csv",
        "metrics_val_by_season": metrics_dir / f"{run_id}_val_by_season.csv",
        "metrics_val_by_region": metrics_dir / f"{run_id}_val_by_region.csv",
    }
    pd.DataFrame(total.rows(depths, base))[cols].to_csv(paths["metrics_val"], index=False)
    season_rows = [
        r for s in SEASON_ORDER for r in by_season[s].rows(depths, {**base, "season": s})
    ]
    pd.DataFrame(season_rows)[[*cols[:2], "season", *cols[2:]]].to_csv(
        paths["metrics_val_by_season"], index=False
    )
    region_rows = [r for n, a in by_region.items() for r in a.rows(depths, {**base, "region": n})]
    pd.DataFrame(region_rows)[[*cols[:2], "region", *cols[2:]]].to_csv(
        paths["metrics_val_by_region"], index=False
    )
    overall = pd.read_csv(paths["metrics_val"]).iloc[-1]
    logger.info(
        "Run %s on %d val days: RMSE %.3f C, MAE %.3f C, skill vs climatology %.1f%%",
        run_id,
        len(dates),
        overall["rmse_c"],
        overall["mae_c"],
        overall["skill_vs_clim_pct"],
    )
    return paths


def write_manifest(
    config: ExperimentConfig,
    predictions: Path,
    metric_paths: dict[str, Path],
    out_dir: Path | None = None,
) -> Path:
    """``manifest.json`` for this run (PRD §6.3), with paths relative to ``out_dir``."""
    out = Path(out_dir or config.paths.outputs)
    pred = xr.open_zarr(predictions)
    run_id = str(pred.attrs["run_id"])
    dates = pd.to_datetime(pred["time"].values)

    def rel(path: Path | None) -> str | None:
        return None if path is None or not path.exists() else path.relative_to(out).as_posix()

    metrics_dir = out / "metrics"
    baselines = [
        rel(metrics_dir / f"baseline_{name}_val.csv")
        for name in ("climatology", "ridge")
        if (metrics_dir / f"baseline_{name}_val.csv").exists()
    ]
    manifest = {
        "schema_version": 1,
        "synthetic": False,
        "active_run_id": run_id,
        "dataset_version": str(config.dataset_version),
        "model_name": str(pred.attrs.get("model_name", "unet")),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "region": {
            "lon_min": float(config.region.lon_min),
            "lon_max": float(config.region.lon_max),
            "lat_min": float(config.region.lat_min),
            "lat_max": float(config.region.lat_max),
        },
        "grid_resolution_deg": float(config.grid.resolution_deg),
        "depths_m": [float(d) for d in config.depths_m],
        "available_dates": {"start": str(dates.min().date()), "end": str(dates.max().date())},
        "splits": {
            "train": [str(d) for d in config.splits.train],
            "val": [str(d) for d in config.splits.val],
            "test": [str(d) for d in config.splits.test],
        },
        "test_evaluated": False,
        "files": {
            "predictions": rel(predictions),
            "metrics_val": rel(metric_paths["metrics_val"]),
            "metrics_val_by_season": rel(metric_paths.get("metrics_val_by_season")),
            "metrics_val_by_region": rel(metric_paths.get("metrics_val_by_region")),
            "metrics_test": None,
            "baselines_val": baselines,
            "argo": rel(metrics_dir / f"argo_{run_id}.csv"),
            "argo_matches": rel(metrics_dir / f"argo_matches_{run_id}.parquet"),
            "headline": rel(metrics_dir / f"{run_id}_headline.csv"),
        },
    }
    if manifest["files"]["predictions"] is None or manifest["files"]["metrics_val"] is None:  # type: ignore[index]
        raise FileNotFoundError("Predictions and validation metrics must exist under out_dir")
    _validate(manifest)
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Manifest written: %s", path)
    return path


def _validate(manifest: dict[str, Any]) -> None:
    import jsonschema

    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation metrics and manifest for one run.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else None
    predictions = Path(args.predictions)
    paths = evaluate_predictions(
        config,
        predictions,
        [Path(p) for p in args.store],
        stats_path=Path(args.stats) if args.stats else None,
        out_dir=out_dir,
    )
    write_manifest(config, predictions, paths, out_dir=out_dir)


if __name__ == "__main__":
    main()
