"""ARGO validation: the model and three references against held-out float profiles
(PRD_AI_ML §1.1, §6.3, §7).

Usage (from ml/):

    python -m oceanscope_ml.evaluation.argo_validation --config configs/experiment.yaml \
        --predictions outputs/predictions/predictions_<run_id>.zarr \
        --store <parts...> --stats <stats.zarr> \
        --argo <argo_profiles.nc>... --holdout <argo_eval_floats.json> [--out-dir outputs]

Writes ``metrics/argo_<run_id>.csv`` (per depth), ``metrics/argo_matches_<run_id>.parquet`` (one
row per matched observation) and ``metrics/<run_id>_headline.csv`` (the §1.1 table). Run it before
``oceanscope_ml.evaluation.evaluate``, which lists these files in the manifest only if they exist.

Matching: an observation goes to the grid cell whose centre is within half a cell in lat and lon,
on the same UTC day. Each product's 15-level profile is interpolated linearly in depth to the
observed pressure (dbar, used as depth in metres), with no extrapolation. The observation is then
labelled with the nearest of the 15 levels (``depth_m``).

Headline: only observations where the model and all three references have a value are used, so
every row of a period is computed on the same matchups. Floats are the resampling unit of the 95%
bootstrap. A period without such matchups reports "n/a" for every product and no verdict.
Only validation dates are read here; the test period is not read by this module (PRD_AI_ML §0).

ARGO data are assimilated into GLORYS, so skill against ARGO is not fully independent.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import check_no_test_leakage
from oceanscope_ml.evaluation.evaluate import BAY_OF_BENGAL_MIN_LON, EQUATORIAL_MAX_LAT

logger = logging.getLogger("oceanscope_ml.evaluation.argo_validation")

# Pre-registered headline settings (PRD_AI_ML §1.1, §7). Changing them needs a dated entry in
# docs/DECISIONS.md before the run. They live here rather than in experiment.yaml because that
# file feeds compute_config_hash for the stores already being built.
HEADLINE_DEPTH_MIN_M = 50.0
HEADLINE_DEPTH_MAX_M = 300.0
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
GRID_HALF_CELL_DEG = 0.125
QC_GOOD = (1, 2)
ARGO_VARIABLES = (
    "platform_number",
    "cycle_number",
    "time",
    "latitude",
    "longitude",
    "pres",
    "temp",
    "temp_qc",
    "pres_qc",
    "position_qc",
)
MODEL = "model"
COMPETITORS = {"glorys12": "temp_glorys_c", "grep": "temp_grep_c", "armor3d": "temp_armor3d_c"}
MATCH_COLUMNS = [
    "platform_number",
    "cycle",
    "time",
    "lat",
    "lon",
    "depth_m",
    "temp_obs_c",
    "temp_pred_c",
    "temp_glorys_c",
    "temp_grep_c",
    "temp_armor3d_c",
]
DEPTH_COLUMNS = [
    "depth_m",
    "n_profiles",
    "n_obs",
    "mae_c",
    "rmse_c",
    "bias_c",
    "glorys_mae_c",
    "glorys_rmse_c",
    "grep_mae_c",
    "grep_rmse_c",
    "armor3d_mae_c",
    "armor3d_rmse_c",
]
HEADLINE_COLUMNS = [
    "period",
    "product",
    "n_floats",
    "n_obs",
    "rmse_c",
    "ci_low_c",
    "ci_high_c",
    "diff_vs_model_c",
    "diff_ci_low_c",
    "diff_ci_high_c",
]


@dataclass(frozen=True)
class ArgoResult:
    matches: pd.DataFrame
    per_depth: pd.DataFrame
    headline: pd.DataFrame


def _in_bay_of_bengal(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (lon >= BAY_OF_BENGAL_MIN_LON) & (lat >= EQUATORIAL_MAX_LAT)


def _nearest_cell(grid: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest index on an ascending grid, and whether x lies within half a cell of its centre."""
    edges = (grid[1:] + grid[:-1]) / 2.0
    idx = np.searchsorted(edges, x)
    inside = np.abs(grid[idx] - x) <= GRID_HALF_CELL_DEG + 1e-9
    return idx, inside


def _interp_depth(levels: np.ndarray, depths: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear interpolation in depth of ``levels`` [n, n_depth] at ``z`` [n]; NaN outside."""
    k = np.clip(np.searchsorted(depths, z, side="right"), 1, depths.size - 1)
    rows = np.arange(z.size)
    v0 = levels[rows, k - 1]
    v1 = levels[rows, k]
    w = (z - depths[k - 1]) / (depths[k] - depths[k - 1])
    inside = (z >= depths[0]) & (z <= depths[-1])
    return np.where(inside, v0 + w * (v1 - v0), np.nan)


def _cells(field: xr.DataArray, i_lat: np.ndarray, i_lon: np.ndarray) -> np.ndarray:
    """[n, n_depth] profiles at the given cells of one day's (depth, lat, lon) field."""
    values = field.transpose("depth", "lat", "lon").values
    return np.asarray(values[:, i_lat, i_lon].T, dtype=float)


def load_profiles(
    paths: list[Path], holdout: set[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Profile levels of held-out floats in the validation window, with QC flags 1–2 (PRD §7).

    Only the time of each row is read for the whole file. The other variables are loaded for
    validation-window rows alone, so profiles from the sealed test year are never loaded here.
    """
    frames = []
    window_end = end + pd.Timedelta(days=1)
    for path in paths:
        with xr.open_dataset(path) as ds:
            missing = [v for v in ARGO_VARIABLES if v not in ds.variables]
            if missing:
                raise ValueError(f"{path.name} lacks {missing}; PRD_AI_ML §7 needs the QC flags")
            time_values = pd.to_datetime(ds["time"].values)
            rows = np.flatnonzero((time_values >= start) & (time_values < window_end))
            sub = ds[list(ARGO_VARIABLES)].isel({ds["time"].dims[0]: rows})
            frames.append(sub.to_dataframe().reset_index(drop=True))
    df = pd.concat(frames, ignore_index=True)
    df["platform_number"] = df["platform_number"].astype(str).str.strip()
    qc_ok = pd.Series(True, index=df.index)
    for col in ("temp_qc", "pres_qc", "position_qc"):
        qc_ok &= pd.to_numeric(df[col], errors="coerce").isin(QC_GOOD)
    keep = (
        df["platform_number"].isin(holdout)
        & qc_ok
        & np.isfinite(df["temp"])
        & np.isfinite(df["pres"])
        & (df["pres"] >= 0)
    )
    out = df.loc[keep, ["platform_number", "cycle_number", "time", "latitude", "longitude"]]
    out = out.assign(pres=df.loc[keep, "pres"], temp_obs_c=df.loc[keep, "temp"])
    return out.rename(columns={"cycle_number": "cycle", "latitude": "lat", "longitude": "lon"})


def match_profiles(
    profiles: pd.DataFrame, predictions: xr.Dataset, references: xr.Dataset
) -> pd.DataFrame:
    """One row per held-out observation that the model covers (see module docstring)."""
    depths = np.asarray(predictions["depth"].values, dtype=float)
    lat_grid = np.asarray(predictions["lat"].values, dtype=float)
    lon_grid = np.asarray(predictions["lon"].values, dtype=float)
    if np.any(np.diff(lat_grid) <= 0) or np.any(np.diff(lon_grid) <= 0):
        raise ValueError("The prediction grid must be ascending in lat and lon")
    days = pd.DatetimeIndex(predictions["time"].values).normalize()
    obs = profiles.reset_index(drop=True)
    if obs.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    day = pd.DatetimeIndex(obs["time"].values).normalize()
    li, lat_ok = _nearest_cell(lat_grid, obs["lat"].to_numpy(float))
    lo, lon_ok = _nearest_cell(lon_grid, obs["lon"].to_numpy(float))
    keep = np.asarray(lat_ok & lon_ok & day.isin(days))
    picked = pd.DataFrame({"row": np.flatnonzero(keep), "day": day[keep]})

    pred = predictions[["temp_pred", "temp_glorys", "coverage"]]
    refs = references[["grep_mean", "armor3d"]]
    z_all = obs["pres"].to_numpy(float)
    pieces: list[pd.DataFrame] = []
    for d, group in picked.groupby("day"):
        rows = group["row"].to_numpy()
        t = int(days.get_loc(d))
        i_lat, i_lon = li[rows], lo[rows]
        z = z_all[rows]
        pred_day = pred.isel(time=t).load()
        ref_day = refs.sel(time=d).load()
        cov = np.asarray(pred_day["coverage"].values)[i_lat, i_lon]
        model = _interp_depth(_cells(pred_day["temp_pred"], i_lat, i_lon), depths, z)
        glorys = _interp_depth(_cells(pred_day["temp_glorys"], i_lat, i_lon), depths, z)
        grep = _interp_depth(_cells(ref_day["grep_mean"], i_lat, i_lon), depths, z)
        armor = _interp_depth(_cells(ref_day["armor3d"], i_lat, i_lon), depths, z)
        ok = (cov == 1) & np.isfinite(model)
        level = depths[np.abs(z[:, None] - depths[None, :]).argmin(axis=1)]
        pieces.append(
            pd.DataFrame(
                {
                    "platform_number": obs["platform_number"].to_numpy(str)[rows][ok],
                    "cycle": obs["cycle"].to_numpy()[rows][ok],
                    "time": obs["time"].to_numpy()[rows][ok],
                    "lat": obs["lat"].to_numpy(float)[rows][ok],
                    "lon": obs["lon"].to_numpy(float)[rows][ok],
                    "depth_m": level[ok],
                    "temp_obs_c": obs["temp_obs_c"].to_numpy(float)[rows][ok],
                    "temp_pred_c": model[ok],
                    "temp_glorys_c": glorys[ok],
                    "temp_grep_c": grep[ok],
                    "temp_armor3d_c": armor[ok],
                }
            )
        )
    if not pieces:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    matches = pd.concat(pieces, ignore_index=True)
    for col in COMPETITORS.values():
        matches[col] = matches[col].astype("Float64")
    return matches[MATCH_COLUMNS]


def _error_stats(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    err = pred - ref
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2))), float(np.mean(err))


def depth_table(common: pd.DataFrame, depths: np.ndarray) -> pd.DataFrame:
    """Per-level errors of the model and each reference on the common matchups."""
    rows: list[dict[str, Any]] = []
    for d in depths:
        sub = common[np.isclose(common["depth_m"].to_numpy(float), d)]
        row: dict[str, Any] = {
            "depth_m": float(d),
            "n_profiles": int(sub[["platform_number", "cycle"]].drop_duplicates().shape[0]),
            "n_obs": len(sub),
        }
        ref = sub["temp_obs_c"].to_numpy(float)
        for prefix, col in (("", "temp_pred_c"), *((k + "_", v) for k, v in COMPETITORS.items())):
            if len(sub):
                mae, rmse, bias = _error_stats(sub[col].to_numpy(float), ref)
            else:
                mae, rmse, bias = np.nan, np.nan, np.nan
            row[f"{prefix}mae_c"] = round(mae, 3)
            row[f"{prefix}rmse_c"] = round(rmse, 3)
            if prefix == "":
                row["bias_c"] = round(bias, 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=DEPTH_COLUMNS)


def _unavailable_rows(period: str) -> pd.DataFrame:
    rows = [
        {
            "period": period,
            "product": product,
            "n_floats": 0,
            "n_obs": 0,
            **{c: np.nan for c in HEADLINE_COLUMNS[4:]},
        }
        for product in (MODEL, *COMPETITORS)
    ]
    return pd.DataFrame(rows, columns=HEADLINE_COLUMNS)


def headline_rows(band: pd.DataFrame, period: str) -> pd.DataFrame:
    """The §1.1 rows for one period, with 95% float-bootstrap intervals (paired draws)."""
    if band.empty:
        return _unavailable_rows(period)
    ref = band["temp_obs_c"].to_numpy(float)
    floats, f_idx = np.unique(band["platform_number"].to_numpy(str), return_inverse=True)
    n_floats = floats.size
    n_per_float = np.bincount(f_idx, minlength=n_floats).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n_floats, size=(BOOTSTRAP_DRAWS, n_floats))
    weights = np.stack([np.bincount(d, minlength=n_floats) for d in draws]).astype(float)

    products = {MODEL: "temp_pred_c", **COMPETITORS}
    point: dict[str, float] = {}
    boot: dict[str, np.ndarray] = {}
    for product, col in products.items():
        sse = np.bincount(f_idx, weights=(band[col].to_numpy(float) - ref) ** 2, minlength=n_floats)
        point[product] = float(np.sqrt(sse.sum() / len(band)))
        boot[product] = np.sqrt(weights @ sse / (weights @ n_per_float))

    rows: list[dict[str, Any]] = []
    for product in products:
        ci_low, ci_high = np.percentile(boot[product], CI_PERCENTILES)
        row: dict[str, Any] = {
            "period": period,
            "product": product,
            "n_floats": n_floats,
            "n_obs": len(band),
            "rmse_c": point[product],
            "ci_low_c": float(ci_low),
            "ci_high_c": float(ci_high),
        }
        if product == MODEL:
            row.update(diff_vs_model_c=0.0, diff_ci_low_c=0.0, diff_ci_high_c=0.0)
        else:
            diff_low, diff_high = np.percentile(boot[MODEL] - boot[product], CI_PERCENTILES)
            row.update(
                diff_vs_model_c=point[MODEL] - point[product],
                diff_ci_low_c=float(diff_low),
                diff_ci_high_c=float(diff_high),
            )
        rows.append(row)
    out = pd.DataFrame(rows, columns=HEADLINE_COLUMNS)
    num = HEADLINE_COLUMNS[4:]
    out[num] = out[num].round(6)
    return out


def headline_verdict(headline: pd.DataFrame, period: str) -> bool | None:
    """Met only if every competitor's upper 95% bound on (model − competitor) RMSE is below 0.

    Same rule as the backend's /headline (PRD_BACKEND §4). None when a competitor has no data.
    """
    rows = headline[headline["period"] == period]
    competitors = rows[rows["product"] != MODEL]
    if len(competitors) < len(COMPETITORS) or competitors["diff_ci_high_c"].isna().any():
        return None
    return bool((competitors["diff_ci_high_c"] < 0).all())


def validate(profiles: pd.DataFrame, predictions: xr.Dataset, references: xr.Dataset) -> ArgoResult:
    """Matches, per-depth table and §1.1 headline for validation dates (see module docstring)."""
    depths = np.asarray(predictions["depth"].values, dtype=float)
    matches = match_profiles(profiles, predictions, references)
    common = matches.dropna(subset=list(COMPETITORS.values()))
    per_depth = depth_table(common, depths)
    in_band = (common["depth_m"] >= HEADLINE_DEPTH_MIN_M) & (
        common["depth_m"] <= HEADLINE_DEPTH_MAX_M
    )
    in_band &= _in_bay_of_bengal(common["lat"].to_numpy(float), common["lon"].to_numpy(float))
    headline = pd.concat(
        [headline_rows(common[in_band], "val"), _unavailable_rows("test")], ignore_index=True
    )
    return ArgoResult(matches=matches, per_depth=per_depth, headline=headline)


def _load_holdout(path: Path) -> set[str]:
    floats = json.loads(Path(path).read_text(encoding="utf-8")).get("held_out_floats") or []
    if not floats:
        raise ValueError(f"{path} lists no held-out floats (PRD_AI_ML §7)")
    return {str(f).strip() for f in floats}


def run(
    config: ExperimentConfig,
    predictions: Path,
    store_paths: list[Path],
    argo_paths: list[Path],
    holdout_path: Path,
    stats_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Validate one run against held-out floats; write the three metric files, return paths."""
    pred = xr.open_zarr(predictions)
    run_id = str(pred.attrs["run_id"])
    dates = pd.DatetimeIndex(pred["time"].values).normalize()
    check_no_test_leakage(dates, config.splits.test)
    val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
    if dates.size == 0 or dates.min() < val_start or dates.max() > val_end:
        raise ValueError("The prediction store must hold validation dates only")
    if not np.allclose(np.asarray(pred["depth"].values, dtype=float), config.depths_m):
        raise ValueError("Prediction depths differ from configs/experiment.yaml depths_m")

    profiles = load_profiles(argo_paths, _load_holdout(holdout_path), val_start, val_end)
    logger.info("Held-out profile levels in the validation window: %d", len(profiles))
    references = open_parts(store_paths, stats_path)[["grep_mean", "armor3d"]]
    result = validate(profiles, pred, references)

    metrics_dir = Path(out_dir or config.paths.outputs) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "argo": metrics_dir / f"argo_{run_id}.csv",
        "argo_matches": metrics_dir / f"argo_matches_{run_id}.parquet",
        "headline": metrics_dir / f"{run_id}_headline.csv",
    }
    result.per_depth.to_csv(paths["argo"], index=False)
    result.matches.to_parquet(paths["argo_matches"], index=False)
    result.headline.to_csv(paths["headline"], index=False)

    verdict = headline_verdict(result.headline, "val")
    label = {True: "Met", False: "Not met", None: "Not yet evaluated"}[verdict]
    logger.info(
        "Run %s: %d matched observations, headline (val) %s", run_id, len(result.matches), label
    )
    return paths


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARGO validation and §1.1 headline for one run.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--argo", nargs="+", required=True, help="ARGO profile NetCDF files")
    parser.add_argument("--holdout", required=True, help="argo_eval_floats.json")
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    run(
        config,
        Path(args.predictions),
        [Path(p) for p in args.store],
        [Path(p) for p in args.argo],
        Path(args.holdout),
        stats_path=Path(args.stats) if args.stats else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
