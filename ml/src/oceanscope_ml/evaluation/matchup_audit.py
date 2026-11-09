"""Audit of the ARGO matchup: duplicates, pressure vs depth, and the time-matching window.

Usage (from ml/):

    python -m oceanscope_ml.evaluation.matchup_audit --config configs/experiment.yaml \
        --predictions <predictions_run.zarr> --store <part stores...> [--stats stats.zarr] \
        --argo <argo_YYYY-MM.nc...> --holdout configs/argo_eval_floats.json \
        [--half dev] [--out-dir outputs]

Checks things that can look like model error but come from the matching (PRD_AI_ML §7):

1. **Duplicates** among the held-out profile levels: repeated (float, cycle, pressure) rows, one
   cycle reported at more than one time, and identical (time, position) from different floats.
2. **Pressure used as depth.** The matchup reads pressure (dbar) as depth (m). The Saunders
   (1981) formula gives the true depth; the table lists the difference at each model level and
   the temperature effect it implies with each profile's own vertical gradient.
3. **Time window.** The model is a daily field and a float samples at one moment. The match is
   repeated with every profile moved one day earlier and one day later; the "3-day" row averages
   the three matches. Each product uses the same shifted day, so the comparison stays fair. The
   primary window stays the same day (PRD_AI_ML §7); these rows are a sensitivity check and
   never choose anything.

Only the chosen half of the held-out floats is read (``--half``, default dev). Writes
``<out>/metrics/matchup_audit_<label>.csv`` (time-window rows) and
``matchup_audit_<label>.json`` (duplicates and pressure-depth table).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import check_no_test_leakage
from oceanscope_ml.evaluation.argo_validation import (
    HALVES,
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    _in_bay_of_bengal,
    _load_holdout,
    holdout_halves,
    load_profiles,
    match_profiles,
)

logger = logging.getLogger("oceanscope_ml.evaluation.matchup_audit")

PRODUCTS = {"model": "temp_pred_c", "glorys12": "temp_glorys_c", "grep": "temp_grep_c",
            "armor3d": "temp_armor3d_c"}  # fmt: skip
KEYS = ["platform_number", "cycle", "depth_m", "temp_obs_c"]


def saunders_depth(pres: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Depth (m) from pressure (dbar), Saunders (1981)."""
    s2 = np.sin(np.deg2rad(lat)) ** 2
    c1 = (5.92 + 5.25 * s2) * 1e-3
    return (1 - c1) * pres - 2.21e-6 * pres**2


def duplicates(levels: pd.DataFrame) -> dict[str, int]:
    prof = levels.drop_duplicates(["platform_number", "cycle"])
    times = levels.groupby(["platform_number", "cycle"])["time"].nunique()
    same_spot = prof.groupby(["time", "lat", "lon"])["platform_number"].nunique()
    return {
        "levels": int(len(levels)),
        "profiles": int(len(prof)),
        "floats": int(levels["platform_number"].nunique()),
        "repeated_float_cycle_pressure_rows": int(
            levels.duplicated(["platform_number", "cycle", "pres"]).sum()
        ),
        "cycles_with_more_than_one_time": int((times > 1).sum()),
        "time_position_shared_by_several_floats": int((same_spot > 1).sum()),
    }


def depth_effect(levels: pd.DataFrame, model_depths: np.ndarray) -> list[dict[str, float]]:
    """Per model level: mean (pressure - depth) and RMS temperature effect of reading pressure
    as depth, with each profile's own gradient dT/dp."""
    df = levels.sort_values(["platform_number", "cycle", "pres"]).copy()
    df["dz"] = df["pres"] - saunders_depth(df["pres"].to_numpy(float), df["lat"].to_numpy(float))
    grads = []
    for _, g in df.groupby(["platform_number", "cycle"], sort=False):
        p, t = g["pres"].to_numpy(float), g["temp_obs_c"].to_numpy(float)
        grads.append(
            np.gradient(t, p) if p.size > 1 and np.all(np.diff(p) > 0) else np.full(p.size, np.nan)
        )
    df["grad"] = np.concatenate(grads) if grads else []
    df["effect"] = df["grad"] * df["dz"]
    level = model_depths[np.abs(df["pres"].to_numpy(float)[:, None] - model_depths[None]).argmin(1)]
    rows = []
    for d in model_depths:
        sub = df[level == d]
        eff = sub["effect"].to_numpy(float)
        eff = eff[np.isfinite(eff)]
        rows.append(
            {
                "depth_m": float(d),
                "n_obs": int(len(sub)),
                "mean_pres_minus_depth_m": float(sub["dz"].mean()) if len(sub) else np.nan,
                "rms_temp_effect_c": float(np.sqrt(np.mean(eff**2))) if eff.size else np.nan,
            }
        )
    return rows


def _rmse_rows(m: pd.DataFrame, shift: str) -> list[dict]:
    band = (
        _in_bay_of_bengal(m["lat"].to_numpy(float), m["lon"].to_numpy(float))
        & (m["depth_m"] >= HEADLINE_DEPTH_MIN_M)
        & (m["depth_m"] <= HEADLINE_DEPTH_MAX_M)
    )
    rows = []
    for group, sel in (("all", np.ones(len(m), bool)), ("bob_50_300", band)):
        for product, col in PRODUCTS.items():
            err = (m.loc[sel, col] - m.loc[sel, "temp_obs_c"]).to_numpy(float)
            err = err[np.isfinite(err)]
            rmse = float(np.sqrt(np.mean(err**2))) if err.size else np.nan
            rows.append(
                {"window": shift, "group": group, "product": product, "n_obs": int(err.size),
                 "rmse_c": rmse}
            )  # fmt: skip
    return rows


def time_window(profiles: pd.DataFrame, pred: xr.Dataset, references: xr.Dataset) -> pd.DataFrame:
    shifted = {}
    for days in (-1, 0, 1):
        p = profiles.copy()
        p["time"] = pd.to_datetime(p["time"]) + pd.Timedelta(days=days)
        m = match_profiles(p, pred, references)
        m["time"] = pd.to_datetime(m["time"]) - pd.Timedelta(days=days)
        m["dup"] = m.groupby(KEYS).cumcount()
        shifted[days] = m
    rows = []
    for days, m in shifted.items():
        rows += _rmse_rows(m, f"{days:+d} day" if days else "same day")
    joined = shifted[0]
    cols = list(PRODUCTS.values())
    for days in (-1, 1):
        other = shifted[days][KEYS + ["dup"] + cols].rename(columns={c: f"{c}{days}" for c in cols})
        joined = joined.merge(other, on=KEYS + ["dup"], how="inner")
    mean = joined.copy()
    for c in cols:
        mean[c] = joined[[c, f"{c}-1", f"{c}1"]].mean(axis=1, skipna=False)
    rows += _rmse_rows(mean, "3-day mean")
    return pd.DataFrame(rows)


def run(
    config: ExperimentConfig,
    predictions: Path,
    store_paths: list[Path],
    argo_paths: list[Path],
    holdout_path: Path,
    stats_path: Path | None = None,
    out_dir: Path | None = None,
    half: str = "dev",
) -> dict[str, Path]:
    if half not in HALVES:
        raise ValueError(f"half must be one of {HALVES}")
    pred = xr.open_zarr(predictions)
    dates = pd.DatetimeIndex(pred["time"].values).normalize()
    check_no_test_leakage(dates, config.splits.test)
    val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
    if dates.size == 0 or dates.min() < val_start or dates.max() > val_end:
        raise ValueError("The prediction store must hold validation dates only")
    run_id = str(pred.attrs["run_id"])
    label = run_id if half == "all" else f"{run_id}_{half}"
    profiles = load_profiles(argo_paths, _load_holdout(holdout_path), val_start, val_end)
    if half != "all":
        halves = holdout_halves(profiles)
        keep = set(halves.loc[halves["half"] == half, "platform_number"])
        profiles = profiles[profiles["platform_number"].isin(keep)].reset_index(drop=True)
    references = open_parts(store_paths, stats_path)[["grep_mean", "armor3d"]]
    summary = {
        "duplicates": duplicates(profiles),
        "pressure_vs_depth": depth_effect(profiles, np.asarray(pred["depth"].values, float)),
    }
    window = time_window(profiles, pred, references)
    metrics = Path(out_dir or config.paths.outputs) / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    paths = {
        "window": metrics / f"matchup_audit_{label}.csv",
        "summary": metrics / f"matchup_audit_{label}.json",
    }
    window.round(4).to_csv(paths["window"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Duplicates: %s", summary["duplicates"])
    return paths


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the ARGO matchup.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--argo", nargs="+", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--half", default="dev", choices=HALVES)
    parser.add_argument("--out-dir", default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    paths = run(
        load_config(args.config),
        Path(args.predictions),
        [Path(p) for p in args.store],
        [Path(p) for p in args.argo],
        Path(args.holdout),
        stats_path=Path(args.stats) if args.stats else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        half=args.half,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}))


if __name__ == "__main__":
    main()
