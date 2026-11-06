"""Ensemble weights fitted on non-held-out ARGO floats (validation period).

Usage (from ml/):

    python -m oceanscope_ml.inference.ensemble_weights --config configs/experiment.yaml \
        --predictions <predictions_a.zarr> <predictions_b.zarr> ... \
        --argo <argo_YYYY-MM.nc...> --holdout configs/argo_eval_floats.json \
        --run-id <name> [--out-dir outputs] [--step 0.05]

At every level of every non-held-out profile that all members cover, the ensemble error is
sum_k w_k (T_pred_k - T_argo), since the weights sum to 1. One set of weights (>= 0, sum 1, on a
grid of ``step``) is picked by the lowest RMSE over all those levels. It is one set for the
whole domain and every depth on purpose: per-depth or per-region weights would be fitted on far
fewer floats and overfit. Held-out floats are never read.

Writes ``<out>/metrics/ensemble_weights_<run_id>.csv`` (every grid point: weights, RMSE, Bay of
Bengal 50-300 m RMSE) and ``ensemble_weights_<run_id>.json`` with the chosen weights, for
``inference.ensemble --weights``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.argo import profile_levels, read_levels
from oceanscope_ml.data.transform import check_no_test_leakage
from oceanscope_ml.evaluation.evaluate import BAY_OF_BENGAL_MIN_LON, EQUATORIAL_MAX_LAT
from oceanscope_ml.inference.argo_correct import residuals

logger = logging.getLogger("oceanscope_ml.inference.ensemble_weights")

KEYS = ["platform", "day", "lat", "lon"]


def simplex_grid(n: int, step: float) -> np.ndarray:
    """All weight vectors of length n on a grid of ``step`` that are >= 0 and sum to 1."""
    k = int(round(1.0 / step))
    rows = [c for c in itertools.product(range(k + 1), repeat=n - 1) if sum(c) <= k]
    return np.array([[*c, k - sum(c)] for c in rows], dtype=float) / k


def member_errors(
    profiles: pd.DataFrame, members: list[xr.Dataset]
) -> tuple[np.ndarray, np.ndarray]:
    """[n_levels, n_members] error (pred - argo) where every member is valid, and the Bay of
    Bengal 50-300 m flag of each level."""
    frames = []
    for k, ds in enumerate(members):
        res = residuals(profiles, ds)
        df = pd.DataFrame({"platform": res.platform, "day": res.day, "lat": res.lat,
                           "lon": res.lon})  # fmt: skip
        df[f"r{k}"] = list(res.values)
        frames.append(df.drop_duplicates(KEYS))
    joined = frames[0]
    for df in frames[1:]:
        joined = joined.merge(df, on=KEYS, how="inner")
    if joined.empty:
        raise ValueError("No profile is covered by every member")
    depths = np.asarray(members[0]["depth"].values, dtype=float)
    stack = np.stack([np.vstack(joined[f"r{k}"].to_numpy()) for k in range(len(members))], -1)
    err = -stack.reshape(-1, len(members))  # residual = argo - pred, error = pred - argo
    cells = (joined["lon"].to_numpy() >= BAY_OF_BENGAL_MIN_LON) & (
        joined["lat"].to_numpy() >= EQUATORIAL_MAX_LAT
    )
    band = (cells[:, None] & ((depths >= 50) & (depths <= 300))[None, :]).reshape(-1)
    ok = np.isfinite(err).all(axis=1)
    return err[ok], band[ok]


def choose(err: np.ndarray, band: np.ndarray, step: float) -> tuple[np.ndarray, pd.DataFrame]:
    grid = simplex_grid(err.shape[1], step)
    e = err @ grid.T  # [n, G]
    rmse = np.sqrt(np.mean(e**2, axis=0))
    rmse_band = np.sqrt(np.mean(e[band] ** 2, axis=0)) if band.any() else np.full(len(grid), np.nan)
    table = pd.DataFrame(grid, columns=[f"w{k}" for k in range(err.shape[1])])
    table["rmse_c"] = rmse
    table["bob_band_rmse_c"] = rmse_band
    return grid[int(np.argmin(rmse))], table


def run(
    config: ExperimentConfig,
    predictions: list[Path],
    argo_paths: list[Path],
    holdout: set[str],
    run_id: str,
    out_dir: Path | None = None,
    step: float = 0.05,
) -> list[float]:
    members = [xr.open_zarr(p) for p in predictions]
    val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
    for ds in members:
        dates = pd.DatetimeIndex(ds["time"].values).normalize()
        check_no_test_leakage(dates, config.splits.test)
        if dates.size == 0 or dates.min() < val_start or dates.max() > val_end:
            raise ValueError("Members must hold validation dates only")
    depths = [float(d) for d in members[0]["depth"].values]
    profiles = profile_levels(read_levels(argo_paths, val_start, val_end), depths, holdout)
    if profiles["platform_number"].isin(holdout).any():
        raise AssertionError("A held-out float reached the ensemble weights")
    err, band = member_errors(profiles, members)
    weights, table = choose(err, band, step)
    equal = np.full(len(members), 1.0 / len(members))
    e_eq = err @ equal
    e_w = err @ weights
    logger.info("Levels used: %d (Bay of Bengal 50-300 m: %d)", err.shape[0], int(band.sum()))
    logger.info("Equal weights: RMSE %.3f C (BoB band %.3f C)", np.sqrt(np.mean(e_eq**2)),
                np.sqrt(np.mean(e_eq[band] ** 2)))  # fmt: skip
    logger.info("Chosen %s: RMSE %.3f C (BoB band %.3f C)", np.round(weights, 3).tolist(),
                np.sqrt(np.mean(e_w**2)), np.sqrt(np.mean(e_w[band] ** 2)))  # fmt: skip
    out = Path(out_dir or config.paths.outputs) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(out / f"ensemble_weights_{run_id}.csv", index=False)
    chosen = {"weights": [float(w) for w in weights], "members": [str(p) for p in predictions]}
    (out / f"ensemble_weights_{run_id}.json").write_text(json.dumps(chosen), encoding="utf-8")
    return [float(w) for w in weights]


def _load_holdout(path: Path) -> set[str]:
    floats = json.loads(Path(path).read_text(encoding="utf-8")).get("held_out_floats") or []
    if not floats:
        raise ValueError(f"{path} lists no held-out floats (PRD_AI_ML §7)")
    return {str(f).strip() for f in floats}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit ensemble weights on non-held-out floats.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--argo", nargs="+", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--step", type=float, default=0.05)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    weights = run(
        load_config(args.config),
        [Path(p) for p in args.predictions],
        [Path(p) for p in args.argo],
        _load_holdout(Path(args.holdout)),
        args.run_id,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        step=args.step,
    )
    print(json.dumps({"weights": weights}))


if __name__ == "__main__":
    main()
