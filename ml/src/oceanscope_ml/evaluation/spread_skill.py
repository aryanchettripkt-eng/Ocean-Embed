"""Is the ensemble spread a usable uncertainty? Spread-vs-error at held-out ARGO floats.

Usage (from ml/):

    python -m oceanscope_ml.evaluation.spread_skill \
        --matches argo_matches_<a>_dev.parquet argo_matches_<b>_dev.parquet ... \
        --run-id <ensemble>_dev [--out-dir outputs]

Each ``--matches`` file is one member's ARGO match table (``evaluation.argo_validation``), made
with the same floats. At every observation all members cover, the ensemble mean is the mean of
the members (the same as the ensemble store, since the average and the depth interpolation are
both linear) and the spread is the members' standard deviation (ddof = 0).

Per depth, plus the Bay of Bengal 50-300 m band (``bob_50_300``) and ``all``, it writes
``<out>/metrics/spread_skill_<run_id>.csv`` with: ``n_obs``, ``rmse_c`` (ensemble mean vs ARGO),
``spread_rms_c``, ``ratio`` (rmse / spread; 1 means calibrated, above 1 means the spread is too
small), ``frac_within_1sd`` and ``frac_within_2sd`` (Gaussian: 0.68, 0.95), and
``corr_spread_abs_err`` (do larger spreads go with larger errors?).

``frac_within_*_scaled`` repeat the coverage with the spread multiplied by ``scale``, the
overall ratio fitted on half of these floats (``fit_half``: by float, fixed seed) and checked on
the other half, so the scaled numbers are not scored on the floats that fitted them.

Nothing here reads the test year: match files only hold validation dates.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from oceanscope_ml.evaluation.argo_validation import (
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    _in_bay_of_bengal,
)

logger = logging.getLogger("oceanscope_ml.evaluation.spread_skill")

KEYS = ["platform_number", "cycle", "time", "depth_m", "temp_obs_c"]
FIT_SEED = 11
COLUMNS = [
    "group",
    "n_obs",
    "rmse_c",
    "spread_rms_c",
    "ratio",
    "frac_within_1sd",
    "frac_within_2sd",
    "corr_spread_abs_err",
    "scale",
    "frac_within_1sd_scaled",
    "frac_within_2sd_scaled",
]


def combine(members: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per observation that every member covers: obs, mean, spread, lat, lon."""
    if len(members) < 2:
        raise ValueError("Spread needs at least two members")
    frames = []
    for k, m in enumerate(members):
        df = m[KEYS + ["lat", "lon", "temp_pred_c"]].copy()
        df["platform_number"] = df["platform_number"].astype(str)
        df["dup"] = df.groupby(KEYS).cumcount()
        df = df.rename(columns={"temp_pred_c": f"pred_{k}"})
        if k:
            df = df.drop(columns=["lat", "lon"])
        frames.append(df)
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=KEYS + ["dup"], how="inner")
    preds = out[[f"pred_{k}" for k in range(len(members))]].to_numpy(float)
    ok = np.isfinite(preds).all(axis=1) & np.isfinite(out["temp_obs_c"].to_numpy(float))
    out = out.loc[ok].reset_index(drop=True)
    preds = preds[ok]
    out["mean_c"] = preds.mean(axis=1)
    out["spread_c"] = preds.std(axis=1)
    out["err_c"] = out["mean_c"] - out["temp_obs_c"]
    return out[["platform_number", "lat", "lon", "depth_m", "temp_obs_c", "mean_c", "spread_c",
                "err_c"]]  # fmt: skip


def _fit_floats(platforms: np.ndarray, seed: int = FIT_SEED) -> set[str]:
    floats = np.sort(np.unique(platforms))
    rng = np.random.default_rng(seed)
    pick = rng.permutation(floats.size)[: int(np.ceil(floats.size / 2))]
    return {str(f) for f in floats[pick]}


def _row(group: str, df: pd.DataFrame, scale: float) -> dict:
    err = df["err_c"].to_numpy(float)
    sd = df["spread_c"].to_numpy(float)
    n = err.size
    if n == 0:
        return {"group": group, "n_obs": 0}
    rmse = float(np.sqrt(np.mean(err**2)))
    spread = float(np.sqrt(np.mean(sd**2)))
    a = np.abs(err)
    corr = float(np.corrcoef(sd, a)[0, 1]) if n > 2 and sd.std() > 0 and a.std() > 0 else np.nan
    return {
        "group": group,
        "n_obs": n,
        "rmse_c": rmse,
        "spread_rms_c": spread,
        "ratio": rmse / spread if spread > 0 else np.nan,
        "frac_within_1sd": float(np.mean(a <= sd)),
        "frac_within_2sd": float(np.mean(a <= 2 * sd)),
        "corr_spread_abs_err": corr,
        "scale": scale,
        "frac_within_1sd_scaled": float(np.mean(a <= scale * sd)),
        "frac_within_2sd_scaled": float(np.mean(a <= 2 * scale * sd)),
    }


def spread_skill(combined: pd.DataFrame) -> pd.DataFrame:
    """The table described in the module docstring. Scaled columns use only check-half rows."""
    fit = combined["platform_number"].isin(_fit_floats(combined["platform_number"].to_numpy()))
    f = combined[fit]
    spread = float(np.sqrt(np.mean(f["spread_c"] ** 2))) if len(f) else np.nan
    scale = float(np.sqrt(np.mean(f["err_c"] ** 2)) / spread) if spread and spread > 0 else np.nan
    rows = []
    groups: list[tuple[str, pd.Series]] = [
        (f"{d:g}", np.isclose(combined["depth_m"], d)) for d in np.unique(combined["depth_m"])
    ]
    band = (
        _in_bay_of_bengal(combined["lat"].to_numpy(float), combined["lon"].to_numpy(float))
        & (combined["depth_m"] >= HEADLINE_DEPTH_MIN_M)
        & (combined["depth_m"] <= HEADLINE_DEPTH_MAX_M)
    )
    groups += [("bob_50_300", band), ("all", pd.Series(True, index=combined.index))]
    for name, sel in groups:
        raw = _row(name, combined[sel], scale)
        check = _row(name, combined[sel & ~fit], scale)
        for col in ("frac_within_1sd_scaled", "frac_within_2sd_scaled"):
            raw[col] = check.get(col, np.nan)
        rows.append(raw)
    return pd.DataFrame(rows, columns=COLUMNS)


def run(match_paths: list[Path], run_id: str, out_dir: Path) -> Path:
    members = [pd.read_parquet(p) for p in match_paths]
    combined = combine(members)
    logger.info("Observations covered by all %d members: %d", len(members), len(combined))
    table = spread_skill(combined)
    dest = out_dir / "metrics" / f"spread_skill_{run_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(dest, index=False)
    return dest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ensemble spread vs error at ARGO floats.")
    parser.add_argument("--matches", nargs="+", required=True, help="Members' ARGO match files")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="outputs")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    print(run([Path(p) for p in args.matches], args.run_id, Path(args.out_dir)))


if __name__ == "__main__":
    main()
