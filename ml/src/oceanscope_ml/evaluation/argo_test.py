"""2021 ARGO headline: score one corrected ensemble on the sealed test year, once.

Usage (from ml/), only when Aryan says to run the 2021 ARGO test (DECISIONS 2026-11-06):

    python -m oceanscope_ml.evaluation.argo_test --config configs/experiment.yaml \
        --member main2_ai=<best.pt> --member main2_ft=<best_ft.pt> --member main3=<best.pt> \
        --weights 0.1,0,0.9 --base-id ens3 \
        --residual-model <models/residual_model_ens3w_rm2.txt> \
        --oi-settings <models/argo_correct_ens3w_rm2_oi.json> \
        --store <parts...> --stats <stats.zarr> --argo <argo_2020-12.nc ... argo_2021-12.nc> \
        --holdout configs/argo_eval_floats.json --owner-go

Besides ``final_test`` this is the only code that reads the test year. Everything that was fitted
or tuned (member checkpoints, ensemble weights, the tree model, the per-depth correction plan and
its guards) comes in fixed, from runs on the validation years; nothing is fitted on 2021. The
2021 profiles of *non-held-out* floats are used the way live mode would use them: as the members'
ARGO inputs, and as the residuals the tree's nearby-float features and the per-depth correction
spread around. Held-out floats are only ever scored.

Steps:

1. Predict 2021 with each member (``predict_store(split="test", allow_test=True)``).
2. ``<base>_test``: the plain mean (the live recipe, for reference); ``<base>w_test``: the
   weighted mean.
3. The saved tree model on ``<base>w_test`` (``<base>w_rm_test``), then the saved per-depth plan
   with the same guards on top (``<run_id>_test``, run_id from the settings file).
4. Score both ``<run_id>_test`` and ``<base>_test`` against **all** held-out floats in 2021
   (the dev/final halves split the validation years; 2021 is untouched for every float).
   Bay of Bengal 50-300 m, float-bootstrap 95% CIs, as ``argo_validation``. A reference with
   values at fewer than 95% of the matchups (GREP ends in 2020) is reported as n/a and left out
   of the verdict, as DECISIONS 2026-10-25 says for the 2021 headline.

Refuses without ``--owner-go`` and when ``metrics/<run_id>_test_headline.csv`` already exists.
Writes ``metrics/<id>_test_headline.csv``, ``argo_<id>_test.csv``,
``argo_matches_<id>_test.parquet`` for both scored runs, and ``argo_test_<run_id>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.argo import profile_levels, read_levels
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.evaluation.argo_validation import (
    COMPETITORS,
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    MODEL,
    _in_bay_of_bengal,
    _load_holdout,
    depth_table,
    headline_rows,
    load_profiles,
    match_profiles,
)
from oceanscope_ml.inference import argo_correct as ac
from oceanscope_ml.inference import residual_model as rm
from oceanscope_ml.inference.ensemble import ensemble
from oceanscope_ml.inference.predict import predict_store

logger = logging.getLogger("oceanscope_ml.evaluation.argo_test")

PERIOD = "test"
MIN_REFERENCE_COVER = 0.95
WINDOW_DAYS = 31  # ARGO residuals before the first test day (the longest correction window)


def check_may_run(metrics_dir: Path, run_id: str, go: bool) -> None:
    if not go:
        raise PermissionError(
            "The 2021 ARGO test is scored once. Run it only when Aryan says run the 2021 ARGO "
            "test (pass --owner-go)."
        )
    done = metrics_dir / f"{run_id}_test_headline.csv"
    if done.exists():
        raise FileExistsError(f"{done} exists: the 2021 ARGO test has already been scored")


def test_window(config: ExperimentConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = (pd.Timestamp(d) for d in config.splits.test)
    return start, end


def _check_test_store(config: ExperimentConfig, path: Path) -> None:
    start, end = test_window(config)
    dates = pd.DatetimeIndex(xr.open_zarr(path)["time"].values).normalize()
    if dates.size == 0 or dates.min() < start or dates.max() > end:
        raise ValueError(f"{path} must hold test dates only")


def input_residuals(
    config: ExperimentConfig, pred_path: Path, argo_paths: list[Path], holdout: set[str]
) -> ac.Residuals:
    """Residuals of the non-held-out floats against a test-year store."""
    start, end = test_window(config)
    pred = xr.open_zarr(pred_path)
    depths = [float(d) for d in pred["depth"].values]
    levels = read_levels(argo_paths, start - pd.Timedelta(days=WINDOW_DAYS), end)
    profiles = profile_levels(levels, depths, holdout)
    res = ac.residuals(profiles, pred)
    if np.isin(res.platform, list(holdout)).any():
        raise AssertionError("A held-out float reached the 2021 corrections")
    logger.info("2021 residuals: %d profiles from %d non-held-out floats", len(res),
                np.unique(res.platform).size)  # fmt: skip
    return res


def apply_tree(
    config: ExperimentConfig,
    pred_path: Path,
    model_path: Path,
    store_paths: list[Path],
    stats_path: Path | None,
    argo_paths: list[Path],
    holdout: set[str],
    run_id: str,
    dest: Path,
) -> Path:
    """The saved tree model on a test-year store (nothing refitted)."""
    _check_test_store(config, pred_path)
    model, meta = rm.load_saved(model_path)
    pred = xr.open_zarr(pred_path)
    surface = rm._surface(store_paths, stats_path, pred) if store_paths else None
    res = input_residuals(config, pred_path, argo_paths, holdout)
    return rm.write_store(pred_path, surface, res, model, list(meta["names"]), run_id, dest,
                          str(meta["features"]), meta.get("max_depth_m"),
                          int(meta.get("window_days", rm.WINDOW_DAYS)))  # fmt: skip


def apply_oi(
    config: ExperimentConfig,
    pred_path: Path,
    settings_path: Path,
    argo_paths: list[Path],
    holdout: set[str],
    run_id: str,
    dest: Path,
) -> Path:
    """The saved per-depth correction plan, with its guards, on a test-year store."""
    _check_test_store(config, pred_path)
    plan, meta = ac.load_settings(settings_path)
    depths = np.asarray(xr.open_zarr(pred_path)["depth"].values, dtype=float)
    if len(plan) != depths.size:
        raise ValueError("The saved plan has a different number of depths")
    res = input_residuals(config, pred_path, argo_paths, holdout)
    if meta.get("min_depth_m"):
        res = ac.drop_shallow(res, depths, float(meta["min_depth_m"]))
    if meta.get("max_depth_m") is not None:
        res = ac.drop_deep(res, depths, float(meta["max_depth_m"]))
    if meta.get("outlier_sigma") is not None:
        res, dropped = ac.drop_outliers(res, float(meta["outlier_sigma"]))
        logger.info("2021 outlier residuals dropped per depth: %s", dropped.tolist())
    return ac.write_store(pred_path, res, plan, run_id, dest)


def available_references(band: pd.DataFrame) -> list[str]:
    """References with values at >= 95% of the band's matchups."""
    if band.empty:
        return []
    return [
        name for name, col in COMPETITORS.items() if band[col].notna().mean() >= MIN_REFERENCE_COVER
    ]


def score(matches: pd.DataFrame, depths: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Per-depth table, headline rows and the references used, for test-year matches."""
    lat, lon = matches["lat"].to_numpy(float), matches["lon"].to_numpy(float)
    in_band = (
        (matches["depth_m"] >= HEADLINE_DEPTH_MIN_M)
        & (matches["depth_m"] <= HEADLINE_DEPTH_MAX_M)
        & _in_bay_of_bengal(lat, lon)
    )
    refs = available_references(matches[in_band])
    out = matches.copy()
    for name, col in COMPETITORS.items():
        if name not in refs:
            out[col] = np.nan  # n/a in every row, never a partial score
    common = out.dropna(subset=[COMPETITORS[r] for r in refs])
    band = common[in_band.loc[common.index]]
    return depth_table(common, depths), headline_rows(band, PERIOD), refs


def verdict(headline: pd.DataFrame, refs: list[str]) -> bool | None:
    """Met only if, for every available reference, the upper 95% bound of
    (model - reference) RMSE is below 0. None without the model row or a reference."""
    rows = headline.set_index("product")
    if not refs or MODEL not in rows.index or not np.isfinite(rows.loc[MODEL, "rmse_c"]):
        return None
    return bool(all(float(rows.loc[r, "diff_ci_high_c"]) < 0 for r in refs))


def score_store(
    config: ExperimentConfig,
    pred_path: Path,
    profiles: pd.DataFrame,
    references: xr.Dataset,
    metrics_dir: Path,
) -> dict[str, Any]:
    _check_test_store(config, pred_path)
    pred = xr.open_zarr(pred_path)
    label = str(pred.attrs["run_id"])
    depths = np.asarray(pred["depth"].values, dtype=float)
    matches = match_profiles(profiles, pred, references)
    per_depth, headline, refs = score(matches, depths)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_depth.to_csv(metrics_dir / f"argo_{label}.csv", index=False)
    matches.to_parquet(metrics_dir / f"argo_matches_{label}.parquet", index=False)
    headline.to_csv(metrics_dir / f"{label}_headline.csv", index=False)
    v = verdict(headline, refs)
    logger.info("2021 %s: %d matches, references %s, headline %s", label, len(matches), refs,
                {True: "Met", False: "Not met", None: "n/a"}[v])  # fmt: skip
    return {"label": label, "references": refs, "verdict": v}


def run_argo_test(
    config: ExperimentConfig,
    members: dict[str, Path],
    weights: list[float],
    base_id: str,
    residual_model: Path,
    oi_settings: Path,
    store_paths: list[Path],
    argo_paths: list[Path],
    holdout_path: Path,
    *,
    go: bool,
    stats_path: Path | None = None,
    out_dir: Path | None = None,
    device_choice: str = "auto",
) -> dict[str, Any]:
    out = Path(out_dir or config.paths.outputs)
    metrics_dir = out / "metrics"
    run_id = str(ac.load_settings(oi_settings)[1]["run_id"])
    check_may_run(metrics_dir, run_id, go)
    if len(members) < 2 or len(weights) != len(members):
        raise ValueError("Pass at least two members and one weight per member")
    holdout = _load_holdout(holdout_path)
    preds = out / "argo_test" / "predictions"
    work = out / "argo_test"

    member_stores = []
    for name, ckpt in members.items():
        member_stores.append(
            predict_store(
                config,
                store_paths,
                ckpt,
                out_dir=work,
                stats_path=stats_path,
                device_choice=device_choice,
                run_id=f"{name}_test",
                split="test",
                allow_test=True,
                argo_paths=argo_paths,
                holdout=holdout,
            )  # fmt: skip
        )
    plain = ensemble(config, member_stores, f"{base_id}_test", out_dir=work, allow_test=True)
    weighted = ensemble(config, member_stores, f"{base_id}w_test", out_dir=work,
                        weights=weights, allow_test=True)  # fmt: skip
    tree_id = f"{base_id}w_rm_test"
    tree = apply_tree(config, weighted, residual_model, store_paths, stats_path, argo_paths,
                      holdout, tree_id, preds / f"predictions_{tree_id}.zarr")  # fmt: skip
    final = apply_oi(config, tree, oi_settings, argo_paths, holdout, f"{run_id}_test",
                     preds / f"predictions_{run_id}_test.zarr")  # fmt: skip

    start, end = test_window(config)
    profiles = load_profiles(argo_paths, holdout, start, end)
    logger.info("Held-out profile levels in 2021: %d from %d floats", len(profiles),
                profiles["platform_number"].nunique())  # fmt: skip
    references = open_parts(store_paths, stats_path)[["grep_mean", "armor3d"]]
    results = [score_store(config, p, profiles, references, metrics_dir) for p in (final, plain)]
    summary = {
        "run_id": run_id,
        "base_id": base_id,
        "members": {k: str(v) for k, v in members.items()},
        "weights": weights,
        "residual_model": str(residual_model),
        "oi_settings": str(oi_settings),
        "results": results,
    }
    (metrics_dir / f"argo_test_{run_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a corrected ensemble on 2021 ARGO, once.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--member", action="append", required=True, help="name=checkpoint")
    parser.add_argument("--weights", required=True, help="Comma-separated, one per member")
    parser.add_argument("--base-id", default="ens3")
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--oi-settings", required=True)
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--argo", nargs="+", required=True, help="Monthly ARGO files, 2020-12..")
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--owner-go", action="store_true",
                        help="Set only when Aryan has said run the 2021 ARGO test.")  # fmt: skip
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    members = {}
    for m in args.member:
        name, _, path = m.partition("=")
        if not path:
            raise SystemExit(f"--member {m}: use name=checkpoint")
        members[name] = Path(path)
    summary = run_argo_test(
        load_config(args.config),
        members,
        [float(w) for w in args.weights.split(",")],
        args.base_id,
        Path(args.residual_model),
        Path(args.oi_settings),
        [Path(p) for p in args.store],
        [Path(p) for p in args.argo],
        Path(args.holdout),
        go=args.owner_go,
        stats_path=Path(args.stats) if args.stats else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        device_choice=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
