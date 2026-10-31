"""Final test: score the sealed test year once (PRD_AI_ML §0 rule 3, §5 stage 8, P8-1).

This is the only code that reads test-period data. It runs once, and only when Aryan says
"run the final test". ``run_final_test`` refuses unless ``go=True``, which the CLI sets only
with ``--owner-go``. It also refuses when the test metrics already exist or the manifest already
says ``test_evaluated``, so the result is fixed after one run.

Steps: predict the test days with the run's checkpoint into ``<out>/final_test/`` (never into the
publish folder), score them the way ``evaluate`` scores validation days into
``metrics/<run_id>_test.csv`` (same columns, ``split = test``), then add ``metrics_test`` to
``manifest.json`` and set ``test_evaluated`` to true.

If a run stops after the CSV is written and before the manifest is updated, the CSV stays and the
next call refuses. Check why it stopped before removing the CSV.

Usage (from ml/, after the validation evaluation wrote outputs/manifest.json):

    python -m oceanscope_ml.evaluation.final_test --config configs/experiment.yaml \
        --store <parts...> --stats <stats.zarr> --checkpoint <checkpoints/<run_id>/best.pt> \
        --owner-go
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
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.evaluation.evaluate import (
    BLOCK_DAYS,
    SEASON_ORDER,
    SEASONS,
    _Acc,
    _validate,
    region_masks,
)
from oceanscope_ml.inference.predict import predict_store

logger = logging.getLogger("oceanscope_ml.final_test")

METRIC_COLS = ["model", "split", "depth_m", "n", "mae_c", "rmse_c", "bias_c", "skill_vs_clim_pct"]


def check_may_run(out: Path, go: bool) -> dict[str, Any]:
    """Return the manifest, or raise unless the owner said go and no test result exists yet."""
    if not go:
        raise PermissionError(
            "The final test scores the sealed test year once. Run it only when Aryan says "
            "run the final test (pass --owner-go)."
        )
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {out}; run the validation evaluation first")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["synthetic"]:
        raise ValueError("This manifest is marked synthetic; the final test runs on real runs only")
    if manifest["test_evaluated"]:
        raise FileExistsError("test_evaluated is already true: the final test has run for this run")
    test_csv = out / "metrics" / f"{manifest['active_run_id']}_test.csv"
    if test_csv.exists():
        raise FileExistsError(f"{test_csv} exists: the final test has already been scored")
    return manifest


def score_test_predictions(
    config: ExperimentConfig,
    predictions: Path,
    store_path: Path | list[Path],
    stats_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write ``metrics/<run_id>_test.csv`` for a test-period prediction store; return its path."""
    out = Path(out_dir or config.paths.outputs)
    pred = xr.open_zarr(predictions)
    run_id = str(pred.attrs["run_id"])
    model_name = str(pred.attrs.get("model_name", "unet"))
    dates = pd.to_datetime(pred["time"].values)
    test_start, test_end = (pd.Timestamp(d) for d in config.splits.test)
    if len(dates) == 0 or dates.min() < test_start or dates.max() > test_end:
        raise ValueError("The final test store must hold test-period dates only")

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

    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{run_id}_test.csv"
    rows = total.rows(depths, {"model": model_name, "split": "test"})
    pd.DataFrame(rows)[METRIC_COLS].to_csv(path, index=False)
    overall = pd.read_csv(path).iloc[-1]
    logger.info(
        "Final test, run %s on %d test days: RMSE %.3f C, MAE %.3f C, skill vs climatology %.1f%%",
        run_id,
        len(dates),
        overall["rmse_c"],
        overall["mae_c"],
        overall["skill_vs_clim_pct"],
    )
    return path


def run_final_test(
    config: ExperimentConfig,
    store_path: Path | list[Path],
    checkpoint: Path,
    *,
    go: bool,
    out_dir: Path | None = None,
    stats_path: Path | None = None,
    device_choice: str = "auto",
) -> Path:
    """Score the test year once for the run in ``outputs/manifest.json``; return the CSV path."""
    out = Path(out_dir or config.paths.outputs)
    manifest = check_may_run(out, go)
    run_id = str(manifest["active_run_id"])
    if checkpoint.parent.name != run_id:
        raise ValueError(f"Checkpoint {checkpoint} is not from run {run_id}")

    predictions = predict_store(
        config,
        store_path,
        checkpoint,
        out_dir=out / "final_test",
        stats_path=stats_path,
        device_choice=device_choice,
        run_id=run_id,
        split="test",
        allow_test=True,
    )
    csv = score_test_predictions(config, predictions, store_path, stats_path, out_dir=out)

    manifest["files"]["metrics_test"] = csv.relative_to(out).as_posix()
    manifest["test_evaluated"] = True
    _validate(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Manifest updated: test_evaluated = true (%s)", csv)
    return csv


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the sealed test year once.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--store", nargs="+", required=True, help="Store path(s), as for train")
    parser.add_argument("--stats", default=None, help="Stats store with clim and norm_stats")
    parser.add_argument("--checkpoint", required=True, help="checkpoints/<run_id>/best.pt")
    parser.add_argument("--out-dir", default=None, help="Default paths.outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--owner-go",
        action="store_true",
        help="Set only when Aryan has said run the final test. Scores the test year once.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    stores = [Path(s) for s in args.store]
    run_final_test(
        load_config(args.config),
        stores if len(stores) > 1 else stores[0],
        Path(args.checkpoint),
        go=args.owner_go,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        stats_path=Path(args.stats) if args.stats else None,
        device_choice=args.device,
    )


if __name__ == "__main__":
    main()
