"""Average several prediction stores into one ensemble store (validation dates only).

Usage (from ml/):

    python -m oceanscope_ml.inference.ensemble --config configs/experiment.yaml \
        --predictions <predictions_a.zarr> <predictions_b.zarr> ... --run-id <name> \
        [--out-dir outputs] [--weights w1,w2,...]

All members must have the same time, depth, lat and lon coordinates. ``temp_pred`` is the plain
mean of the members, ``coverage`` is 1 only where every member is valid, and ``temp_glorys`` is
copied from the first member. With ``--weights`` (one per member, >= 0, summing to 1; see
``inference.ensemble_weights``) ``temp_pred`` is the weighted mean instead. The layout matches
``inference.predict``, so the ARGO check, the
ARGO correction and ``evaluation.evaluate`` read the result like any run. Test-period dates are
refused (the 2021 test year stays sealed).

Writes ``<out>/predictions/predictions_<run_id>.zarr``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.transform import check_no_test_leakage

logger = logging.getLogger("oceanscope_ml.inference.ensemble")

BLOCK_DAYS = 16
COORDS = ("time", "depth", "lat", "lon")


def _check_members(
    config: ExperimentConfig, members: list[xr.Dataset], allow_test: bool = False
) -> None:
    first = members[0]
    for k, ds in enumerate(members):
        dates = pd.DatetimeIndex(ds["time"].values).normalize()
        if allow_test:  # evaluation.argo_test only: test dates and nothing else
            start, end = (pd.Timestamp(d) for d in config.splits.test)
            if dates.size == 0 or dates.min() < start or dates.max() > end:
                raise ValueError(f"Member {k} must hold test dates only")
        else:
            check_no_test_leakage(dates, config.splits.test)
            val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
            if dates.size == 0 or dates.min() < val_start or dates.max() > val_end:
                raise ValueError(f"Member {k} must hold validation dates only")
        for c in COORDS:
            if not np.array_equal(np.asarray(ds[c].values), np.asarray(first[c].values)):
                raise ValueError(f"Member {k} has different {c} coordinates from member 0")


def ensemble(
    config: ExperimentConfig,
    predictions: list[Path],
    run_id: str,
    out_dir: Path | None = None,
    weights: list[float] | None = None,
    allow_test: bool = False,
) -> Path:
    """Write the (weighted) mean of ``predictions`` as ``predictions_<run_id>.zarr``.
    ``allow_test`` (test-year members) is set only by ``evaluation.argo_test``."""
    if len(predictions) < 2:
        raise ValueError("An ensemble needs at least two prediction stores")
    w = np.full(len(predictions), 1.0 / len(predictions))
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.size != len(predictions) or (w < 0).any() or not np.isclose(w.sum(), 1.0):
            raise ValueError("Weights must be one per member, >= 0 and sum to 1")
    members = [xr.open_zarr(p) for p in predictions]
    _check_members(config, members, allow_test)
    ids = [str(ds.attrs.get("run_id", Path(p).stem)) for ds, p in zip(members, predictions)]
    logger.info("Ensemble %s of %s", run_id, ", ".join(ids))

    out = Path(out_dir or config.paths.outputs)
    dest = out / "predictions" / f"predictions_{run_id}.zarr"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(predictions[0], dest)

    dtype = members[0]["temp_pred"].dtype
    cov_dtype = members[0]["coverage"].dtype
    n_time = int(members[0].sizes["time"])
    for b0 in range(0, n_time, BLOCK_DAYS):
        b1 = min(b0 + BLOCK_DAYS, n_time)
        total = None
        valid = None
        for wk, ds in zip(w, members, strict=True):
            block = ds.isel(time=slice(b0, b1))
            temp = wk * block["temp_pred"].transpose(*COORDS).values.astype(float)
            ok = block["coverage"].transpose("time", "lat", "lon").values == 1
            total = temp if total is None else total + temp
            valid = ok if valid is None else valid & ok
        assert total is not None and valid is not None
        mean = total
        mean = np.where(valid[:, None], mean, np.nan)
        block_out = xr.Dataset(
            {
                "temp_pred": (COORDS, mean.astype(dtype)),
                "coverage": (("time", "lat", "lon"), valid.astype(cov_dtype)),
            }
        )
        block_out.to_zarr(dest, region={"time": slice(b0, b1)}, mode="r+")
        logger.info("Averaged %d / %d days", b1, n_time)

    group = zarr.open_group(str(dest), mode="r+")
    group.attrs.update(
        {
            "run_id": run_id,
            "ensemble_members": json.dumps(ids),
            "ensemble_weights": json.dumps([round(float(x), 6) for x in w]),
        }
    )
    group.attrs.pop("base_run_id", None)
    group.attrs.pop("argo_correction", None)
    zarr.consolidate_metadata(str(dest))
    return dest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Average prediction stores into an ensemble.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    parser.add_argument("--weights", default=None, help="Comma-separated, one per member")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    dest = ensemble(
        load_config(args.config),
        [Path(p) for p in args.predictions],
        args.run_id,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        weights=[float(x) for x in args.weights.split(",")] if args.weights else None,
    )
    print(json.dumps({"store": str(dest)}))


if __name__ == "__main__":
    main()
