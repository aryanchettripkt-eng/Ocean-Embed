"""Write the prediction store for the validation dates (PRD_AI_ML §4 stage 7, §6.2).

Usage (from ml/):

    python -m oceanscope_ml.inference.predict --config configs/experiment.yaml \
        --store <parts...> --stats <stats.zarr> --checkpoint checkpoints/<run_id>/best.pt \
        [--cache-dir /tmp/cache] [--out-dir outputs]

Writes ``<out>/predictions/predictions_<run_id>.zarr`` with ``temp_pred`` (°C, climatology
added back), ``temp_glorys`` and ``coverage`` on the dates of ``inference.publish_splits``.

Only the validation split is written here. Test dates are sealed (PRD §0 rule 3): they are read
only by ``oceanscope_ml.evaluation.final_test``, run once on the owner's request.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader, Subset

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.train import _crop_to_target, build_model, resolve_device

logger = logging.getLogger("oceanscope_ml.inference.predict")

# PRD §6.2: a cell is "insufficient inputs" if any of these is invalid on day t, or fewer
# than MIN_CORE_VALID of the core inputs are valid there.
REQUIRED_INPUTS = ("sla", "sst", "u_wind", "v_wind")
MIN_CORE_VALID = 7
COVERAGE_LAND, COVERAGE_VALID, COVERAGE_INSUFFICIENT = 0, 1, 2
WRITE_BLOCK_DAYS = 32


def load_checkpoint(
    config: ExperimentConfig, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, str]:
    """Model from a ``best.pt``/``last.pt`` checkpoint, after checking its channel order."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    channels = channel_names(config)
    if ckpt["channel_names"] != channels:
        raise ValueError(
            f"Checkpoint {checkpoint} was trained with a different channel list than the config"
        )
    model = build_model(config, len(channels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, str(ckpt.get("run_id", checkpoint.parent.name))


def coverage_for_day(ds: xr.Dataset, t: int, ocean: np.ndarray, core: list[str]) -> np.ndarray:
    """int8 [lat, lon]: 0 land, 1 valid prediction, 2 insufficient inputs (PRD §6.2)."""
    valid = {v: np.isfinite(np.asarray(ds[v].isel(time=t).values)) for v in core if v in ds}
    missing = [v for v in REQUIRED_INPUTS if v not in valid]
    if missing:
        raise KeyError(f"Store lacks required inputs {missing}")
    enough = np.sum(np.stack(list(valid.values())), axis=0) >= MIN_CORE_VALID
    for v in REQUIRED_INPUTS:
        enough &= valid[v]
    cov = np.full(ocean.shape, COVERAGE_LAND, dtype=np.int8)
    cov[ocean & enough] = COVERAGE_VALID
    cov[ocean & ~enough] = COVERAGE_INSUFFICIENT
    return cov


def split_runs(dates: pd.DatetimeIndex, label: str) -> dict[str, str]:
    """``{"start/end": label}`` for each unbroken run of days (``split_of_each_date``)."""
    runs: dict[str, str] = {}
    if len(dates) == 0:
        return runs
    breaks = np.flatnonzero(np.diff(dates.values) != np.timedelta64(1, "D"))
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(dates) - 1]])
    for s, e in zip(starts, ends):
        runs[f"{dates[s].date()}/{dates[e].date()}"] = label
    return runs


def _publish_splits(config: ExperimentConfig) -> list[str]:
    splits = [str(s) for s in config.inference.get("publish_splits", ["val"])]
    if "test" in splits:
        logger.info("Test dates are sealed; they are written only by final_test (PRD §0 rule 3).")
    chosen = [s for s in splits if s == "val"]
    if not chosen:
        raise ValueError(f"inference.publish_splits {splits} has no split predict may write")
    return chosen


def predict_store(
    config: ExperimentConfig,
    store_path: Path | list[Path],
    checkpoint: Path,
    out_dir: Path | None = None,
    stats_path: Path | None = None,
    cache_dir: Path | None = None,
    device_choice: str = "auto",
    run_id: str | None = None,
) -> Path:
    """Run the model on every validation day and write the prediction store."""
    device = resolve_device(device_choice)
    model, ckpt_run_id = load_checkpoint(config, checkpoint, device)
    run_id = run_id or ckpt_run_id
    split = _publish_splits(config)[0]

    dataset = OceanScopeDataset(
        store_path, config, split=split, stats_path=stats_path, cache_dir=cache_dir
    )
    ds = open_parts(store_path if isinstance(store_path, list) else [store_path], stats_path)
    times = pd.to_datetime(ds["time"].values)
    positions = np.asarray(dataset._positions)
    dates = pd.DatetimeIndex(times[positions])
    ocean = np.asarray(ds["ocean_mask"].values, dtype=bool)
    sigma = dataset.sigma_d.astype(np.float32)[:, None, None]
    core = [str(v) for v in config.inputs.get("core", [])]
    depths = [float(d) for d in config.depths_m]
    n_d, n_lat, n_lon = len(depths), int(ds.sizes["lat"]), int(ds.sizes["lon"])
    dtype = str(config.inference.get("publish_dtype", "float16"))

    out_base = Path(out_dir or config.paths.outputs) / "predictions"
    out_base.mkdir(parents=True, exist_ok=True)
    dest = out_base / f"predictions_{run_id}.zarr"
    if dest.exists():
        shutil.rmtree(dest)

    attrs: dict[str, Any] = {
        "run_id": run_id,
        "dataset_version": str(config.dataset_version),
        "model_name": str(config.train.get("model", "unet")),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_of_each_date": json.dumps(split_runs(dates, split)),
        "depths_m": depths,
    }
    coords = {
        "time": dates.values,
        "depth": np.array(depths, dtype=np.float64),
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
    }
    template = xr.Dataset(
        {
            "temp_pred": (
                ("time", "depth", "lat", "lon"),
                _lazy_empty((len(dates), n_d, n_lat, n_lon), dtype),
                {"units": "degC"},
            ),
            "temp_glorys": (
                ("time", "depth", "lat", "lon"),
                _lazy_empty((len(dates), n_d, n_lat, n_lon), dtype),
                {"units": "degC"},
            ),
            "coverage": (
                ("time", "lat", "lon"),
                _lazy_empty((len(dates), n_lat, n_lon), "int8"),
                {
                    "flag_values": [0, 1, 2],
                    "flag_meanings": "land_unsupported valid_prediction insufficient_inputs",
                },
            ),
        },
        coords=coords,
        attrs=attrs,
    )
    template["depth"].attrs.update({"units": "m", "positive": "down"})
    template["lat"].attrs["units"] = "degrees_north"
    template["lon"].attrs["units"] = "degrees_east"
    encoding = {
        "temp_pred": {"chunks": (1, n_d, n_lat, n_lon), "dtype": dtype},
        "temp_glorys": {"chunks": (1, n_d, n_lat, n_lon), "dtype": dtype},
        "coverage": {"chunks": (1, n_lat, n_lon), "dtype": "int8"},
    }
    template.to_zarr(dest, mode="w", compute=False, zarr_format=2, encoding=encoding)

    clim = np.asarray(ds["clim"].values, dtype=np.float32)  # (366, depth, lat, lon)
    batch_size = int(config.train.get("batch_size", 8))
    logger.info("Predicting %d %s days with %s into %s", len(dates), split, checkpoint, dest)
    with torch.no_grad():
        for b0 in range(0, len(dates), WRITE_BLOCK_DAYS):
            b1 = min(b0 + WRITE_BLOCK_DAYS, len(dates))
            loader = DataLoader(
                Subset(dataset, list(range(b0, b1))),
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(config.train.get("num_workers", 2)),
            )
            preds: list[np.ndarray] = []
            for batch in loader:
                x = batch["x"].to(device)
                pred = _crop_to_target(model(x).float(), batch["y"])
                preds.append(pred.cpu().numpy())
            anomaly = np.concatenate(preds)  # (days, depth, lat, lon), normalized
            doy = dates[b0:b1].day_of_year.values - 1
            temp_pred = anomaly * sigma[None] + clim[doy]
            cov = np.stack([coverage_for_day(ds, int(t), ocean, core) for t in positions[b0:b1]])
            temp_pred = np.where((cov == COVERAGE_VALID)[:, None], temp_pred, np.nan)
            glorys = np.asarray(ds["thetao"].isel(time=positions[b0:b1]).values, np.float32)
            block = xr.Dataset(
                {
                    "temp_pred": (("time", "depth", "lat", "lon"), temp_pred.astype(dtype)),
                    "temp_glorys": (("time", "depth", "lat", "lon"), glorys.astype(dtype)),
                    "coverage": (("time", "lat", "lon"), cov),
                },
            )
            block.to_zarr(dest, region={"time": slice(b0, b1)})
            logger.info("Predicted %d / %d days", b1, len(dates))
    zarr.consolidate_metadata(str(dest))
    logger.info("Prediction store ready: %s", dest)
    return dest


def _lazy_empty(shape: tuple[int, ...], dtype: str) -> Any:
    import dask.array as da

    fill = 0 if dtype == "int8" else np.nan
    return da.full(shape, fill, dtype=dtype, chunks=(1, *shape[1:]))


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the OceanScope prediction store.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--store", nargs="+", required=True, help="Processed store or part stores")
    parser.add_argument("--stats", default=None, help="Stats store of a multi-part build")
    parser.add_argument("--checkpoint", required=True, help="checkpoints/<run_id>/best.pt")
    parser.add_argument("--cache-dir", default=None, help="Training cache dir (faster reads)")
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    parser.add_argument("--run-id", default=None, help="Default: the checkpoint's run id")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    predict_store(
        config,
        [Path(p) for p in args.store],
        Path(args.checkpoint),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        stats_path=Path(args.stats) if args.stats else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        device_choice=args.device,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
