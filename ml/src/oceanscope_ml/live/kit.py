"""The live kit: what a live run needs from the trained run, and the model step (PRD §12).

A kit is a folder:

    kit.json          run_id (the published name), checkpoint_run_id, channel names,
                      dataset version, created time
    best.pt           the published checkpoint, unchanged (no online learning)
    stats.zarr        train-only climatology `clim` and `norm_stats` (the stats store)
    static.nc         ocean_mask, bathymetry, woa_clim_profile from a part store

It lives in its own Hugging Face repo (``live.kit_repo``), not in the outputs dataset repo,
because the backend downloads the whole outputs repo.

Usage (from ml/, once per published run, after evaluation):

    python -m oceanscope_ml.live kit --checkpoint checkpoints/<run_id>/best.pt \
        --stats <stats.zarr> --static-store <any part.zarr> --out kit/ [--run-id <name>] \
        [--upload]

``--run-id`` is the name the run was published under (``active_run_id`` in the manifest),
when it differs from the name stored in the checkpoint. A fine-tuned run is the usual case:
``best_ft.pt`` of ``main2_ft_l0.3`` is published as ``main2_ft``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.inference.predict import COVERAGE_VALID, coverage_for_day, load_checkpoint
from oceanscope_ml.train import _crop_to_target

logger = logging.getLogger("oceanscope_ml.live.kit")

STATIC_VARS = ("ocean_mask", "bathymetry", "woa_clim_profile")


@dataclass
class Kit:
    root: Path
    run_id: str  # the published name (manifest active_run_id)
    checkpoint_run_id: str = ""  # the name inside best.pt; defaults to run_id

    def __post_init__(self) -> None:
        if not self.checkpoint_run_id:
            self.checkpoint_run_id = self.run_id

    @property
    def checkpoint(self) -> Path:
        return self.root / "best.pt"

    @property
    def stats(self) -> Path:
        return self.root / "stats.zarr"

    @property
    def static_path(self) -> Path:
        return self.root / "static.nc"

    def static(self) -> xr.Dataset:
        return xr.open_dataset(self.static_path).load()


def build_kit(
    config: ExperimentConfig,
    checkpoint: Path,
    stats: Path,
    static_store: Path,
    out: Path,
    run_id: str | None = None,
) -> Kit:
    """Assemble a kit folder from a trained run's files.

    ``run_id`` is the published name; default: the name stored in the checkpoint.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt["channel_names"] != channel_names(config):
        raise ValueError("Checkpoint channel list does not match the config")
    ckpt_run_id = str(ckpt.get("run_id", checkpoint.parent.name))
    run_id = run_id or ckpt_run_id
    stats_ds = xr.open_zarr(stats)
    if "clim" not in stats_ds or "norm_stats" not in stats_ds.attrs:
        raise ValueError(f"{stats} is not a stats store (needs clim and norm_stats)")
    part = xr.open_zarr(static_store)
    missing = [v for v in STATIC_VARS if v not in part]
    if missing:
        raise ValueError(f"{static_store} lacks static fields {missing}")
    static = part[list(STATIC_VARS)].load()
    static.attrs = {"source_store": str(static_store)}

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, out / "best.pt")
    if (out / "stats.zarr").exists():
        shutil.rmtree(out / "stats.zarr")
    shutil.copytree(stats, out / "stats.zarr")
    static.to_netcdf(out / "static.nc")
    meta = {
        "run_id": run_id,
        "checkpoint_run_id": ckpt_run_id,
        "dataset_version": str(config.dataset_version),
        "channel_names": channel_names(config),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "kit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Kit for run %s written to %s", run_id, out)
    return Kit(out, run_id, ckpt_run_id)


def open_kit(root: Path, config: ExperimentConfig) -> Kit:
    """Open a kit folder and check it matches the config."""
    meta = json.loads((root / "kit.json").read_text(encoding="utf-8"))
    if meta["channel_names"] != channel_names(config):
        raise ValueError(f"Kit {root} was built for a different channel list than the config")
    if str(meta["dataset_version"]) != str(config.dataset_version):
        raise ValueError(
            f"Kit dataset_version {meta['dataset_version']} != config {config.dataset_version}"
        )
    for name in ("best.pt", "stats.zarr", "static.nc"):
        if not (root / name).exists():
            raise FileNotFoundError(f"Kit {root} lacks {name}")
    return Kit(root, str(meta["run_id"]), str(meta.get("checkpoint_run_id") or meta["run_id"]))


def upload_kit(kit: Kit, repo_id: str, api: Any = None) -> None:
    """Upload a kit folder to a private Hugging Face model repo (needs HF_TOKEN)."""
    if api is None:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is not set")
        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(kit.root),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Live kit for run {kit.run_id}",
    )


def download_kit(repo_id: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN", "").strip() or None
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(dest), token=token)
    return dest


def predict_days(
    config: ExperimentConfig,
    kit: Kit,
    window_store: Path,
    days: list[pd.Timestamp],
) -> xr.Dataset:
    """Model output for ``days`` from a window store: temp_pred (°C) and coverage.

    Same steps as ``inference.predict``: anomaly × sigma_d + climatology, then NaN wherever
    coverage is not "valid prediction" (PRD §6.2).
    """
    device = torch.device("cpu")
    model, ckpt_run_id = load_checkpoint(config, kit.checkpoint, device)
    if ckpt_run_id != kit.checkpoint_run_id:
        raise ValueError(
            f"Kit best.pt holds run {ckpt_run_id}, kit.json says {kit.checkpoint_run_id}"
        )
    dataset = OceanScopeDataset(
        window_store, config, split="live", stats_path=kit.stats, live_dates=list(days)
    )
    ds = open_parts([window_store], kit.stats)
    ocean = np.asarray(ds["ocean_mask"].values, dtype=bool)
    sigma = dataset.sigma_d.astype(np.float32)[:, None, None]
    core = [str(v) for v in config.inputs.get("core", [])]
    preds, covs = [], []
    with torch.no_grad():
        for i, day in enumerate(days):
            sample = dataset[i]
            out = model(sample["x"][None].to(device)).float()
            anomaly = _crop_to_target(out, sample["y"][None])[0].cpu().numpy()
            clim = np.asarray(ds["clim"].isel(dayofyear=day.day_of_year - 1).values, np.float32)
            temp = anomaly * sigma + clim
            cov = coverage_for_day(ds, int(dataset._positions[i]), ocean, core)
            preds.append(np.where((cov == COVERAGE_VALID)[None], temp, np.nan).astype(np.float32))
            covs.append(cov)
    return xr.Dataset(
        {
            "temp_pred": (("time", "depth", "lat", "lon"), np.stack(preds)),
            "coverage": (("time", "lat", "lon"), np.stack(covs)),
        },
        coords={
            "time": pd.DatetimeIndex(days),
            "depth": np.asarray(config.depths_m, dtype=np.float64),
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
        },
    )
