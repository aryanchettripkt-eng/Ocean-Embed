"""The live kit: what a live run needs from the trained run, and the model step (PRD §12).

A kit is a folder:

    kit.json          run_id (the published name), the members, dataset version, created time
    best.pt           a single-run kit: the published checkpoint, unchanged (no online learning)
    members/<name>.pt an ensemble kit: one checkpoint per member, unchanged
    stats.zarr        train-only climatology `clim` and `norm_stats` (the stats store)
    static.nc         ocean_mask, bathymetry, woa_clim_profile from a part store
    argo_eval_floats.json  the held-out float list, when a member has ARGO input channels

Each member in ``kit.json`` has its name, file, the run name stored in the checkpoint and its
channel names. An ensemble's prediction is the plain mean of its members, like
``inference.ensemble``: members may differ in their optional channels (ARGO inputs, ``year``)
but share the stats store. Older kits (``best.pt`` and a top-level ``channel_names``) still open
as one-member kits.

It lives in its own Hugging Face repo (``live.kit_repo``), not in the outputs dataset repo,
because the backend downloads the whole outputs repo.

Usage (from ml/, once per published run, after evaluation):

    # one run
    python -m oceanscope_ml.live kit --checkpoint checkpoints/<run_id>/best.pt \
        --stats <stats.zarr> --static-store <any part.zarr> --out kit/ [--run-id <name>] \
        [--upload]

    # an ensemble (the published name is required)
    python -m oceanscope_ml.live kit --member main2_ai=<best.pt> --member main2_ft=<best.pt> \
        --member main3=<best.pt> --run-id ens3 --stats <stats.zarr> \
        --static-store <any part.zarr> --out kit/ [--upload]

``--run-id`` is the name the run was published under (``active_run_id`` in the manifest),
when it differs from the name stored in the checkpoint. A fine-tuned run is the usual case:
``best_ft.pt`` of ``main2_ft_l0.3`` is published as ``main2_ft``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.argo_inputs import ArgoInputs, enable_argo_input, enable_year_input
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.inference.predict import COVERAGE_VALID, coverage_for_day, load_checkpoint
from oceanscope_ml.train import _crop_to_target

logger = logging.getLogger("oceanscope_ml.live.kit")

STATIC_VARS = ("ocean_mask", "bathymetry", "woa_clim_profile")
HOLDOUT_FILE = "argo_eval_floats.json"


def uses_argo(names: list[str]) -> bool:
    return any(str(n).startswith("argo_") for n in names)


def member_config(config: ExperimentConfig, names: list[str]) -> ExperimentConfig:
    """A copy of ``config`` with the optional channels a member was trained with turned on.

    Raises if ``names`` is not the config's channel list with those options.
    """
    cfg = copy.deepcopy(config)
    if uses_argo(names):
        enable_argo_input(cfg)
    if "year" in names:
        enable_year_input(cfg)
    if channel_names(cfg) != list(names):
        raise ValueError("Checkpoint channel list does not match the config")
    return cfg


def argo_window_days(config: ExperimentConfig) -> int:
    return int((config.inputs.get("argo_input") or {}).get("window_days", 10))


@dataclass
class KitMember:
    name: str
    file: str  # relative to the kit root
    checkpoint_run_id: str
    channel_names: list[str] = field(default_factory=list)

    @property
    def uses_argo(self) -> bool:
        return uses_argo(self.channel_names)


@dataclass
class Kit:
    root: Path
    run_id: str  # the published name (manifest active_run_id)
    checkpoint_run_id: str = ""  # one-member kits: the name inside best.pt; defaults to run_id
    members: list[KitMember] = field(default_factory=list)
    holdout_file: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_run_id:
            self.checkpoint_run_id = self.run_id
        if not self.members:
            self.members = [KitMember(self.run_id, "best.pt", self.checkpoint_run_id)]

    @property
    def checkpoint(self) -> Path:
        return self.root / self.members[0].file

    @property
    def stats(self) -> Path:
        return self.root / "stats.zarr"

    @property
    def static_path(self) -> Path:
        return self.root / "static.nc"

    @property
    def needs_argo(self) -> bool:
        return any(m.uses_argo for m in self.members)

    def static(self) -> xr.Dataset:
        return xr.open_dataset(self.static_path).load()

    def holdout(self) -> set[str]:
        """The held-out ARGO floats (never inputs); empty when no member uses ARGO."""
        if not self.holdout_file:
            return set()
        raw = json.loads((self.root / self.holdout_file).read_text(encoding="utf-8"))
        return {str(f) for f in raw.get("held_out_floats") or []}


def _load_holdout(path: Path) -> set[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    floats = {str(f) for f in raw.get("held_out_floats") or []}
    if not floats:
        raise ValueError(f"{path} has no held_out_floats")
    return floats


def build_kit(
    config: ExperimentConfig,
    checkpoint: Path | dict[str, Path],
    stats: Path,
    static_store: Path,
    out: Path,
    run_id: str | None = None,
    holdout: Path | None = None,
) -> Kit:
    """Assemble a kit folder from a trained run's files.

    ``checkpoint`` is one ``best.pt``, or ``{member name: best.pt}`` for an ensemble.
    ``run_id`` is the published name; default for one checkpoint: the name stored in it.
    ``holdout`` (``argo_eval_floats.json``) is needed when a member has ARGO input channels.
    """
    single = not isinstance(checkpoint, dict)
    if isinstance(checkpoint, dict):
        paths = {k: Path(v) for k, v in checkpoint.items()}
    else:
        paths = {"": Path(checkpoint)}
    if not single and len(paths) < 2:
        raise ValueError("An ensemble kit needs at least two members")
    if not single and not run_id:
        raise ValueError("An ensemble kit needs the published name (run_id)")
    members: list[KitMember] = []
    for name, path in paths.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        names = [str(n) for n in ckpt["channel_names"]]
        window = ckpt.get("argo_window_days")
        if uses_argo(names) and window and int(window) != argo_window_days(config):
            raise ValueError(
                f"{path} uses a {window}-day ARGO window; live mode fetches "
                f"{argo_window_days(config)} days. Widen the live ARGO fetch before using it."
            )
        member_config(config, names)  # raises on a channel list the config cannot make
        ckpt_run_id = str(ckpt.get("run_id", path.parent.name))
        file = "best.pt" if single else f"members/{name}.pt"
        members.append(KitMember(name or ckpt_run_id, file, ckpt_run_id, names))
    run_id = run_id or members[0].checkpoint_run_id
    needs_argo = any(m.uses_argo for m in members)
    if needs_argo and holdout is None:
        raise ValueError("A member has ARGO input channels: pass the held-out float list")
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
    for stale in ("best.pt", "members", HOLDOUT_FILE):
        target = out / stale
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for member, path in zip(members, paths.values()):
        (out / member.file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / member.file)
    if (out / "stats.zarr").exists():
        shutil.rmtree(out / "stats.zarr")
    shutil.copytree(stats, out / "stats.zarr")
    static.to_netcdf(out / "static.nc")
    holdout_file = None
    if needs_argo:
        assert holdout is not None
        _load_holdout(holdout)
        shutil.copy2(holdout, out / HOLDOUT_FILE)
        holdout_file = HOLDOUT_FILE
    meta: dict[str, Any] = {
        "run_id": run_id,
        "dataset_version": str(config.dataset_version),
        "members": [m.__dict__ for m in members],
        "argo_holdout": holdout_file,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if single:
        meta["checkpoint_run_id"] = members[0].checkpoint_run_id
        meta["channel_names"] = members[0].channel_names
    (out / "kit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Kit for run %s (%s) written to %s", run_id, ", ".join(m.name for m in members), out
    )
    return Kit(
        out,
        run_id,
        members[0].checkpoint_run_id if single else "",
        members,
        holdout_file,
    )


def open_kit(root: Path, config: ExperimentConfig) -> Kit:
    """Open a kit folder and check it matches the config."""
    meta = json.loads((root / "kit.json").read_text(encoding="utf-8"))
    if str(meta["dataset_version"]) != str(config.dataset_version):
        raise ValueError(
            f"Kit dataset_version {meta['dataset_version']} != config {config.dataset_version}"
        )
    run_id = str(meta["run_id"])
    if meta.get("members"):
        members = [
            KitMember(
                str(m["name"]),
                str(m["file"]),
                str(m["checkpoint_run_id"]),
                [str(n) for n in m["channel_names"]],
            )
            for m in meta["members"]
        ]
    else:  # a kit from before ensembles
        ckpt_run_id = str(meta.get("checkpoint_run_id") or run_id)
        members = [KitMember(run_id, "best.pt", ckpt_run_id, list(meta["channel_names"]))]
    for m in members:
        try:
            member_config(config, m.channel_names)
        except ValueError:
            raise ValueError(
                f"Kit {root} member {m.name} was built for a different channel list than the config"
            ) from None
    needed = [m.file for m in members] + ["stats.zarr", "static.nc"]
    holdout_file = meta.get("argo_holdout")
    if any(m.uses_argo for m in members):
        if not holdout_file:
            raise ValueError(f"Kit {root} has ARGO members but no held-out float list")
        needed.append(str(holdout_file))
    for name in needed:
        if not (root / name).exists():
            raise FileNotFoundError(f"Kit {root} lacks {name}")
    single_id = members[0].checkpoint_run_id if len(members) == 1 else ""
    return Kit(root, run_id, single_id, members, str(holdout_file) if holdout_file else None)


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
        delete_patterns=["*"],  # files of an older kit layout go; uploaded ones stay
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
    argo_profiles: pd.DataFrame | None = None,
) -> xr.Dataset:
    """Model output for ``days`` from a window store: temp_pred (°C) and coverage.

    Same steps as ``inference.predict``: anomaly × sigma_d + climatology, then NaN wherever
    coverage is not "valid prediction" (PRD §6.2). An ensemble kit averages its members, like
    ``inference.ensemble``. Members with ARGO input channels need ``argo_profiles`` (rows in
    the ``data.argo.profile_levels`` layout, held-out floats already removed) covering the
    ARGO window before each day; every profile is an input ("predict" mode).
    """
    device = torch.device("cpu")
    ds = open_parts([window_store], kit.stats)
    ocean = np.asarray(ds["ocean_mask"].values, dtype=bool)
    core = [str(v) for v in config.inputs.get("core", [])]
    holdout = kit.holdout()
    anomaly_sum: np.ndarray | None = None
    sigma: np.ndarray | None = None
    positions: np.ndarray | None = None
    for member in kit.members:
        cfg = copy.deepcopy(config)
        model, ckpt_run_id = load_checkpoint(cfg, kit.root / member.file, device)
        if ckpt_run_id != member.checkpoint_run_id:
            raise ValueError(
                f"Kit {member.file} holds run {ckpt_run_id}, kit.json says "
                f"{member.checkpoint_run_id}"
            )
        dataset = OceanScopeDataset(
            window_store, cfg, split="live", stats_path=kit.stats, live_dates=list(days)
        )
        if dataset._has_argo_channels:
            if argo_profiles is None:
                raise ValueError(f"Member {member.name} has ARGO input channels: no profiles given")
            argo_cfg = cfg.inputs.get("argo_input") or {}
            argo = ArgoInputs(
                argo_profiles,
                ds,
                dataset.sigma_d,
                holdout,
                window_days=argo_window_days(cfg),
                train_drop_frac=float(argo_cfg.get("train_drop_frac", 0.3)),
                seed=int(cfg.train.get("seed", 42)),
            )
            dataset.set_argo(argo, "predict")
        with torch.no_grad():
            anomalies = []
            for i in range(len(days)):
                sample = dataset[i]
                out = model(sample["x"][None].to(device)).float()
                anomalies.append(_crop_to_target(out, sample["y"][None])[0].cpu().numpy())
        stacked = np.stack(anomalies).astype(np.float64)
        anomaly_sum = stacked if anomaly_sum is None else anomaly_sum + stacked
        sigma = dataset.sigma_d.astype(np.float32)[:, None, None]
        positions = np.asarray(dataset._positions)
        del model
    assert anomaly_sum is not None and sigma is not None and positions is not None
    anomaly_mean = (anomaly_sum / len(kit.members)).astype(np.float32)
    preds, covs = [], []
    for i, day in enumerate(days):
        clim = np.asarray(ds["clim"].isel(dayofyear=day.day_of_year - 1).values, np.float32)
        temp = anomaly_mean[i] * sigma + clim
        cov = coverage_for_day(ds, int(positions[i]), ocean, core)
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
