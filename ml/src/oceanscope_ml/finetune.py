"""ARGO fine-tuning of a trained U-Net (PRD_AI_ML §5.9).

Usage (from ml/):

    python -m oceanscope_ml.finetune --config configs/experiment.yaml --store <parts...> \
        --stats <stats.zarr> --base checkpoints/main2/best.pt --argo <argo_YYYY-MM.nc...> \
        --holdout configs/argo_eval_floats.json --lambda 0.1 --run-id main2_ft [--cache-dir DIR]

The loss is ``L + lambda * L_argo``. ``L`` is the training loss of ``train.py`` (GLORYS, mixed
with GREP). ``L_argo`` is the same depth-weighted masked MSE between the predicted anomaly and the
observed ARGO anomaly ((T_argo - clim) / sigma_d), at the cells and days of training-period
profiles. Profiles are interpolated to the 15 depths with no extrapolation and placed in their
nearest cell; several profiles in one cell and day are averaged.

Floats listed in the hold-out file are never read as targets, in any year (PRD §7). The ARGO
score used to pick the best epoch (and so lambda) comes from validation-period profiles of
non-held-out floats only. The test period is never read.

Only training days with at least one profile are used, so an epoch is shorter than in ``train.py``.
Each epoch writes ``<checkpoints>/<run_id>/last_ft.pt`` (a rerun resumes from it) and the best epoch
by validation ARGO RMSE goes to ``best_ft.pt``, which ``inference.predict`` reads like ``best.pt``.
Epoch 0 is the base model, so ``best_ft.pt`` is the base model if no epoch improves on it.
The per-epoch history goes to ``<outputs>/metrics/finetune_<run_id>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Subset

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.argo import profile_levels, read_levels
from oceanscope_ml.data.argo_inputs import (
    ArgoInputs,
    argo_inputs_for,
    enable_argo_input,
    expand_input_weights,
    set_argo_window,
)
from oceanscope_ml.data.argo_inputs import argo_window as argo_window_of
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.preprocess import compute_config_hash
from oceanscope_ml.evaluation.argo_validation import _nearest_cell
from oceanscope_ml.evaluation.evaluate import BAY_OF_BENGAL_MIN_LON, EQUATORIAL_MAX_LAT
from oceanscope_ml.models.loss import depth_weight_vector, masked_weighted_mse, total_loss
from oceanscope_ml.train import (
    _crop_to_target,
    _restore_rng,
    _rng_state,
    _save_atomic,
    _to_device,
    build_model,
    resolve_device,
    set_seed,
    validate,
)

logger = logging.getLogger("oceanscope_ml.finetune")

HISTORY_COLUMNS = [
    "epoch",
    "lambda",
    "train_loss",
    "val_rmse_c",
    "val_argo_rmse_c",
    "val_argo_bob_band_rmse_c",
    "val_argo_n_obs",
]
BAND_M = (50.0, 300.0)

# Per day: cell indices [n] and anomaly profiles [n, D] (NaN where a depth has no value).
DayTargets = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class FineTuneState:
    epoch: int = 0
    best_epoch: int = 0
    best_val_argo_rmse_c: float = float("inf")
    epochs_since_best: int = 0
    finished: bool = False
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class FineTuneResult:
    run_id: str
    lam: float
    best_epoch: int
    best_val_argo_rmse_c: float
    base_val_argo_rmse_c: float
    best_path: Path


def load_holdout(path: Path) -> set[str]:
    floats = json.loads(Path(path).read_text(encoding="utf-8")).get("held_out_floats") or []
    if not floats:
        raise ValueError(f"{path} lists no held-out floats (PRD_AI_ML §7)")
    return {str(f).strip() for f in floats}


def build_targets(
    profiles: pd.DataFrame, ds: xr.Dataset, sigma_d: np.ndarray
) -> dict[pd.Timestamp, DayTargets]:
    """Normalized ARGO anomaly profiles per day, on the store grid (ocean cells only)."""
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    n_depth = sigma_d.size
    if profiles.empty:
        return {}
    i_lat, lat_ok = _nearest_cell(lat, profiles["lat"].to_numpy(float))
    i_lon, lon_ok = _nearest_cell(lon, profiles["lon"].to_numpy(float))
    ocean = np.asarray(ds["ocean_mask"].values).astype(bool)
    keep = lat_ok & lon_ok
    keep[keep] &= ocean[i_lat[keep], i_lon[keep]]
    prof = profiles.loc[keep].reset_index(drop=True)
    i_lat, i_lon = i_lat[keep], i_lon[keep]
    if prof.empty:
        return {}
    doy = pd.DatetimeIndex(prof["day"]).day_of_year.to_numpy() - 1
    clim = (
        ds["clim"]
        .isel(
            dayofyear=xr.DataArray(doy, dims="p"),
            lat=xr.DataArray(i_lat, dims="p"),
            lon=xr.DataArray(i_lon, dims="p"),
        )
        .transpose("p", "depth")
        .values
    )
    temps = prof[[f"t_{d}" for d in range(n_depth)]].to_numpy(float)
    anomaly = (temps - np.asarray(clim, dtype=float)) / sigma_d[None, :]
    frame = pd.DataFrame(anomaly, columns=range(n_depth))
    frame["day"] = pd.DatetimeIndex(prof["day"])
    frame["i_lat"] = i_lat
    frame["i_lon"] = i_lon
    mean = frame.groupby(["day", "i_lat", "i_lon"]).mean()  # NaN-aware: skips missing depths
    out: dict[pd.Timestamp, DayTargets] = {}
    for day, group in mean.groupby(level="day"):
        idx = group.index
        out[pd.Timestamp(day)] = (
            idx.get_level_values("i_lat").to_numpy(int),
            idx.get_level_values("i_lon").to_numpy(int),
            group[list(range(n_depth))].to_numpy(np.float32),
        )
    return out


class ArgoTargetDataset(Dataset[dict[str, torch.Tensor]]):
    """Days of a base dataset that have ARGO targets, with ``argo_y`` and ``argo_mask`` added."""

    def __init__(
        self,
        base: OceanScopeDataset,
        dates: pd.DatetimeIndex,
        targets: dict[pd.Timestamp, DayTargets],
        n_lat: int,
        n_lon: int,
    ) -> None:
        self.base = base
        self.targets = targets
        self.n_lat, self.n_lon = n_lat, n_lon
        self.items = [
            (i, pd.Timestamp(d)) for i, d in enumerate(dates) if pd.Timestamp(d) in targets
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        index, day = self.items[k]
        sample = self.base[index]
        i_lat, i_lon, values = self.targets[day]
        n_depth = values.shape[1]
        y = np.zeros((n_depth, self.n_lat, self.n_lon), dtype=np.float32)
        mask = np.zeros(y.shape, dtype=bool)
        finite = np.isfinite(values)
        y[:, i_lat, i_lon] = np.where(finite, values, 0.0).T
        mask[:, i_lat, i_lon] = finite.T
        sample["argo_y"] = torch.from_numpy(y)
        sample["argo_mask"] = torch.from_numpy(mask)
        return sample


def _split_dates(ds: xr.Dataset, base: OceanScopeDataset) -> pd.DatetimeIndex:
    # The base dataset's store positions, in its own order (one per sample).
    times = pd.DatetimeIndex(ds["time"].values)
    return times[np.asarray(base._positions)].normalize()


def _bob_band_mask(ds: xr.Dataset, depths: np.ndarray) -> np.ndarray:
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    cells = (lon[None, :] >= BAY_OF_BENGAL_MIN_LON) & (lat[:, None] >= EQUATORIAL_MAX_LAT)
    band = (depths >= BAND_M[0]) & (depths <= BAND_M[1])
    return band[:, None, None] & cells[None, :, :]


@torch.no_grad()
def validate_argo(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    sigma_d: torch.Tensor,
    bob_band: torch.Tensor,
) -> tuple[float, float, int]:
    """RMSE in °C against ARGO anomalies: all cells and depths, and Bay of Bengal 50-300 m."""
    model.eval()
    sq = sq_band = 0.0
    n = n_band = 0.0
    sigma = sigma_d.view(1, -1, 1, 1)
    band = bob_band.unsqueeze(0)
    for batch in loader:
        batch = _to_device(batch, device)
        pred = _crop_to_target(model(batch["x"]).float(), batch["argo_y"])
        err = ((pred - batch["argo_y"]) * sigma) ** 2
        mask = batch["argo_mask"]
        sq += float(err[mask].sum())
        n += float(mask.sum())
        in_band = mask & band
        sq_band += float(err[in_band].sum())
        n_band += float(in_band.sum())
    rmse = float(np.sqrt(sq / n)) if n else float("nan")
    rmse_band = float(np.sqrt(sq_band / n_band)) if n_band else float("nan")
    return rmse, rmse_band, int(n)


WARM_BLOCK_DAYS = 120
WARM_CAP_QUANTILE = 0.99


def warm_scores(
    ds: xr.Dataset,
    dates: pd.DatetimeIndex,
    sigma: np.ndarray,
    band: np.ndarray,
    detrend: bool = False,
) -> np.ndarray:
    """Per day: mean positive GLORYS anomaly (in sigma units) over the Bay of Bengal 50-300 m
    band, from the training targets and the train-only climatology (never ARGO, never val).
    Zero everywhere when the grid has no band cell.

    With ``detrend`` the score is instead how far the day's mean band anomaly (signed) sits above
    the mean of that year's days, so a warming trend against the 1993-2017 climatology does not
    make whole recent years "warm" (main3_aiw: 3142 of 4264 days had weight above 2)."""
    if not band.any():
        return np.zeros(len(dates))
    depth_idx = np.flatnonzero(band.any(axis=(1, 2)))
    lat_idx = np.flatnonzero(band.any(axis=(0, 2)))
    lon_idx = np.flatnonzero(band.any(axis=(0, 1)))
    sub = band[np.ix_(depth_idx, lat_idx, lon_idx)]
    clim = np.asarray(
        ds["clim"].isel(depth=depth_idx, lat=lat_idx, lon=lon_idx).values, dtype=np.float32
    )
    sig = np.asarray(sigma, dtype=np.float32)[depth_idx][:, None, None]
    times = pd.DatetimeIndex(ds["time"].values).normalize()
    pos = times.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("A training date is not in the store")
    out = np.zeros(len(dates))
    for b0 in range(0, len(dates), WARM_BLOCK_DAYS):
        b1 = min(b0 + WARM_BLOCK_DAYS, len(dates))
        field = ds["thetao"].isel(time=pos[b0:b1], depth=depth_idx, lat=lat_idx, lon=lon_idx)
        th = np.asarray(field.transpose("time", "depth", "lat", "lon").values, dtype=np.float32)
        anom = (th - clim[dates[b0:b1].day_of_year.to_numpy() - 1]) / sig[None]
        ok = sub[None] & np.isfinite(anom)
        cells = np.where(ok, anom if detrend else np.clip(anom, 0.0, None), np.nan)
        out[b0:b1] = np.nan_to_num(np.nanmean(cells.reshape(b1 - b0, -1), axis=1))
    if detrend:
        years = pd.Series(out).groupby(dates.year.to_numpy()).transform("mean").to_numpy()
        out = np.clip(out - years, 0.0, None)
    return out


def warm_weights(scores: np.ndarray, strength: float) -> np.ndarray:
    """1 on the coolest day, ``strength`` on days at or above the 99th percentile of warmth."""
    cap = float(np.quantile(scores, WARM_CAP_QUANTILE)) if scores.size else 0.0
    if cap <= 0:
        return np.ones_like(scores)
    return 1.0 + (strength - 1.0) * np.clip(scores / cap, 0.0, 1.0)


def _loader(
    dataset: Dataset[dict[str, torch.Tensor]],
    config: ExperimentConfig,
    device: torch.device,
    order: list[int] | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    train_cfg = config.train
    data = Subset(dataset, order) if order is not None else dataset
    return DataLoader(
        data,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=bool(train_cfg.get("pin_memory", True)) and device.type == "cuda",
    )


def _write_history(config: ExperimentConfig, run_id: str, history: list[dict[str, float]]) -> Path:
    path = Path(config.paths.outputs) / "metrics" / f"finetune_{run_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in HISTORY_COLUMNS})
    return path


def finetune_run(
    config: ExperimentConfig,
    store_paths: list[Path],
    base_checkpoint: Path,
    argo_paths: list[Path],
    holdout: set[str],
    run_id: str,
    lam: float,
    stats_path: Path | None = None,
    cache_dir: Path | None = None,
    device_choice: str = "auto",
    epochs: int | None = None,
    patience: int = 3,
    argo_input: bool = False,
    huber_delta: float | None = None,
    vgrad_weight: float = 0.0,
    warm_weight: float = 1.0,
    warm_detrend: bool = False,
    argo_window: int | None = None,
) -> FineTuneResult:
    """Fine-tune ``base_checkpoint`` with ARGO targets; resumable; returns the best epoch.

    With ``argo_input`` the ARGO input channels (P4-6) are turned on: the base model's first
    layer is widened with zero weights for them, and the ARGO targets are the profiles hidden
    from the inputs (``oceanscope_ml.data.argo_inputs``), never the ones the sample sees.

    ``huber_delta`` and ``vgrad_weight`` switch on the Huber loss (GLORYS, GREP and ARGO terms)
    and the vertical-gradient term on GLORYS (``models.loss``). Both are off by default.

    ``warm_weight`` W > 1 draws training days with replacement, in proportion to a weight from 1
    (no warm anomaly) to W (the warmest 1% of days), where warmth is the mean positive GLORYS
    anomaly over the Bay of Bengal 50-300 m band (``warm_scores``). Aimed at the warm-anomaly
    errors in the regime table (DECISIONS 2026-11-06). Needs ``argo_input``. ``warm_detrend``
    judges warmth against each year's own mean instead (``warm_scores``).

    ``argo_window`` (days, with ``argo_input``) replaces the config's ARGO input window
    (``inputs.argo_input.window_days``, 10). The window is saved in the checkpoints as
    ``argo_window_days``, and ``predict.load_checkpoint`` applies it.
    """
    if warm_weight < 1.0:
        raise ValueError("warm_weight must be >= 1 (1 = off)")
    if warm_weight > 1.0 and not argo_input:
        raise ValueError("warm_weight needs argo_input")
    loss_settings = {
        "huber_delta": huber_delta,
        "vgrad_weight": float(vgrad_weight),
        "warm_weight": float(warm_weight),
    }
    if warm_detrend:  # only recorded when on, so older checkpoints still match
        loss_settings["warm_detrend"] = True
    if argo_window is not None:
        if not argo_input:
            raise ValueError("argo_window needs argo_input")
        loss_settings["argo_window"] = int(argo_window)
    no_options = {"huber_delta": None, "vgrad_weight": 0.0, "warm_weight": 1.0}
    if argo_input:
        enable_argo_input(config)
        if argo_window is not None:
            set_argo_window(config, argo_window)
    ft_cfg = config.finetune_argo
    max_epochs = int(epochs if epochs is not None else ft_cfg.get("epochs", 10))
    lr = float(ft_cfg.get("lr", 1e-4))
    aux_weight = float(config.target.get("aux_weight", 0.0))
    device = resolve_device(device_choice)
    use_amp = bool(config.train.get("amp", True)) and device.type == "cuda"
    config_hash = compute_config_hash(config.raw_dict)
    channels = channel_names(config)
    ckpt_dir = Path(config.paths.checkpoints) / run_id
    last_path, best_path = ckpt_dir / "last_ft.pt", ckpt_dir / "best_ft.pt"

    base = torch.load(base_checkpoint, map_location=device, weights_only=False)
    base_state = base["model"]
    if argo_input and argo_window is None and base.get("argo_window_days"):
        set_argo_window(config, int(base["argo_window_days"]))  # keep the base model's window
    if argo_input and base["channel_names"] != channels:
        if base["channel_names"] != [c for c in channels if not c.startswith("argo_")]:
            raise ValueError(f"{base_checkpoint} channels differ beyond the ARGO channels")
        base_state = expand_input_weights(base_state, base["channel_names"], channels)
    elif base["channel_names"] != channels:
        raise ValueError(f"{base_checkpoint} was trained with a different channel list")
    if base.get("config_hash") != config_hash:
        logger.warning("Base checkpoint config hash differs from the current config")

    set_seed(int(config.train.get("seed", 42)))
    model = build_model(config, len(channels)).to(device)
    model.load_state_dict(base_state)
    weight_decay = float(config.train.get("weight_decay", 1e-4))
    if argo_input:
        # The new ARGO weights start at zero; the first layer learns 10x faster to pick them up.
        first = [p for n, p in model.named_parameters() if n.startswith("encoders.0.0.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("encoders.0.0.")]
        groups = [{"params": first, "lr": 10 * lr}, {"params": rest, "lr": lr}]
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    depth_weights = depth_weight_vector(config).to(device)

    state = FineTuneState()
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if (
            ckpt["lambda"] != lam
            or ckpt["base_checkpoint"] != str(base_checkpoint)
            or {**no_options, **ckpt.get("loss", {})} != loss_settings
        ):
            raise ValueError(f"Run {run_id} was started with other settings. Use a new --run-id.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        _restore_rng(ckpt["rng"])
        state = FineTuneState(**ckpt["state"])
        logger.info("Resuming fine-tune %s from epoch %d", run_id, state.epoch)

    stores = open_parts(store_paths, stats_path)
    train_ds = OceanScopeDataset(
        store_paths, config, split="train", stats_path=stats_path, cache_dir=cache_dir
    )
    val_ds = OceanScopeDataset(
        store_paths, config, split="val", stats_path=stats_path, cache_dir=cache_dir
    )
    sigma_np = np.asarray(train_ds.sigma_d, dtype=float)
    n_lat, n_lon = int(stores.sizes["lat"]), int(stores.sizes["lon"])
    depths = np.asarray(config.depths_m, dtype=float)

    def targets_for(split: str) -> dict[pd.Timestamp, DayTargets]:
        start, end = (pd.Timestamp(d) for d in getattr(config.splits, split))
        levels = read_levels(argo_paths, start, end)
        profiles = profile_levels(levels, config.depths_m, holdout)
        if profiles["platform_number"].isin(holdout).any():
            raise AssertionError("A held-out float reached the fine-tuning targets")
        targets = build_targets(profiles, stores, sigma_np)
        logger.info(
            "ARGO %s: %d profiles from %d non-held-out floats on %d days",
            split,
            len(profiles),
            profiles["platform_number"].nunique(),
            len(targets),
        )
        return targets

    argo_inputs: ArgoInputs | None = None
    train_argo: Dataset[dict[str, torch.Tensor]]
    val_argo: Dataset[dict[str, torch.Tensor]]
    if argo_input:
        argo_inputs = argo_inputs_for(
            config, argo_paths, holdout, stores, sigma_np, depths_m=list(config.depths_m)
        )
        train_ds.set_argo(argo_inputs, "train")
        val_ds.set_argo(argo_inputs, "val")
        train_dates, val_dates = _split_dates(stores, train_ds), _split_dates(stores, val_ds)
        train_argo = Subset(
            train_ds, [i for i, d in enumerate(train_dates) if argo_inputs.has_profiles(d)]
        )
        val_argo = Subset(
            val_ds, [i for i, d in enumerate(val_dates) if argo_inputs.has_profiles(d)]
        )
    else:
        train_argo = ArgoTargetDataset(
            train_ds, _split_dates(stores, train_ds), targets_for("train"), n_lat, n_lon
        )
        val_argo = ArgoTargetDataset(
            val_ds, _split_dates(stores, val_ds), targets_for("val"), n_lat, n_lon
        )
    if len(train_argo) == 0 or len(val_argo) == 0:
        raise ValueError("No training or validation days with ARGO targets")
    day_weights: torch.Tensor | None = None
    if warm_weight > 1.0:
        assert isinstance(train_argo, Subset)
        days = _split_dates(stores, train_ds)[np.asarray(train_argo.indices)]
        scores = warm_scores(stores, days, sigma_np, _bob_band_mask(stores, depths),
                             detrend=warm_detrend)  # fmt: skip
        w = warm_weights(scores, warm_weight)
        day_weights = torch.as_tensor(w, dtype=torch.float64)
        logger.info("Warm-day weighting x%g: mean weight %.2f, %d of %d days above 2",
                    warm_weight, w.mean(), int((w > 2).sum()), w.size)  # fmt: skip
    logger.info("Fine-tune days: %d train, %d val", len(train_argo), len(val_argo))

    sigma_d = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    bob_band = torch.from_numpy(_bob_band_mask(stores, depths)).to(device)
    val_loader = _loader(val_ds, config, device)
    val_argo_loader = _loader(val_argo, config, device)

    def evaluate_epoch(train_loss: float) -> dict[str, float]:
        val_rmse, _ = validate(model, val_loader, device, sigma_d)  # type: ignore[arg-type]
        argo_rmse, argo_band, n_obs = validate_argo(
            model, val_argo_loader, device, sigma_d, bob_band
        )
        row = {
            "epoch": float(state.epoch),
            "lambda": lam,
            "train_loss": train_loss,
            "val_rmse_c": val_rmse,
            "val_argo_rmse_c": argo_rmse,
            "val_argo_bob_band_rmse_c": argo_band,
            "val_argo_n_obs": float(n_obs),
        }
        logger.info(
            "Fine-tune %s epoch %d: train loss %.5f, val RMSE vs GLORYS %.3f C, "
            "val ARGO RMSE %.3f C (Bay of Bengal 50-300 m %.3f C, %d obs)",
            run_id,
            state.epoch,
            train_loss,
            val_rmse,
            argo_rmse,
            argo_band,
            n_obs,
        )
        return row

    def save_best() -> None:
        _save_atomic(
            {
                "model": model.state_dict(),
                "epoch": state.epoch,
                "val_argo_rmse_c": state.best_val_argo_rmse_c,
                "channel_names": channels,
                "config_hash": config_hash,
                "run_id": run_id,
                "lambda": lam,
                "base_checkpoint": str(base_checkpoint),
                "argo_input": argo_input,
                "argo_window_days": argo_window_of(config) if argo_input else None,
                "loss": loss_settings,
            },
            best_path,
        )

    def save_last() -> None:
        _save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "state": asdict(state),
                "lambda": lam,
                "base_checkpoint": str(base_checkpoint),
                "loss": loss_settings,
                "argo_window_days": argo_window_of(config) if argo_input else None,
                "channel_names": channels,
                "config_hash": config_hash,
                "run_id": run_id,
            },
            last_path,
        )

    if not state.history:  # epoch 0: the base model, before fine-tuning
        row = evaluate_epoch(float("nan"))
        state.history.append(row)
        state.best_val_argo_rmse_c = row["val_argo_rmse_c"]
        save_best()
        save_last()
        _write_history(config, run_id, state.history)

    seed = int(config.train.get("seed", 42))
    while not state.finished and state.epoch < max_epochs:
        start = time.perf_counter()
        generator = torch.Generator().manual_seed(seed + 1000 + state.epoch)
        if day_weights is not None:
            order = torch.multinomial(
                day_weights, len(train_argo), replacement=True, generator=generator
            ).tolist()
        else:
            order = torch.randperm(len(train_argo), generator=generator).tolist()
        if argo_inputs is not None:
            argo_inputs.epoch = state.epoch  # a new draw of hidden profiles each epoch
        model.train()
        total, count = 0.0, 0
        for batch in _loader(train_argo, config, device, order):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = _crop_to_target(model(batch["x"]), batch["y"])
            pred = pred.float()
            loss = total_loss(
                pred,
                batch["y"],
                batch["mask"],
                batch["grep_y"],
                batch["grep_mask"],
                batch["has_grep"],
                depth_weights,
                aux_weight,
                huber_delta=huber_delta,
                vgrad_weight=vgrad_weight,
                sigma_d=sigma_d,
            ) + lam * masked_weighted_mse(
                pred, batch["argo_y"], batch["argo_mask"], depth_weights, huber_delta
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = batch["x"].shape[0]
            total += float(loss.detach()) * n
            count += n
        state.epoch += 1
        row = evaluate_epoch(total / max(count, 1))
        state.history.append(row)
        if row["val_argo_rmse_c"] < state.best_val_argo_rmse_c:
            state.best_val_argo_rmse_c = row["val_argo_rmse_c"]
            state.best_epoch = state.epoch
            state.epochs_since_best = 0
            save_best()
        else:
            state.epochs_since_best += 1
        if state.epochs_since_best >= patience or state.epoch >= max_epochs:
            state.finished = True
        save_last()
        _write_history(config, run_id, state.history)
        logger.info("Epoch took %.1f min", (time.perf_counter() - start) / 60.0)

    base_argo = state.history[0]["val_argo_rmse_c"]
    logger.info(
        "Fine-tune %s (lambda %g): best epoch %d, val ARGO RMSE %.3f C (base model %.3f C)",
        run_id,
        lam,
        state.best_epoch,
        state.best_val_argo_rmse_c,
        base_argo,
    )
    return FineTuneResult(
        run_id=run_id,
        lam=lam,
        best_epoch=state.best_epoch,
        best_val_argo_rmse_c=state.best_val_argo_rmse_c,
        base_val_argo_rmse_c=base_argo,
        best_path=best_path,
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARGO fine-tuning of a trained U-Net (§5.9).")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--base", required=True, help="checkpoints/<run>/best.pt")
    parser.add_argument("--argo", nargs="+", required=True, help="Monthly ARGO NetCDF files")
    parser.add_argument("--holdout", required=True, help="argo_eval_floats.json")
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--argo-input", action="store_true", help="Turn the ARGO input channels on (P4-6)"
    )
    parser.add_argument(
        "--huber-delta", type=float, default=None, help="Huber loss threshold (normalized units)"
    )
    parser.add_argument(
        "--vgrad-weight", type=float, default=0.0, help="Weight of the vertical-gradient term"
    )
    parser.add_argument(
        "--warm-weight", type=float, default=1.0, help="Draw warm Bay of Bengal days up to Wx"
    )
    parser.add_argument(
        "--argo-window", type=int, default=None, help="ARGO input window in days (default 10)"
    )
    parser.add_argument(
        "--warm-detrend", action="store_true", help="Judge warmth against each year's own mean"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    lam = args.lam if args.lam is not None else float(config.finetune_argo.get("lambda", 0.1))
    result = finetune_run(
        config,
        [Path(p) for p in args.store],
        Path(args.base),
        [Path(p) for p in args.argo],
        load_holdout(Path(args.holdout)),
        args.run_id,
        lam,
        stats_path=Path(args.stats) if args.stats else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        device_choice=args.device,
        epochs=args.epochs,
        patience=args.patience,
        argo_input=args.argo_input,
        huber_delta=args.huber_delta,
        vgrad_weight=args.vgrad_weight,
        warm_weight=args.warm_weight,
        warm_detrend=args.warm_detrend,
        argo_window=args.argo_window,
    )
    print(json.dumps({k: str(v) for k, v in asdict(result).items()}))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
