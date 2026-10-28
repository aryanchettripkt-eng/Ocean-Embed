"""U-Net training with resume and run logging (PRD_AI_ML §5.5).

Usage (from ml/):

    python -m oceanscope_ml.train --config configs/experiment.yaml \
        [--store data/processed/oceanscope_v1.zarr] [--run-id ID] [--device auto|cpu|cuda] \
        [--epochs N]

Training and early stopping use the validation split only. The test split is never read here.

Each epoch writes ``<checkpoints>/<run_id>/last.pt``, which holds the model, optimizer, scheduler,
AMP scaler, epoch counter, best-so-far values, RNG states and config hash. The best model by
validation RMSE (in °C) goes to ``best.pt``. Running the same command again resumes from
``last.pt``. A finished run appends one row to ``<outputs>/metrics/runs.csv``.

Scheduler: cosine annealing from ``train.lr`` to ``train.min_lr`` over ``train.max_epochs``,
stepped once per epoch. The schedule always uses ``max_epochs`` from the config, so ``--epochs``
only controls how far this invocation runs, and a resumed run follows the same schedule.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.preprocess import compute_config_hash
from oceanscope_ml.models.loss import depth_weight_vector, total_loss
from oceanscope_ml.models.unet import UNet

logger = logging.getLogger("oceanscope_ml.train")

RUNS_COLUMNS = [
    "run_id",
    "git_commit",
    "dataset_version",
    "config_hash",
    "epochs",
    "best_epoch",
    "val_rmse_c_mean",
    "val_mae_c_mean",
    "wall_minutes",
    "gpu_name",
    "torch_version",
]


@dataclass
class RunState:
    """Progress that must survive a restart. Stored in last.pt."""

    epoch: int = 0  # completed epochs
    best_epoch: int = 0
    best_val_rmse_c: float = float("inf")
    best_val_mae_c: float = float("nan")
    epochs_since_best: int = 0
    wall_seconds: float = 0.0
    finished: bool = False


@dataclass
class TrainResult:
    run_id: str
    epochs_completed: int
    best_epoch: int
    best_val_rmse_c: float
    finished: bool
    n_channels: int
    n_parameters: int
    checkpoint_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def git_short_hash() -> str:
    """Short commit hash of the code being run, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def default_run_id(model_name: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{date}_{model_name}_{git_short_hash()}"


def build_model(config: ExperimentConfig, n_channels: int) -> UNet:
    train_cfg = config.train
    if str(train_cfg.get("model", "unet")) != "unet":
        raise ValueError("Only train.model: unet is implemented")
    return UNet(
        in_channels=n_channels,
        out_channels=len(config.depths_m),
        base_channels=int(train_cfg.get("base_channels", 32)),
        depth_levels=int(train_cfg.get("depth_levels", 4)),
    )


def build_optimizer(config: ExperimentConfig, model: UNet) -> torch.optim.Optimizer:
    train_cfg = config.train
    if str(train_cfg.get("optimizer", "adamw")) != "adamw":
        raise ValueError("Only train.optimizer: adamw is implemented")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(
    config: ExperimentConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    train_cfg = config.train
    if str(train_cfg.get("scheduler", "cosine")) != "cosine":
        raise ValueError("Only train.scheduler: cosine is implemented")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("max_epochs", 80)),
        eta_min=float(train_cfg.get("min_lr", 1e-5)),
    )


def epoch_loader(
    dataset: OceanScopeDataset,
    epoch: int,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Shuffled loader whose order depends only on (seed, epoch), so a resumed run sees the
    same batches as an uninterrupted one."""
    train_cfg = config.train
    generator = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)) + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    return DataLoader(
        Subset(dataset, order),
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=bool(train_cfg.get("pin_memory", True)) and device.type == "cuda",
    )


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _crop_to_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cut the prediction back to the target grid.

    The dataset pads inputs on the bottom and right to a multiple of 2**depth_levels, and the
    targets keep the store grid. The U-Net returns the padded size, so the extra rows and columns
    are dropped here and never enter the loss or the validation error.
    """
    return pred[..., : target.shape[-2], : target.shape[-1]]


def train_one_epoch(
    model: UNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    depth_weights: torch.Tensor,
    aux_weight: float,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = _crop_to_target(model(batch["x"]), batch["y"])
        loss = total_loss(
            pred.float(),
            batch["y"],
            batch["mask"],
            batch["grep_y"],
            batch["grep_mask"],
            batch["has_grep"],
            depth_weights,
            aux_weight,
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = batch["x"].shape[0]
        total += float(loss.detach()) * n
        count += n
    return total / max(count, 1)


@torch.no_grad()
def validate(
    model: UNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    sigma_d: torch.Tensor,
) -> tuple[float, float]:
    """Masked RMSE and MAE in degrees C over all valid validation cells and depths."""
    model.eval()
    sq_sum = 0.0
    abs_sum = 0.0
    n_valid = 0.0
    sigma = sigma_d.view(1, -1, 1, 1)
    for batch in loader:
        batch = _to_device(batch, device)
        pred = _crop_to_target(model(batch["x"]).float(), batch["y"])
        error_c = (pred - batch["y"]) * sigma
        mask = batch["mask"].float()
        sq_sum += float(((error_c * mask) ** 2).sum())
        abs_sum += float((error_c.abs() * mask).sum())
        n_valid += float(mask.sum())
    if n_valid == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(sq_sum / n_valid)), abs_sum / n_valid


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_atomic(obj: dict[str, Any], path: Path) -> None:
    """Write to a temporary file, then rename, so an interrupted save never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_run_row(config: ExperimentConfig, row: dict[str, Any]) -> Path:
    runs_csv = Path(config.paths.outputs) / "metrics" / "runs.csv"
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not runs_csv.exists()
    with runs_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUNS_COLUMNS})
    return runs_csv


def _gpu_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def train_run(
    config: ExperimentConfig,
    store_path: str | Path | list[Path],
    run_id: str,
    device_choice: str = "auto",
    epochs: int | None = None,
    stats_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> TrainResult:
    """Train or resume one run, up to ``epochs`` completed epochs (default: max_epochs)."""
    train_cfg = config.train
    max_epochs = int(train_cfg.get("max_epochs", 80))
    stop_epoch = min(max_epochs, epochs) if epochs is not None else max_epochs
    patience = int(train_cfg.get("early_stopping_patience", 10))
    aux_weight = float(config.target.get("aux_weight", 0.0))
    use_amp_flag = bool(train_cfg.get("amp", True))
    device = resolve_device(device_choice)
    use_amp = use_amp_flag and device.type == "cuda"
    config_hash = compute_config_hash(config.raw_dict)
    ckpt_dir = Path(config.paths.checkpoints) / run_id
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    set_seed(int(train_cfg.get("seed", 42)))
    channels = channel_names(config)
    model = build_model(config, len(channels)).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    depth_weights = depth_weight_vector(config).to(device)

    state = RunState()
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["config_hash"] != config_hash:
            raise ValueError(
                f"Run {run_id} was started with config hash {checkpoint['config_hash']}, "
                f"but the current config hash is {config_hash}. Use a new --run-id."
            )
        if checkpoint["channel_names"] != channels:
            raise ValueError("Channel order changed since this run started. Use a new --run-id.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        state = RunState(**checkpoint["state"])
        logger.info("Resuming run %s from epoch %d", run_id, state.epoch)
    else:
        logger.info("Starting run %s on %s", run_id, device)

    if state.finished or state.epoch >= stop_epoch:
        logger.info("Run %s has nothing left to do (epoch %d)", run_id, state.epoch)
        return _result(run_id, state, len(channels), model, ckpt_dir)

    train_ds = OceanScopeDataset(
        store_path, config, split="train", stats_path=stats_path, cache_dir=cache_dir
    )
    val_ds = OceanScopeDataset(
        store_path, config, split="val", stats_path=stats_path, cache_dir=cache_dir
    )
    if train_ds.n_channels != len(channels):
        raise ValueError("Dataset channel count does not match channel_list")
    sigma_d = torch.tensor(train_ds.sigma_d, dtype=torch.float32, device=device)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=bool(train_cfg.get("pin_memory", True)) and device.type == "cuda",
    )

    logger.info(
        "Run %s: %d input channels, %d parameters, epochs %d -> %d",
        run_id,
        len(channels),
        model.n_parameters,
        state.epoch,
        stop_epoch,
    )

    while state.epoch < stop_epoch:
        epoch_start = time.perf_counter()
        loader = epoch_loader(train_ds, state.epoch, config, device)
        train_loss = train_one_epoch(
            model, loader, optimizer, scaler, device, use_amp, depth_weights, aux_weight
        )
        val_rmse, val_mae = validate(model, val_loader, device, sigma_d)
        scheduler.step()
        state.epoch += 1
        state.wall_seconds += time.perf_counter() - epoch_start

        improved = val_rmse < state.best_val_rmse_c
        if improved:
            state.best_val_rmse_c = val_rmse
            state.best_val_mae_c = val_mae
            state.best_epoch = state.epoch
            state.epochs_since_best = 0
        else:
            state.epochs_since_best += 1

        logger.info(
            "Epoch %d/%d: train loss %.5f, val RMSE %.3f C, val MAE %.3f C, lr %.2e%s",
            state.epoch,
            max_epochs,
            train_loss,
            val_rmse,
            val_mae,
            scheduler.get_last_lr()[0],
            " (best)" if improved else "",
        )

        if improved:
            _save_atomic(
                {
                    "model": model.state_dict(),
                    "epoch": state.epoch,
                    "val_rmse_c": val_rmse,
                    "channel_names": channels,
                    "config_hash": config_hash,
                    "run_id": run_id,
                },
                best_path,
            )

        if state.epochs_since_best >= patience:
            logger.info("Early stopping after %d epochs without improvement", patience)
            state.finished = True
        elif state.epoch >= max_epochs:
            state.finished = True

        _save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "state": asdict(state),
                "channel_names": channels,
                "config_hash": config_hash,
                "run_id": run_id,
            },
            last_path,
        )

        if state.finished:
            _append_run_row(
                config,
                {
                    "run_id": run_id,
                    "git_commit": git_short_hash(),
                    "dataset_version": config.dataset_version,
                    "config_hash": config_hash,
                    "epochs": state.epoch,
                    "best_epoch": state.best_epoch,
                    "val_rmse_c_mean": state.best_val_rmse_c,
                    "val_mae_c_mean": state.best_val_mae_c,
                    "wall_minutes": round(state.wall_seconds / 60.0, 3),
                    "gpu_name": _gpu_name(device),
                    "torch_version": torch.__version__,
                },
            )
            break

    return _result(run_id, state, len(channels), model, ckpt_dir)


def _result(
    run_id: str, state: RunState, n_channels: int, model: UNet, ckpt_dir: Path
) -> TrainResult:
    return TrainResult(
        run_id=run_id,
        epochs_completed=state.epoch,
        best_epoch=state.best_epoch,
        best_val_rmse_c=state.best_val_rmse_c,
        finished=state.finished,
        n_channels=n_channels,
        n_parameters=model.n_parameters,
        checkpoint_dir=ckpt_dir,
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or resume the OceanScope U-Net.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--store",
        nargs="+",
        default=None,
        help="Processed Zarr store, or the part stores of a multi-part build (default from config)",
    )
    parser.add_argument(
        "--stats", default=None, help="Stats store with clim and norm_stats, for a multi-part build"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local fast-disk folder for the training cache, built on first use (much faster)",
    )
    parser.add_argument("--run-id", default=None, help="Run name (default: date_unet_githash)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Stop once this many epochs are complete (default: train.max_epochs)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    store: Path | list[Path] = (
        [Path(p) for p in args.store]
        if args.store
        else Path(config.paths.processed) / f"oceanscope_{config.dataset_version}.zarr"
    )
    run_id = args.run_id or default_run_id(str(config.train.get("model", "unet")))
    result = train_run(
        config,
        store,
        run_id,
        device_choice=args.device,
        epochs=args.epochs,
        stats_path=args.stats,
        cache_dir=args.cache_dir,
    )
    logger.info(
        "Run %s: %d epochs complete, best epoch %d (val RMSE %.3f C), finished=%s",
        result.run_id,
        result.epochs_completed,
        result.best_epoch,
        result.best_val_rmse_c,
        result.finished,
    )


if __name__ == "__main__":
    main()
