"""Small synthetic Zarr store with the layout that ``OceanScopeDataset`` reads.

The store has a 9 x 11 grid, so odd sizes are covered, and one year of daily data, which is
enough for the 60-day history. Split dates are short so tests stay fast. Statistics are computed
from the synthetic training period only, as ``build_store`` does for the real store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from oceanscope_ml.config import ExperimentConfig, load_config

EXPERIMENT_YAML = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"

SYNTHETIC_SPLITS: dict[str, list[str]] = {
    "train": ["2000-03-01", "2000-06-30"],
    "val": ["2000-08-01", "2000-09-30"],
    "test": ["2000-11-01", "2000-12-31"],
}
SPLIT_NAMES = ("train", "val", "test")
SYNTHETIC_TIME = ("2000-01-01", "2000-12-31")
N_LAT = 9
N_LON = 11
DAYS_PER_YEAR = 366  # 2000 is a leap year, so the clim axis has 366 entries
DYNAMIC_VARS = (
    "sla",
    "adt",
    "sst",
    "sss",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
    "wind_stress_curl",
    "net_heat_flux",
    "precip",
)


def synthetic_config(tmp_path: Path, **train_overrides: Any) -> ExperimentConfig:
    """The real experiment config with short synthetic splits and tiny training settings.

    The YAML is written under ``tmp_path`` and loaded again, so ``raw_dict`` (and so the config
    hash) reflects the changes.
    """
    raw = yaml.safe_load(EXPERIMENT_YAML.read_text(encoding="utf-8"))
    raw["splits"] = {**raw["splits"], **SYNTHETIC_SPLITS}
    raw["train"] = {
        **raw["train"],
        "base_channels": 8,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "amp": False,
        "max_epochs": 2,
        **train_overrides,
    }
    raw["paths"] = {
        "raw": str(tmp_path),
        "processed": str(tmp_path),
        "outputs": str(tmp_path / "outputs"),
        "checkpoints": str(tmp_path / "checkpoints"),
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(config_path)


def _ocean_mask() -> np.ndarray:
    mask = np.ones((N_LAT, N_LON), dtype=bool)
    mask[:2, :3] = False  # a land corner, so NaN and mask handling are exercised
    return mask


def _clim(depths: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Seasonal climatology [366, D, H, W] that decays with depth and varies with latitude."""
    doy = np.arange(DAYS_PER_YEAR)
    season = np.cos(2.0 * np.pi * doy / DAYS_PER_YEAR)[:, None, None, None]
    decay = np.exp(-depths / 150.0)[None, :, None, None]
    lat_term = (0.1 * (lat - lat.mean()))[None, None, :, None]
    base = 12.0 + 3.0 * season * decay + lat_term
    return np.broadcast_to(base, (DAYS_PER_YEAR, len(depths), len(lat), N_LON)).copy()


def build_synthetic_store(path: Path, config: ExperimentConfig, seed: int = 0) -> Path:
    """Write a synthetic store to ``path`` and return it. Splits come from ``config``."""
    rng = np.random.default_rng(seed)
    time = pd.date_range(*SYNTHETIC_TIME, freq="D")
    n_time = len(time)
    depths = np.asarray(config.depths_m, dtype=np.float64)
    n_depth = len(depths)
    lat = 5.125 + 0.25 * np.arange(N_LAT)
    lon = 60.125 + 0.25 * np.arange(N_LON)
    ocean = _ocean_mask()
    land_3d = ~ocean[None, None, :, :]

    doy_index = time.day_of_year.to_numpy() - 1
    clim_all = _clim(depths, lat)
    clim = clim_all.astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=(n_time, n_depth, N_LAT, N_LON))
    scale = np.exp(-depths / 300.0)[None, :, None, None] * 0.6
    thetao = clim_all[doy_index] + noise * scale
    thetao = np.where(land_3d, np.nan, thetao).astype(np.float32)

    grep = thetao + rng.normal(0.0, 0.2, size=thetao.shape).astype(np.float32)
    grep[::4] = np.nan  # every fourth day has no GREP, so has_grep is False there

    train_start, train_end = (pd.Timestamp(d) for d in config.splits.train)
    train_sel = np.asarray((time >= train_start) & (time <= train_end))
    anomaly = thetao - clim_all[doy_index]
    sigma_d = np.nanstd(anomaly[train_sel][:, :, ocean], axis=(0, 2))
    sigma_d = np.where(sigma_d > 0, sigma_d, 1.0)

    data_vars: dict[str, Any] = {}
    channel_stats: dict[str, dict[str, Any]] = {}
    for var in DYNAMIC_VARS:
        values = rng.normal(0.0, 1.0, size=(n_time, N_LAT, N_LON)) * (1.0 + len(var) / 10.0)
        values = np.where(ocean[None, :, :], values + 0.5, np.nan).astype(np.float32)
        data_vars[var] = (("time", "lat", "lon"), values)
        inside = values[train_sel][:, ocean]
        channel_stats[var] = {"mean": float(np.nanmean(inside)), "std": float(np.nanstd(inside))}

    bathymetry = np.where(ocean, rng.uniform(100.0, 4000.0, size=(N_LAT, N_LON)), np.nan)
    data_vars["bathymetry"] = (("lat", "lon"), bathymetry.astype(np.float32))
    channel_stats["bathymetry"] = {
        "mean": float(np.nanmean(bathymetry)),
        "std": float(np.nanstd(bathymetry)),
    }

    month = np.arange(1, 13)
    woa = 12.0 + 2.0 * np.cos(2.0 * np.pi * month[:, None, None, None] / 12.0)
    lat_term = (0.1 * (lat - lat.mean()))[None, None, :, None]
    woa = woa * np.exp(-depths / 150.0)[None, :, None, None] + lat_term
    woa = np.where(ocean[None, None, :, :], woa, np.nan).astype(np.float32)
    data_vars["woa_clim_profile"] = (("month", "depth", "lat", "lon"), woa)
    woa_mean = [float(np.nanmean(woa[:, d][:, ocean])) for d in range(n_depth)]
    woa_std = [float(np.nanstd(woa[:, d][:, ocean])) for d in range(n_depth)]
    channel_stats["woa_clim_profile"] = {"mean": woa_mean, "std": woa_std}

    data_vars["ocean_mask"] = (("lat", "lon"), ocean)
    data_vars["thetao"] = (("time", "depth", "lat", "lon"), thetao)
    data_vars["grep_mean"] = (("time", "depth", "lat", "lon"), grep.astype(np.float32))
    data_vars["target_valid"] = (
        ("time", "depth", "lat", "lon"),
        np.broadcast_to(ocean[None, None], thetao.shape).copy(),
    )
    data_vars["clim"] = (("dayofyear", "depth", "lat", "lon"), clim)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "depth": depths,
            "lat": lat,
            "lon": lon,
            "month": month,
            "dayofyear": np.arange(1, DAYS_PER_YEAR + 1),
        },
    )
    ds.attrs["norm_stats"] = {
        "channels": channel_stats,
        "sigma_d": [float(s) for s in sigma_d],
    }
    ds.attrs["split_dates"] = {name: list(getattr(config.splits, name)) for name in SPLIT_NAMES}
    ds.attrs["synthetic"] = True

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w")
    return path
