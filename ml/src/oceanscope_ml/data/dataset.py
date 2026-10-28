"""PyTorch dataset over the processed Zarr store (PRD_AI_ML §5.1, §5.2, §5.4).

One sample is one target day with its history window. The store is read lazily, one day at a
time, so the whole store never has to fit in memory.

Normalization statistics and sigma_d come from the store attributes. This module never computes
statistics. Test-period dates are refused (PRD §0 rule 3).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.channels import ChannelSpec, channel_list
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import TestLeakageError, check_no_test_leakage

logger = logging.getLogger("oceanscope_ml.data.dataset")

TRAINABLE_SPLITS = ("train", "val")


def _split_range(config: ExperimentConfig, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    bounds = getattr(config.splits, split)
    return pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])


def _lookback_days(channels: list[ChannelSpec]) -> int:
    """Number of days before the target day that the channels read."""
    return max((c.lag + c.window_days - 1 for c in channels), default=0)


def _check_store_matches_config(ds: xr.Dataset, config: ExperimentConfig) -> None:
    times = pd.to_datetime(ds["time"].values)
    if len(times) > 1 and not np.all(np.diff(times.values) == np.timedelta64(1, "D")):
        raise ValueError("Store time axis must be consecutive days without gaps")

    store_depths = np.asarray(ds["depth"].values, dtype=np.float64)
    cfg_depths = np.asarray(config.depths_m, dtype=np.float64)
    if store_depths.shape != cfg_depths.shape or np.any(np.abs(store_depths - cfg_depths) > 0.05):
        raise ValueError(
            f"Store depths {store_depths.tolist()} do not match "
            f"config depths_m {cfg_depths.tolist()}"
        )

    stored_splits = ds.attrs.get("split_dates")
    if stored_splits is not None:
        for name in ("train", "val", "test"):
            stored = [str(x) for x in stored_splits.get(name, [])]
            wanted = [str(x) for x in getattr(config.splits, name)]
            if stored and stored != wanted:
                raise ValueError(
                    f"Store split '{name}' is {stored} but config says {wanted}. "
                    "Rebuild the store or fix the config."
                )


class OceanScopeDataset(torch.utils.data.Dataset):
    """Target days from one split, each with its input history.

    Returns a dict with:
    - ``x``: float32 [C, H_pad, W_pad]; normalized inputs, NaN set to 0, zero-padded.
    - ``y``: float32 [D, H, W]; (thetao - clim) / sigma_d, 0 where ``mask`` is False.
    - ``mask``: bool [D, H, W]; target valid.
    - ``grep_y``, ``grep_mask``: as ``y`` and ``mask`` for GREP, zeros when GREP is missing.
    - ``has_grep``: bool scalar; True when GREP exists for this date.
    """

    def __init__(
        self,
        store_path: str | Path | list[Path],
        config: ExperimentConfig,
        split: str,
        stats_path: str | Path | None = None,
    ) -> None:
        if split not in TRAINABLE_SPLITS:
            raise TestLeakageError(
                f"Dataset split must be one of {TRAINABLE_SPLITS}; got {split!r}. "
                "The test period is read only by final_test (PRD_AI_ML §0 rule 3)."
            )
        # One store, or the part stores of a multi-part build plus the stats store that
        # holds `clim` and `norm_stats` (oceanscope_ml.data.parts).
        paths = store_path if isinstance(store_path, list) else [store_path]
        self.store_paths = [Path(p) for p in paths]
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.config = config
        self.split = split
        self.channels = channel_list(config)
        self.depth_levels = int(config.train.get("depth_levels", 4))
        self.pad_multiple = 2**self.depth_levels
        self._lookback = _lookback_days(self.channels)
        self._ds: xr.Dataset | None = None

        with self._open_store() as ds:
            _check_store_matches_config(ds, config)
            self._check_norm_stats(ds)
            self.sigma_d = np.asarray(ds.attrs["norm_stats"]["sigma_d"], dtype=np.float32)
            self._norm = self._build_norm_table(ds)
            self._lat_scaled, self._lon_scaled = self._scale_lat_lon(ds)
            self._positions = self._select_positions(ds)
            self._n_lat = int(ds.sizes["lat"])
            self._n_lon = int(ds.sizes["lon"])
            self._has_grep = "grep_mean" in ds.data_vars
            self._has_target_valid = "target_valid" in ds.data_vars

        if self.sigma_d.shape != (len(config.depths_m),) or np.any(self.sigma_d <= 0):
            raise ValueError("norm_stats sigma_d must be positive, one value per depth")

        self.n_channels = len(self.channels)
        self.padded_hw = (
            _round_up(self._n_lat, self.pad_multiple),
            _round_up(self._n_lon, self.pad_multiple),
        )
        logger.info(
            "Dataset %s: %d samples, %d channels, grid %dx%d padded to %dx%d",
            split,
            len(self._positions),
            self.n_channels,
            self._n_lat,
            self._n_lon,
            self.padded_hw[0],
            self.padded_hw[1],
        )

    # ------------------------------------------------------------------ setup

    def _check_norm_stats(self, ds: xr.Dataset) -> None:
        stats = ds.attrs.get("norm_stats")
        if not stats or "channels" not in stats or "sigma_d" not in stats:
            raise ValueError("Store has no norm_stats with 'channels' and 'sigma_d'")
        if ds.attrs.get("sample_only") or ds.attrs.get("norm_stats_sample_only"):
            raise ValueError(
                "Store norm_stats are sample-only, not training-period statistics. "
                "Build the full store before training."
            )

    def _build_norm_table(self, ds: xr.Dataset) -> dict[str, tuple[Any, Any]]:
        """Map each normalized source variable to its (mean, std) from the store attributes."""
        channel_stats: dict[str, Any] = ds.attrs["norm_stats"]["channels"]
        table: dict[str, tuple[Any, Any]] = {}
        for channel in self.channels:
            if channel.kind == "extra" or channel.kind == "argo":
                continue
            key = channel.source
            if key in table:
                continue
            if key not in channel_stats:
                raise KeyError(
                    f"norm_stats has no entry for '{key}'. The store must provide "
                    "training-period statistics for every input channel."
                )
            entry = channel_stats[key]
            table[key] = (entry["mean"], entry["std"])
        return table

    def _scale_lat_lon(self, ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        def scale(values: np.ndarray) -> np.ndarray:
            lo, hi = float(values.min()), float(values.max())
            if hi == lo:
                return np.zeros(values.shape, dtype=np.float32)
            return (2.0 * (values - lo) / (hi - lo) - 1.0).astype(np.float32)

        return scale(np.asarray(ds["lat"].values, dtype=np.float64)), scale(
            np.asarray(ds["lon"].values, dtype=np.float64)
        )

    def _select_positions(self, ds: xr.Dataset) -> np.ndarray:
        """Store time positions for this split, after the gap and history checks."""
        times = pd.to_datetime(ds["time"].values)
        start, end = _split_range(self.config, self.split)
        gap = pd.Timedelta(days=int(self.config.splits.gap_days))
        last_day = end - gap  # the gap_days at the end of train and val are dropped

        in_split = (times >= start) & (times <= last_day)
        candidates = np.flatnonzero(in_split)
        positions = candidates[candidates >= self._lookback]
        dropped = len(candidates) - len(positions)
        if dropped:
            logger.info(
                "%s: %d days dropped for lack of %d days of history",
                self.split,
                dropped,
                self._lookback,
            )
        if len(positions) == 0:
            raise ValueError(f"No {self.split} samples with enough history in the store")

        selected_dates = times[positions]
        check_no_test_leakage(selected_dates, self.config.splits.test)
        return positions

    # ------------------------------------------------------------- dataset API

    def __len__(self) -> int:
        return len(self._positions)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_ds"] = None  # open handles do not cross process boundaries
        return state

    def _open_store(self) -> xr.Dataset:
        return open_parts(self.store_paths, self.stats_path)

    def _open(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = self._open_store()
        return self._ds

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ds = self._open()
        t = int(self._positions[index])
        day = pd.Timestamp(ds["time"].values[t])
        doy = day.day_of_year
        window = {
            var: ds[var].isel(time=slice(t - self._lookback, t + 1)).values
            for var in self._window_vars()
        }

        planes = [self._input_plane(c, ds, window, t, day, doy) for c in self.channels]
        x = np.stack(planes).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)

        y, mask, grep_y, grep_mask, has_grep = self._targets(ds, t, doy)

        pad_h = self.padded_hw[0] - self._n_lat
        pad_w = self.padded_hw[1] - self._n_lon
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "grep_y": torch.from_numpy(grep_y),
            "grep_mask": torch.from_numpy(grep_mask),
            "has_grep": torch.tensor(has_grep, dtype=torch.bool),
        }

    def _window_vars(self) -> list[str]:
        return sorted({c.source for c in self.channels if c.kind in ("dynamic", "long_history")})

    def _input_plane(
        self,
        channel: ChannelSpec,
        ds: xr.Dataset,
        window: dict[str, np.ndarray],
        t: int,
        day: pd.Timestamp,
        doy: int,
    ) -> np.ndarray:
        """One channel as a float array [H, W], before normalization."""
        shape = (self._n_lat, self._n_lon)
        kind = channel.kind

        if kind == "dynamic":
            plane = window[channel.source][self._lookback - channel.lag]
            return self._normalize(channel.source, plane)

        if kind == "long_history":
            # Index 0 of the window block is lag _lookback; lag L is at index _lookback - L.
            first = channel.lag
            rows = [
                window[channel.source][self._lookback - (first + k)]
                for k in range(channel.window_days)
            ]
            with warnings.catch_warnings():
                # All-NaN cells (land) are expected here and become 0 after nan_to_num.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                frame = np.nanmean(np.stack(rows), axis=0)
            return self._normalize(channel.source, frame)

        if kind == "index":
            value = float(ds[channel.source].isel(time=t).values)
            return self._normalize(channel.source, np.full(shape, value))

        if kind == "static":
            if channel.source == "bathymetry":
                return self._normalize("bathymetry", ds["bathymetry"].values)
            month = day.month
            woa = ds["woa_clim_profile"].sel(month=month).isel(depth=channel.depth_index)
            return self._normalize_depth("woa_clim_profile", channel.depth_index, woa.values)

        if kind == "extra":
            if channel.source == "doy_sin":
                return np.full(shape, np.sin(2.0 * np.pi * doy / 366.0))
            if channel.source == "doy_cos":
                return np.full(shape, np.cos(2.0 * np.pi * doy / 366.0))
            if channel.source == "lat":
                return np.broadcast_to(self._lat_scaled[:, None], shape).astype(np.float64)
            if channel.source == "lon":
                return np.broadcast_to(self._lon_scaled[None, :], shape).astype(np.float64)
            if channel.source == "ocean_mask":
                return np.asarray(ds["ocean_mask"].values, dtype=np.float64)

        raise NotImplementedError(f"Channel kind {kind!r} is not built yet")

    def _normalize(self, source: str, plane: np.ndarray) -> np.ndarray:
        mean, std = self._norm[source]
        return (np.asarray(plane, dtype=np.float64) - float(mean)) / float(std)

    def _normalize_depth(
        self, source: str, depth_index: int | None, plane: np.ndarray
    ) -> np.ndarray:
        mean, std = self._norm[source]
        if isinstance(mean, list):
            if depth_index is None:
                raise ValueError(f"{source} stats are per depth but no depth was given")
            mean, std = mean[depth_index], std[depth_index]
        return (np.asarray(plane, dtype=np.float64) - float(mean)) / float(std)

    def _targets(
        self, ds: xr.Dataset, t: int, doy: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        clim_day = self._clim(ds, doy)
        sigma = self.sigma_d[:, None, None]

        thetao = np.asarray(ds["thetao"].isel(time=t).values, dtype=np.float64)
        anomaly = (thetao - clim_day) / sigma
        mask = np.isfinite(anomaly)
        if self._has_target_valid:
            mask &= np.asarray(ds["target_valid"].isel(time=t).values, dtype=bool)
        y = np.where(mask, anomaly, 0.0).astype(np.float32)

        grep_y = np.zeros_like(y)
        grep_mask = np.zeros(mask.shape, dtype=bool)
        has_grep = False
        if self._has_grep:
            grep = np.asarray(ds["grep_mean"].isel(time=t).values, dtype=np.float64)
            grep_anomaly = (grep - clim_day) / sigma
            grep_mask = np.isfinite(grep_anomaly)
            has_grep = bool(grep_mask.any())
            grep_y = np.where(grep_mask, grep_anomaly, 0.0).astype(np.float32)
        return y, mask, grep_y, grep_mask, has_grep

    def _clim(self, ds: xr.Dataset, doy: int) -> np.ndarray:
        if ds.sizes["dayofyear"] != 366:
            raise ValueError("clim must have 366 day-of-year entries")
        return np.asarray(ds["clim"].isel(dayofyear=doy - 1).values, dtype=np.float64)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple
