"""Normalization statistics, data transforms, and test leakage guards.

Implements PRD_AI_ML §0 (rule 3, 4), §5.1, §5.2, §6.1, and §8 (tests 3, 5).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("oceanscope_ml.data.transform")


class TestLeakageError(ValueError):
    """Raised when test-period data is accessed outside final_test.py (PRD_AI_ML §0 rule 3)."""

    __test__ = False


def check_no_test_leakage(
    dates: pd.DatetimeIndex | np.ndarray | list[Any],
    test_split: list[str] | tuple[str, str],
) -> None:
    """Verify that no date falls within the sealed test period.

    Parameters
    ----------
    dates : collection of date strings or timestamps
    test_split : [test_start, test_end], e.g. ["2021-01-01", "2021-12-31"]

    Raises
    ------
    TestLeakageError : if any date is inside [test_start, test_end]
    """
    if len(test_split) < 2:
        return

    test_start = pd.Timestamp(test_split[0])
    test_end = pd.Timestamp(test_split[1])

    dt_series = pd.to_datetime(dates)
    leakage_mask = (dt_series >= test_start) & (dt_series <= test_end)

    if np.any(leakage_mask):
        leaked_count = int(np.sum(leakage_mask))
        sample_leaked = dt_series[leakage_mask][0].strftime("%Y-%m-%d")
        raise TestLeakageError(
            f"Test leakage detected! {leaked_count} dates fall inside sealed test split "
            f"[{test_start.strftime('%Y-%m-%d')} .. {test_end.strftime('%Y-%m-%d')}]. "
            f"First leaked date: {sample_leaked}. PRD_AI_ML §0 rule 3 strictly forbids "
            "evaluating or computing statistics on the test period."
        )


def compute_norm_stats(
    ds: xr.Dataset,
    train_split: list[str],
    test_split: list[str],
    channels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute training-only per-channel mean/std and per-depth sigma_d.

    Parameters
    ----------
    ds : xr.Dataset containing input variables, thetao, clim, ocean_mask
    train_split : [train_start, train_end]
    test_split : [test_start, test_end]
    channels : list of channel variable names, or None for defaults

    Returns
    -------
    dict[str, Any]
        {"channels": {name: {"mean": float, "std": float}}, "sigma_d": list[float]}
    """
    train_start = pd.Timestamp(train_split[0])
    train_end = pd.Timestamp(train_split[1])

    # Guard against test leakage: training period dates must not fall in test split
    # (PRD §0 rule 3, rule 4)
    check_no_test_leakage([train_start, train_end], test_split)

    # Subset dataset to training period only
    ds_train = ds.sel(time=slice(train_start, train_end))
    if len(ds_train.coords["time"]) == 0:
        raise ValueError(
            f"No dates found in training period [{train_start} .. {train_end}]. "
            "Cannot compute training-only normalization statistics."
        )

    # Verify sliced training dates contain no test leakage
    check_no_test_leakage(ds_train.coords["time"].values, test_split)

    # Ocean mask: valid ocean cells
    if "ocean_mask" in ds_train:
        ocean_mask = ds_train["ocean_mask"].values.astype(bool)
    else:
        # Default: non-NaN on surface layer
        ocean_mask = ~np.isnan(ds_train["thetao"].values[0, 0])

    default_channels = [
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
    ]
    target_channels = channels or [c for c in default_channels if c in ds_train.data_vars]

    channel_stats: dict[str, dict[str, float]] = {}

    for ch in target_channels:
        vals = ds_train[ch].values  # shape (time, lat, lon)
        # Select ocean cells across all training dates
        ocean_vals = vals[:, ocean_mask]
        valid_ocean_vals = ocean_vals[~np.isnan(ocean_vals)]

        if len(valid_ocean_vals) > 0:
            ch_mean = float(np.mean(valid_ocean_vals))
            ch_std = float(np.std(valid_ocean_vals))
            # Relative zero variance guard: preserve small variables (curl ~1e-7; REVIEW_P3 R4)
            if not np.isfinite(ch_std) or ch_std == 0.0 or ch_std < 1e-12:
                ch_std = 1.0
        else:
            ch_mean = 0.0
            ch_std = 1.0

        channel_stats[ch] = {"mean": ch_mean, "std": ch_std}

    # Static channel: bathymetry
    if "bathymetry" in ds_train.data_vars:
        bathy_vals = ds_train["bathymetry"].values[ocean_mask]
        valid_bathy = bathy_vals[~np.isnan(bathy_vals)]
        if len(valid_bathy) > 0:
            channel_stats["bathymetry"] = {
                "mean": float(np.mean(valid_bathy)),
                "std": float(np.std(valid_bathy)),
            }

    # Per-depth anomaly standard deviation sigma_d: std(thetao - clim) over train dates
    # Computed via streaming running sums over time chunks (REVIEW_P3 R3 / memory)
    sigma_d_list: list[float] = []
    if "thetao" in ds_train.data_vars and "clim" in ds_train.data_vars:
        da_thetao = ds_train["thetao"]
        clim_train_vals = ds_train["clim"].values
        train_dates = pd.to_datetime(ds_train.coords["time"].values)
        n_time = len(train_dates)
        n_depths = (
            len(ds_train.coords["depth"]) if "depth" in ds_train.coords else da_thetao.shape[1]
        )

        count_d = np.zeros(n_depths, dtype=np.int64)
        sum_d = np.zeros(n_depths, dtype=np.float64)
        sum_sq_d = np.zeros(n_depths, dtype=np.float64)

        chunk_size = 365
        for start_t in range(0, n_time, chunk_size):
            end_t = min(start_t + chunk_size, n_time)
            chunk_dates = train_dates[start_t:end_t]
            thetao_chunk = da_thetao.isel(time=slice(start_t, end_t)).values

            if clim_train_vals.shape[0] == 366:
                doy_idx = chunk_dates.day_of_year.values - 1
                clim_chunk = clim_train_vals[doy_idx]
            else:
                clim_chunk = clim_train_vals[start_t:end_t]

            anom_chunk = thetao_chunk - clim_chunk

            for d_idx in range(n_depths):
                depth_anom = anom_chunk[:, d_idx, :, :]
                ocean_anom = depth_anom[:, ocean_mask]
                valid_anom = ocean_anom[np.isfinite(ocean_anom)]
                if len(valid_anom) > 0:
                    count_d[d_idx] += len(valid_anom)
                    sum_d[d_idx] += float(np.sum(valid_anom, dtype=np.float64))
                    sum_sq_d[d_idx] += float(np.sum(valid_anom**2, dtype=np.float64))

        for d_idx in range(n_depths):
            if count_d[d_idx] > 0:
                mean_d = sum_d[d_idx] / count_d[d_idx]
                var_d = (sum_sq_d[d_idx] / count_d[d_idx]) - (mean_d**2)
                sig_d = float(np.sqrt(max(var_d, 0.0)))
                if not np.isfinite(sig_d) or sig_d < 1e-6:
                    sig_d = 1.0
            else:
                sig_d = 1.0
            sigma_d_list.append(sig_d)
    else:
        # Fallback 15 depths default
        sigma_d_list = [1.0] * 15

    return {
        "channels": channel_stats,
        "sigma_d": sigma_d_list,
    }


def normalize_inputs(
    arr: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Normalize input channel by z-scoring: (x - mean) / std."""
    safe_std = 1.0 if not np.isfinite(std) or abs(std) < 1e-12 else std
    return ((arr - mean) / safe_std).astype(np.float32)


def denormalize_inputs(
    norm_arr: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Denormalize input channel: x_norm * std + mean."""
    safe_std = 1.0 if abs(std) < 1e-6 else std
    return (norm_arr * safe_std + mean).astype(np.float32)


def normalize_target(
    thetao: np.ndarray,
    clim: np.ndarray,
    sigma_d: np.ndarray | list[float],
) -> np.ndarray:
    """Normalize target temperatures to anomalies scaled by sigma_d: (thetao - clim) / sigma_d.

    Parameters
    ----------
    thetao : np.ndarray, shape (..., depth, lat, lon)
    clim : np.ndarray, same shape as thetao
    sigma_d : shape (depth,) or list of length depth

    Returns
    -------
    np.ndarray, shape of thetao, float32
    """
    sig = np.array(sigma_d, dtype=np.float32)
    # Broadcast sigma_d along depth dimension: (1, depth, 1, 1) or (depth, 1, 1)
    if thetao.ndim == 4:
        sig_broadcast = sig[np.newaxis, :, np.newaxis, np.newaxis]
    elif thetao.ndim == 3:
        sig_broadcast = sig[:, np.newaxis, np.newaxis]
    else:
        sig_broadcast = sig

    safe_sig = np.where(np.abs(sig_broadcast) < 1e-6, 1.0, sig_broadcast)
    return ((thetao - clim) / safe_sig).astype(np.float32)


def denormalize_target(
    y_norm: np.ndarray,
    clim: np.ndarray,
    sigma_d: np.ndarray | list[float],
) -> np.ndarray:
    """Denormalize target anomalies back to temperatures: y_norm * sigma_d + clim.

    Parameters
    ----------
    y_norm : np.ndarray, normalized anomalies, shape (..., depth, lat, lon)
    clim : np.ndarray, climatology, same shape as y_norm
    sigma_d : shape (depth,) or list of length depth

    Returns
    -------
    np.ndarray, reconstructed temperature in °C, float32
    """
    sig = np.array(sigma_d, dtype=np.float32)
    if y_norm.ndim == 4:
        sig_broadcast = sig[np.newaxis, :, np.newaxis, np.newaxis]
    elif y_norm.ndim == 3:
        sig_broadcast = sig[:, np.newaxis, np.newaxis]
    else:
        sig_broadcast = sig

    safe_sig = np.where(np.abs(sig_broadcast) < 1e-6, 1.0, sig_broadcast)
    return (y_norm * safe_sig + clim).astype(np.float32)
