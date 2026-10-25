"""Harmonic climatology model and fitting for OceanScope AI (PRD_AI_ML §2, §5.2).

Fits annual harmonic cycles (mean + 3 harmonics) per cell and depth over
the training period only. Evaluates for dayofyear 1..366.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("oceanscope_ml.data.climatology")


def build_harmonic_design_matrix(
    dayofyears: np.ndarray,
    n_harmonics: int = 3,
    period_days: float = 365.25,
) -> np.ndarray:
    """Build design matrix for harmonic regression.

    Columns: [1, cos(1*w*t), sin(1*w*t), ..., cos(K*w*t), sin(K*w*t)]
    where w = 2*pi / period_days and t = dayofyear - 1.

    Parameters
    ----------
    dayofyears : np.ndarray, shape (M,)
    n_harmonics : int, default 3
    period_days : float, default 365.25

    Returns
    -------
    np.ndarray, shape (M, 2 * n_harmonics + 1)
    """
    m = len(dayofyears)
    p = 2 * n_harmonics + 1
    x_mat = np.ones((m, p), dtype=np.float64)

    t_rad = 2.0 * np.pi * (dayofyears - 1.0) / period_days

    col_idx = 1
    for k in range(1, n_harmonics + 1):
        x_mat[:, col_idx] = np.cos(k * t_rad)
        x_mat[:, col_idx + 1] = np.sin(k * t_rad)
        col_idx += 2

    return x_mat


def fit_harmonic_climatology(
    thetao: np.ndarray | xr.DataArray,
    time_coords: np.ndarray | pd.DatetimeIndex,
    n_harmonics: int = 3,
    period_days: float = 365.25,
    min_valid_samples: int = 7,
) -> np.ndarray:
    """Fit ordinary least squares harmonic climatology per cell and depth.

    Parameters
    ----------
    thetao : array-like, shape (n_time, n_depth, n_lat, n_lon)
    time_coords : array-like of datetime64 or pd.Timestamp, shape (n_time,)
    n_harmonics : int, default 3
    period_days : float, default 365.25
    min_valid_samples : int, minimum observations required to fit (default 7)

    Returns
    -------
    np.ndarray, shape (366, n_depth, n_lat, n_lon), float32
        Climatology evaluated at dayofyear 1..366.
    """
    if isinstance(thetao, xr.DataArray):
        t_arr = thetao.values
    else:
        t_arr = np.asarray(thetao)

    n_time, n_depth, n_lat, n_lon = t_arr.shape
    dt_series = pd.to_datetime(time_coords)
    day_of_years = dt_series.day_of_year.values.astype(np.float64)

    # Design matrix for training dates
    x_train = build_harmonic_design_matrix(day_of_years, n_harmonics, period_days)

    # Design matrix for all 366 days of year: (366, p)
    all_doy = np.arange(1, 367, dtype=np.float64)
    x_366 = build_harmonic_design_matrix(all_doy, n_harmonics, period_days)

    # Output climatology: (366, n_depth, n_lat, n_lon)
    clim_out = np.full((366, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    # Flatten spatial/depth dimensions to (n_time, n_cells)
    n_cells = n_depth * n_lat * n_lon
    t_flat = t_arr.reshape((n_time, n_cells)).astype(np.float64)

    # Check for completely non-NaN ocean cells
    valid_mask = ~np.isnan(t_flat)
    all_valid_cells = np.all(valid_mask, axis=0)

    if np.any(all_valid_cells):
        # Fast vectorized path: X_train (n_time, p) -> Beta: (p, n_valid_cells)
        # Beta = (X^T X)^(-1) X^T Y
        xt_x = x_train.T @ x_train
        xt_y = x_train.T @ t_flat[:, all_valid_cells]
        try:
            beta = np.linalg.solve(xt_x, xt_y)  # (p, n_valid_cells)
            # Eval: (366, p) @ (p, n_valid_cells) -> (366, n_valid_cells)
            pred_366 = x_366 @ beta
            # Assign back
            target_slice = clim_out.reshape((366, n_cells))
            target_slice[:, all_valid_cells] = pred_366.astype(np.float32)
        except np.linalg.LinAlgError:
            logger.warning("LinAlgError in vectorized harmonic fit; falling back to lstsq")

    # Handle remaining cells with partial missing values (e.g. ice, boundaries)
    partial_cells = np.where(~all_valid_cells & (np.sum(valid_mask, axis=0) >= min_valid_samples))[
        0
    ]
    for cell_idx in partial_cells:
        c_mask = valid_mask[:, cell_idx]
        x_sub = x_train[c_mask]
        y_sub = t_flat[c_mask, cell_idx]
        try:
            beta_sub, _, _, _ = np.linalg.lstsq(x_sub, y_sub, rcond=None)
            pred_sub = x_366 @ beta_sub
            clim_out.reshape((366, n_cells))[:, cell_idx] = pred_sub.astype(np.float32)
        except Exception:
            pass

    return clim_out


def evaluate_harmonic_climatology(
    clim_366: np.ndarray,
    query_dates: np.ndarray | pd.DatetimeIndex,
) -> np.ndarray:
    """Evaluate precomputed climatology (366, depth, lat, lon) for specific query dates.

    Parameters
    ----------
    clim_366 : np.ndarray, shape (366, n_depth, n_lat, n_lon)
    query_dates : array-like of dates, length N

    Returns
    -------
    np.ndarray, shape (N, n_depth, n_lat, n_lon), float32
    """
    dt_series = pd.to_datetime(query_dates)
    doy_indices = dt_series.day_of_year.values - 1  # 0-indexed into 0..365
    return clim_366[doy_indices].astype(np.float32)


def apply_climatology(
    thetao: np.ndarray,
    clim_366: np.ndarray,
    time_coords: np.ndarray | pd.DatetimeIndex,
) -> np.ndarray:
    """Compute temperature anomalies relative to harmonic climatology.

    Anomaly = thetao - clim(dayofyear).
    """
    clim_eval = evaluate_harmonic_climatology(clim_366, time_coords)
    return (thetao - clim_eval).astype(np.float32)
