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
    train_split: list[str] | None = None,
    test_split: list[str] | None = None,
) -> np.ndarray:
    """Fit ordinary least squares harmonic climatology per cell and depth.

    Must only use training period dates (PRD_AI_ML §0 rule 4).

    Parameters
    ----------
    thetao : array-like, shape (n_time, n_depth, n_lat, n_lon)
    time_coords : array-like of datetime64 or pd.Timestamp, shape (n_time,)
    n_harmonics : int, default 3
    period_days : float, default 365.25
    min_valid_samples : int, minimum observations required to fit (default 7)
    train_split : [train_start, train_end] (optional guard)
    test_split : [test_start, test_end] (optional guard against test leakage)

    Returns
    -------
    np.ndarray, shape (366, n_depth, n_lat, n_lon), float32
        Climatology evaluated at dayofyear 1..366.
    """
    dt_series = pd.to_datetime(time_coords)

    # Test leakage guard (PRD §0 rule 3, rule 4)
    if test_split is not None and len(test_split) == 2:
        test_start = pd.Timestamp(test_split[0])
        test_end = pd.Timestamp(test_split[1])
        leak_mask = (dt_series >= test_start) & (dt_series <= test_end)
        if np.any(leak_mask):
            first_leaked = dt_series[leak_mask][0].strftime("%Y-%m-%d")
            raise ValueError(
                f"Test leakage detected in climatology fit! Dates fall inside sealed test split "
                f"[{test_start.strftime('%Y-%m-%d')} .. {test_end.strftime('%Y-%m-%d')}]. "
                f"First leaked date: {first_leaked}."
            )

    if train_split is not None and len(train_split) == 2:
        train_start = pd.Timestamp(train_split[0])
        train_end = pd.Timestamp(train_split[1])
        train_mask = (dt_series >= train_start) & (dt_series <= train_end)
        if not np.any(train_mask):
            raise ValueError(
                f"No dates found in training period [{train_start} .. {train_end}] for climatology."
            )
        if not np.all(train_mask):
            # Select only train dates
            dt_series = dt_series[train_mask]
            if isinstance(thetao, xr.DataArray):
                thetao = thetao.isel(time=np.where(train_mask)[0])
            else:
                thetao = np.asarray(thetao)[train_mask]
    n_time = len(dt_series)
    if isinstance(thetao, xr.DataArray):
        n_depth, n_lat, n_lon = thetao.shape[1], thetao.shape[2], thetao.shape[3]
    else:
        n_depth, n_lat, n_lon = thetao.shape[1], thetao.shape[2], thetao.shape[3]

    day_of_years = dt_series.day_of_year.values.astype(np.float64)

    # Design matrix for training dates: (n_time, p)
    x_train = build_harmonic_design_matrix(day_of_years, n_harmonics, period_days)
    p = x_train.shape[1]

    # Design matrix for all 366 days of year: (366, p)
    all_doy = np.arange(1, 367, dtype=np.float64)
    x_366 = build_harmonic_design_matrix(all_doy, n_harmonics, period_days)

    # Output climatology: (366, n_depth, n_lat, n_lon)
    clim_out = np.full((366, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
    n_cells = n_depth * n_lat * n_lon

    # Streaming normal equations per cell, so memory does not grow with n_time
    # (PRD §4.2, REVIEW_P3 R3). A cell's X^T X only sums the rows where that cell is
    # finite, which is the same fit as lstsq on its valid rows. Rows of
    # x_outer = x_t x_t^T flattened, so valid.T @ x_outer gives every cell's X^T X in
    # one matrix product.
    x_outer = np.einsum("tp,tq->tpq", x_train, x_train).reshape(n_time, p * p)
    xt_x_cell = np.zeros((n_cells, p * p), dtype=np.float64)
    xt_y = np.zeros((p, n_cells), dtype=np.float64)
    n_valid_per_cell = np.zeros(n_cells, dtype=np.int32)

    chunk_size = 64
    for start_i in range(0, n_time, chunk_size):
        end_i = min(start_i + chunk_size, n_time)
        x_chunk = x_train[start_i:end_i]

        if isinstance(thetao, xr.DataArray):
            t_chunk = thetao.isel(time=slice(start_i, end_i)).values
        else:
            t_chunk = np.asarray(thetao[start_i:end_i])

        t_chunk_flat = t_chunk.reshape((t_chunk.shape[0], n_cells)).astype(np.float64)
        valid_chunk = np.isfinite(t_chunk_flat)
        n_valid_per_cell += np.sum(valid_chunk, axis=0).astype(np.int32)

        t_chunk_clean = np.where(valid_chunk, t_chunk_flat, 0.0)
        xt_y += x_chunk.T @ t_chunk_clean
        xt_x_cell += valid_chunk.T.astype(np.float64) @ x_outer[start_i:end_i]

    target_slice = clim_out.reshape((366, n_cells))

    # Cells with full valid observations across all training dates share one X^T X
    full_valid_cells = np.where(n_valid_per_cell == n_time)[0]
    if len(full_valid_cells) > 0:
        xt_x = x_train.T @ x_train
        beta = np.linalg.lstsq(xt_x, xt_y[:, full_valid_cells], rcond=None)[0]
        target_slice[:, full_valid_cells] = (x_366 @ beta).astype(np.float32)

    # Cells with some missing values: solve their own normal equations in batches
    partial_cells = np.where((n_valid_per_cell < n_time) & (n_valid_per_cell >= min_valid_samples))[
        0
    ]
    batch = 20_000
    for b0 in range(0, len(partial_cells), batch):
        cells = partial_cells[b0 : b0 + batch]
        a_mat = xt_x_cell[cells].reshape(len(cells), p, p)
        rhs = xt_y[:, cells].T[:, :, None]
        beta_b = (np.linalg.pinv(a_mat) @ rhs)[:, :, 0]  # (n_batch, p)
        target_slice[:, cells] = (x_366 @ beta_b.T).astype(np.float32)

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
