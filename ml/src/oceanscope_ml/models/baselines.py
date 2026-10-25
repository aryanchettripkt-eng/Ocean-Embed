"""OceanScope AI non-neural baselines (PRD_AI_ML §4 Stage 3, §5.6, §6.3).

Implements:
1. Climatology baseline: prediction = clim -> baseline_climatology_val.csv
2. Per-pixel Ridge: features = core + forcing + static + DOY + lat/lon -> baseline_ridge_val.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
import xarray as xr

from oceanscope_ml.data.preprocess import ExperimentConfig, load_config
from oceanscope_ml.data.transform import check_no_test_leakage

logger = logging.getLogger("oceanscope_ml.models.baselines")

CANDIDATE_2D_VARS = [
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


def compute_valid_pixel_mask(
    ds: xr.Dataset,
    ds_t: xr.Dataset,
    doy: int,
    feature_2d_vars: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute boolean 2D mask of valid cells where all inputs, climatology, and target are valid.

    Both Climatology and Ridge baselines MUST be evaluated on this exact mask so skill_vs_clim_pct
    is strictly apples-to-apples (PRD_AI_ML §5.6, REVIEW_P3 A7).

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset (for clim if stored globally by DOY).
    ds_t : xr.Dataset
        Single time-step dataset.
    doy : int
        Day of year.
    feature_2d_vars : list of str
        Names of surface 2D feature variables.

    Returns
    -------
    mask : np.ndarray, shape (n_lat, n_lon), bool
        Valid ocean cell mask.
    clim_t : np.ndarray, shape (n_depth, n_lat, n_lon)
        Climatology array for this day.
    """
    lats = ds_t.coords["lat"].values
    lons = ds_t.coords["lon"].values
    ocean_mask = (
        ds_t["ocean_mask"].values.copy()
        if "ocean_mask" in ds_t
        else np.ones((len(lats), len(lons)), dtype=bool)
    )

    if "clim" in ds:
        if "dayofyear" in ds["clim"].coords:
            clim_t = ds["clim"].sel(dayofyear=doy).values
        else:
            clim_t = ds["clim"].values
    else:
        clim_t = np.zeros_like(ds_t["thetao"].values)

    thetao_t = ds_t["thetao"].values

    valid_cell_mask = ocean_mask.copy()
    if "target_valid" in ds_t:
        target_v = ds_t["target_valid"].values
        valid_cell_mask &= np.all(target_v, axis=0)

    for v in feature_2d_vars:
        if v in ds_t:
            valid_cell_mask &= np.isfinite(ds_t[v].values)

    valid_cell_mask &= np.all(np.isfinite(thetao_t), axis=0)
    valid_cell_mask &= np.all(np.isfinite(clim_t), axis=0)

    return valid_cell_mask, clim_t


def extract_day_features_and_targets(
    ds: xr.Dataset,
    ds_t: xr.Dataset,
    doy: int,
    depths_m: list[float],
    feature_2d_vars: list[str],
    lat_norm_grid: np.ndarray,
    lon_norm_grid: np.ndarray,
    has_bathymetry: bool,
    has_woa: bool,
    bathy_vals: np.ndarray | None,
    woa_vals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract features and target anomalies for a single day on valid ocean cells."""
    valid_cell_mask, clim_t = compute_valid_pixel_mask(ds, ds_t, doy, feature_2d_vars)
    y_idx, x_idx = np.where(valid_cell_mask)
    n_cells = len(y_idx)
    if n_cells == 0:
        return None

    day_features: list[np.ndarray] = []
    for v in feature_2d_vars:
        day_features.append(ds_t[v].values[y_idx, x_idx].astype(np.float32))

    if has_bathymetry and bathy_vals is not None:
        day_features.append(bathy_vals[y_idx, x_idx].astype(np.float32))

    if has_woa and woa_vals is not None:
        woa_surface = (
            woa_vals[0, y_idx, x_idx] if woa_vals.ndim == 3 else woa_vals[0, 0, y_idx, x_idx]
        )
        day_features.append(woa_surface.astype(np.float32))

    doy_angle = 2.0 * np.pi * doy / 365.25
    day_features.append(np.full(n_cells, np.sin(doy_angle), dtype=np.float32))
    day_features.append(np.full(n_cells, np.cos(doy_angle), dtype=np.float32))

    day_features.append(lat_norm_grid[y_idx, x_idx].astype(np.float32))
    day_features.append(lon_norm_grid[y_idx, x_idx].astype(np.float32))

    day_X = np.column_stack(day_features)
    thetao_t = ds_t["thetao"].values
    anom_t = thetao_t - clim_t
    day_Y = anom_t[:, y_idx, x_idx].T.astype(np.float32)

    return day_X, day_Y


def extract_pixel_features_and_targets(
    ds: xr.Dataset,
    target_dates: pd.DatetimeIndex,
    depths_m: list[float],
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract 2D pixel features and 15-depth anomaly targets across specified dates.

    Parameters
    ----------
    ds : xr.Dataset
        Processed Zarr dataset (oceanscope_v1.zarr).
    target_dates : pd.DatetimeIndex
        Dates to sample from (e.g. training split).
    depths_m : list of float
        The 15 target GLORYS depth levels.
    max_samples : int, optional
        Maximum random samples to extract (<= 5M to preserve memory).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (N, P)
        Feature matrix.
    Y : np.ndarray, shape (N, 15)
        Target anomaly matrix (thetao - clim).
    feature_names : list of str
        Names of the P features.
    """
    rng = np.random.default_rng(seed)

    feature_2d_vars = [v for v in CANDIDATE_2D_VARS if v in ds.data_vars]
    has_bathymetry = "bathymetry" in ds.data_vars
    has_woa = "woa_clim_profile" in ds.data_vars

    feature_names = list(feature_2d_vars)
    if has_bathymetry:
        feature_names.append("bathymetry")
    if has_woa:
        feature_names.append("woa_surface")
    feature_names.extend(["doy_sin", "doy_cos", "lat_norm", "lon_norm"])

    lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))

    # Grid coordinates normalized to [-1, 1]
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    lat_norm_grid = (
        (lat_2d - lat_min) / (lat_max - lat_min) * 2.0 - 1.0
        if lat_max > lat_min
        else np.zeros_like(lat_2d)
    )
    lon_norm_grid = (
        (lon_2d - lon_min) / (lon_max - lon_min) * 2.0 - 1.0
        if lon_max > lon_min
        else np.zeros_like(lon_2d)
    )

    bathy_vals = ds["bathymetry"].values if has_bathymetry else None
    woa_vals = ds["woa_clim_profile"].values if has_woa else None

    store_times = pd.to_datetime(ds.coords["time"].values)
    valid_dates = [d for d in target_dates if d in store_times]

    if not valid_dates:
        raise ValueError("No matching target dates found in the processed store.")

    all_X: list[np.ndarray] = []
    all_Y: list[np.ndarray] = []

    for d in valid_dates:
        t_str = d.strftime("%Y-%m-%d")
        ds_t = ds.sel(time=t_str)
        doy = int(d.day_of_year)

        day_result = extract_day_features_and_targets(
            ds=ds,
            ds_t=ds_t,
            doy=doy,
            depths_m=depths_m,
            feature_2d_vars=feature_2d_vars,
            lat_norm_grid=lat_norm_grid,
            lon_norm_grid=lon_norm_grid,
            has_bathymetry=has_bathymetry,
            has_woa=has_woa,
            bathy_vals=bathy_vals,
            woa_vals=woa_vals,
        )
        if day_result is not None:
            day_X, day_Y = day_result
            all_X.append(day_X)
            all_Y.append(day_Y)

    if not all_X:
        raise ValueError("No valid ocean cells found across target dates.")

    X_full = np.vstack(all_X)
    Y_full = np.vstack(all_Y)

    if max_samples is not None and len(X_full) > max_samples:
        indices = rng.choice(len(X_full), size=max_samples, replace=False)
        return X_full[indices], Y_full[indices], feature_names

    return X_full, Y_full, feature_names


def fit_ridge_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train per-pixel Ridge regression using closed-form regularized normal equations.

    Solves: min_W ||(X - mu_X)/sigma_X @ W + mu_Y - Y||_2^2 + alpha * ||W||_2^2

    Parameters
    ----------
    X_train : np.ndarray, shape (N, P)
        Input feature matrix.
    Y_train : np.ndarray, shape (N, 15)
        Target subsurface anomalies at 15 depths.
    alpha : float
        Ridge L2 penalty coefficient.

    Returns
    -------
    W : np.ndarray, shape (P, 15)
        Learned regression coefficients.
    mu_Y : np.ndarray, shape (15,)
        Intercept / target offset.
    mu_X : np.ndarray, shape (P,)
        Feature means for standardization.
    sigma_X : np.ndarray, shape (P,)
        Feature stds for standardization.
    """
    mu_X = np.mean(X_train, axis=0)
    sigma_X = np.std(X_train, axis=0)
    sigma_X[sigma_X < 1e-7] = 1.0

    X_norm = (X_train - mu_X) / sigma_X

    mu_Y = np.mean(Y_train, axis=0)
    Y_cent = Y_train - mu_Y

    p = X_norm.shape[1]
    xtx = X_norm.T @ X_norm
    xty = X_norm.T @ Y_cent

    reg_matrix = xtx + alpha * np.eye(p, dtype=np.float32)
    W = scipy.linalg.solve(reg_matrix, xty, assume_a="pos").astype(np.float32)

    return W, mu_Y, mu_X, sigma_X


def evaluate_climatology_baseline(
    ds: xr.Dataset,
    val_dates: pd.DatetimeIndex,
    depths_m: list[float],
) -> pd.DataFrame:
    """Evaluate climatology baseline on validation split (PRD_AI_ML §5.6, §6.3).

    Prediction: T_pred(t, d, y, x) = clim(doy(t), d, y, x).
    Skill vs clim = 0.0% by definition.
    Evaluated on the exact same mask where all 2D features and targets are valid.
    Uses O(1) streaming per-depth error accumulation (REVIEW_P3 A7).
    """
    feature_2d_vars = [v for v in CANDIDATE_2D_VARS if v in ds.data_vars]

    store_times = pd.to_datetime(ds.coords["time"].values)
    valid_dates = [d for d in val_dates if d in store_times]

    if not valid_dates:
        raise ValueError("No matching validation dates found in store.")

    n_depths = len(depths_m)
    n_d = np.zeros(n_depths, dtype=np.int64)
    sum_err = np.zeros(n_depths, dtype=np.float64)
    sum_abs_err = np.zeros(n_depths, dtype=np.float64)
    sum_sq_err = np.zeros(n_depths, dtype=np.float64)

    for d in valid_dates:
        t_str = d.strftime("%Y-%m-%d")
        ds_t = ds.sel(time=t_str)
        doy = int(d.day_of_year)

        mask_t, clim_t = compute_valid_pixel_mask(ds, ds_t, doy, feature_2d_vars)
        y_idx, x_idx = np.where(mask_t)
        n_cells = len(y_idx)
        if n_cells == 0:
            continue

        thetao_t = ds_t["thetao"].values  # (depth, lat, lon)
        clim_cells = clim_t[:, y_idx, x_idx]  # (depth, n_cells)
        thetao_cells = thetao_t[:, y_idx, x_idx]  # (depth, n_cells)

        err = clim_cells - thetao_cells  # (depth, n_cells)

        for d_idx in range(n_depths):
            e_d = err[d_idx]
            n_d[d_idx] += n_cells
            sum_err[d_idx] += float(np.sum(e_d))
            sum_abs_err[d_idx] += float(np.sum(np.abs(e_d)))
            sum_sq_err[d_idx] += float(np.sum(e_d**2))

    rows: list[dict[str, Any]] = []
    for d_idx, d_m in enumerate(depths_m):
        c_n = int(n_d[d_idx])
        if c_n > 0:
            mae_d = float(sum_abs_err[d_idx] / c_n)
            rmse_d = float(np.sqrt(sum_sq_err[d_idx] / c_n))
            bias_d = float(sum_err[d_idx] / c_n)
        else:
            mae_d, rmse_d, bias_d = np.nan, np.nan, np.nan

        rows.append(
            {
                "model": "climatology",
                "split": "val",
                "depth_m": d_m,
                "n": c_n,
                "mae_c": round(mae_d, 3) if np.isfinite(mae_d) else np.nan,
                "rmse_c": round(rmse_d, 3) if np.isfinite(rmse_d) else np.nan,
                "bias_c": round(bias_d, 3) if np.isfinite(bias_d) else np.nan,
                "skill_vs_clim_pct": 0.0,
            }
        )

    total_n = int(np.sum(n_d))
    if total_n > 0:
        total_mae = float(np.sum(sum_abs_err) / total_n)
        total_rmse = float(np.sqrt(np.sum(sum_sq_err) / total_n))
        total_bias = float(np.sum(sum_err) / total_n)
    else:
        total_mae, total_rmse, total_bias = np.nan, np.nan, np.nan

    rows.append(
        {
            "model": "climatology",
            "split": "val",
            "depth_m": "all",
            "n": total_n,
            "mae_c": round(total_mae, 3) if np.isfinite(total_mae) else np.nan,
            "rmse_c": round(total_rmse, 3) if np.isfinite(total_rmse) else np.nan,
            "bias_c": round(total_bias, 3) if np.isfinite(total_bias) else np.nan,
            "skill_vs_clim_pct": 0.0,
        }
    )

    return pd.DataFrame(rows)


def evaluate_ridge_baseline(
    ds: xr.Dataset,
    val_dates: pd.DatetimeIndex,
    depths_m: list[float],
    W: np.ndarray,
    mu_Y: np.ndarray,
    mu_X: np.ndarray,
    sigma_X: np.ndarray,
    clim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate trained Ridge baseline on validation split (PRD_AI_ML §5.6, §6.3).

    Prediction: T_pred(t, d, y, x) = clim(doy(t), d, y, x) + Y_pred(t, d, y, x).
    Skill vs clim = 100 * (1 - RMSE_ridge / RMSE_clim).
    Evaluated day-by-day on the exact same mask as Climatology with O(1) streaming accumulators.
    """
    feature_2d_vars = [v for v in CANDIDATE_2D_VARS if v in ds.data_vars]
    has_bathymetry = "bathymetry" in ds.data_vars
    has_woa = "woa_clim_profile" in ds.data_vars

    lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))

    lon_2d, lat_2d = np.meshgrid(lons, lats)
    lat_norm_grid = (
        (lat_2d - lat_min) / (lat_max - lat_min) * 2.0 - 1.0
        if lat_max > lat_min
        else np.zeros_like(lat_2d)
    )
    lon_norm_grid = (
        (lon_2d - lon_min) / (lon_max - lon_min) * 2.0 - 1.0
        if lon_max > lon_min
        else np.zeros_like(lon_2d)
    )

    bathy_vals = ds["bathymetry"].values if has_bathymetry else None
    woa_vals = ds["woa_clim_profile"].values if has_woa else None

    store_times = pd.to_datetime(ds.coords["time"].values)
    valid_dates = [d for d in val_dates if d in store_times]

    if not valid_dates:
        raise ValueError("No matching validation dates found in store.")

    n_depths = len(depths_m)
    n_d = np.zeros(n_depths, dtype=np.int64)
    sum_err = np.zeros(n_depths, dtype=np.float64)
    sum_abs_err = np.zeros(n_depths, dtype=np.float64)
    sum_sq_err = np.zeros(n_depths, dtype=np.float64)

    # Pre-index climatology RMSE per depth for skill calculation
    clim_rmse_map = {}
    for _, row in clim_df.iterrows():
        clim_rmse_map[row["depth_m"]] = float(row["rmse_c"])

    for d in valid_dates:
        t_str = d.strftime("%Y-%m-%d")
        ds_t = ds.sel(time=t_str)
        doy = int(d.day_of_year)

        day_result = extract_day_features_and_targets(
            ds=ds,
            ds_t=ds_t,
            doy=doy,
            depths_m=depths_m,
            feature_2d_vars=feature_2d_vars,
            lat_norm_grid=lat_norm_grid,
            lon_norm_grid=lon_norm_grid,
            has_bathymetry=has_bathymetry,
            has_woa=has_woa,
            bathy_vals=bathy_vals,
            woa_vals=woa_vals,
        )
        if day_result is None:
            continue

        day_X, day_Y_true = day_result
        n_cells = len(day_X)
        if n_cells == 0:
            continue

        day_X_norm = (day_X - mu_X) / sigma_X
        day_Y_pred = day_X_norm @ W + mu_Y  # (n_cells, 15)

        # Anomaly error equals total temperature error since clim cancels out:
        # Error = T_pred - T_true = (clim + Y_pred) - (clim + Y_true) = Y_pred - Y_true
        day_err = day_Y_pred - day_Y_true  # (n_cells, 15)

        for d_idx in range(n_depths):
            e_d = day_err[:, d_idx]
            n_d[d_idx] += n_cells
            sum_err[d_idx] += float(np.sum(e_d))
            sum_abs_err[d_idx] += float(np.sum(np.abs(e_d)))
            sum_sq_err[d_idx] += float(np.sum(e_d**2))

    rows: list[dict[str, Any]] = []
    for d_idx, d_m in enumerate(depths_m):
        c_n = int(n_d[d_idx])
        if c_n > 0:
            mae_d = float(sum_abs_err[d_idx] / c_n)
            rmse_d = float(np.sqrt(sum_sq_err[d_idx] / c_n))
            bias_d = float(sum_err[d_idx] / c_n)
            c_rmse = clim_rmse_map.get(d_m, np.nan)
            skill_d = (
                round((1.0 - (rmse_d / c_rmse)) * 100.0, 2)
                if np.isfinite(c_rmse) and c_rmse > 1e-6 and np.isfinite(rmse_d)
                else np.nan
            )
        else:
            mae_d, rmse_d, bias_d, skill_d = np.nan, np.nan, np.nan, np.nan

        rows.append(
            {
                "model": "ridge",
                "split": "val",
                "depth_m": d_m,
                "n": c_n,
                "mae_c": round(mae_d, 3) if np.isfinite(mae_d) else np.nan,
                "rmse_c": round(rmse_d, 3) if np.isfinite(rmse_d) else np.nan,
                "bias_c": round(bias_d, 3) if np.isfinite(bias_d) else np.nan,
                "skill_vs_clim_pct": skill_d,
            }
        )

    total_n = int(np.sum(n_d))
    if total_n > 0:
        total_mae = float(np.sum(sum_abs_err) / total_n)
        total_rmse = float(np.sqrt(np.sum(sum_sq_err) / total_n))
        total_bias = float(np.sum(sum_err) / total_n)
        c_all_rmse = clim_rmse_map.get("all", np.nan)
        skill_all = (
            round((1.0 - (total_rmse / c_all_rmse)) * 100.0, 2)
            if np.isfinite(c_all_rmse) and c_all_rmse > 1e-6 and np.isfinite(total_rmse)
            else np.nan
        )
    else:
        total_mae, total_rmse, total_bias, skill_all = np.nan, np.nan, np.nan, np.nan

    rows.append(
        {
            "model": "ridge",
            "split": "val",
            "depth_m": "all",
            "n": total_n,
            "mae_c": round(total_mae, 3) if np.isfinite(total_mae) else np.nan,
            "rmse_c": round(total_rmse, 3) if np.isfinite(total_rmse) else np.nan,
            "bias_c": round(total_bias, 3) if np.isfinite(total_bias) else np.nan,
            "skill_vs_clim_pct": skill_all,
        }
    )

    return pd.DataFrame(rows)


def run_baselines(
    config: ExperimentConfig,
    store_path: Path | None = None,
    out_dir: Path | None = None,
    models_dir: Path | None = None,
    max_train_samples: int | None = None,
    ridge_alpha: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate non-neural baselines (Climatology and Ridge).

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    store_path : Path, optional
        Path to processed Zarr store (default: data/processed/oceanscope_<version>.zarr).
    out_dir : Path, optional
        Metrics output directory (default: outputs/metrics).
    models_dir : Path, optional
        Model checkpoint directory (default: checkpoints).
    max_train_samples : int, optional
        Max random training samples to fit Ridge regression (<= 5M per PRD §5.6).
        Defaults to config.baselines.max_train_samples.
    ridge_alpha : float, optional
        Regularization parameter for Ridge.
        Defaults to config.baselines.ridge_alpha.

    Returns
    -------
    df_clim : pd.DataFrame
        Climatology validation metrics table.
    df_ridge : pd.DataFrame
        Ridge validation metrics table.
    """
    if store_path is None:
        store_path = Path(config.paths.processed) / f"oceanscope_{config.dataset_version}.zarr"
    if out_dir is None:
        out_dir = Path(config.paths.outputs) / "metrics"
    if models_dir is None:
        models_dir = Path(config.paths.checkpoints)

    base_cfg = getattr(config, "baselines", None)
    if max_train_samples is None:
        max_train_samples = int(getattr(base_cfg, "max_train_samples", 500_000))
    if ridge_alpha is None:
        ridge_alpha = float(getattr(base_cfg, "ridge_alpha", 1.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening processed Zarr store from %s...", store_path)
    ds = xr.open_zarr(store_path)

    # Date splits
    train_split = config.splits.train
    val_split = config.splits.val
    test_split = config.splits.test

    logger.info("Configured splits: Train=%s, Val=%s, Test=%s", train_split, val_split, test_split)

    # Validate no test leakage (PRD_AI_ML §0 Hard Rule 2)
    check_no_test_leakage(train_split, test_split)
    check_no_test_leakage(val_split, test_split)

    store_dates = pd.to_datetime(ds.coords["time"].values)

    # 1. Filter training dates
    train_dates = store_dates[
        (store_dates >= pd.Timestamp(train_split[0]))
        & (store_dates <= pd.Timestamp(train_split[1]))
    ]
    # 2. Filter validation dates
    val_dates = store_dates[
        (store_dates >= pd.Timestamp(val_split[0])) & (store_dates <= pd.Timestamp(val_split[1]))
    ]

    # Strictly enforce splits: no silent 70/30 fallback (PRD_AI_ML §0 Hard Rule 1, REVIEW_P3 A7)
    if len(train_dates) == 0:
        raise ValueError(
            f"No training dates found in processed store matching train split {train_split}."
        )
    if len(val_dates) == 0:
        raise ValueError(
            f"No validation dates found in processed store matching val split {val_split}."
        )

    check_no_test_leakage(train_dates, test_split)
    check_no_test_leakage(val_dates, test_split)

    depths_m = [float(d) for d in config.depths_m]

    # --- 1. Climatology Baseline ---
    logger.info("Computing Climatology baseline on %d validation dates...", len(val_dates))
    df_clim = evaluate_climatology_baseline(ds, val_dates, depths_m)
    clim_path = out_dir / "baseline_climatology_val.csv"
    df_clim.to_csv(clim_path, index=False)
    logger.info("Climatology baseline metrics written to %s", clim_path)

    # --- 2. Per-pixel Ridge Regression Baseline ---
    logger.info(
        "Extracting training samples from %d train dates (max_samples=%d)...",
        len(train_dates),
        max_train_samples,
    )
    X_train, Y_train, feature_names = extract_pixel_features_and_targets(
        ds=ds,
        target_dates=train_dates,
        depths_m=depths_m,
        max_samples=max_train_samples,
        seed=(
            int(config.train.get("seed", 42))
            if isinstance(config.train, dict)
            else int(getattr(config.train, "seed", 42))
        ),
    )

    logger.info(
        "Training Ridge regression on %d pixels with %d features (alpha=%.2f)...",
        len(X_train),
        len(feature_names),
        ridge_alpha,
    )
    W, mu_Y, mu_X, sigma_X = fit_ridge_baseline(X_train, Y_train, alpha=ridge_alpha)

    # Save model artifacts
    model_path = models_dir / "baseline_ridge.npz"
    np.savez_compressed(
        model_path,
        weights=W,
        intercept=mu_Y,
        mu_X=mu_X,
        sigma_X=sigma_X,
        feature_names=np.array(feature_names),
        alpha=ridge_alpha,
    )
    logger.info("Saved Ridge baseline model artifact to %s", model_path)

    logger.info("Evaluating Ridge baseline on validation set...")
    df_ridge = evaluate_ridge_baseline(
        ds=ds,
        val_dates=val_dates,
        depths_m=depths_m,
        W=W,
        mu_Y=mu_Y,
        mu_X=mu_X,
        sigma_X=sigma_X,
        clim_df=df_clim,
    )
    ridge_path = out_dir / "baseline_ridge_val.csv"
    df_ridge.to_csv(ridge_path, index=False)
    logger.info("Ridge baseline metrics written to %s", ridge_path)

    return df_clim, df_ridge


def build_cli() -> argparse.ArgumentParser:
    """Build command-line parser for oceanscope_ml.models.baselines."""
    parser = argparse.ArgumentParser(
        description="Compute OceanScope AI non-neural baselines (Climatology and Ridge, PRD §5.6)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Path to processed Zarr store (default: data/processed/oceanscope_<v>.zarr).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write validation metrics CSVs (default: outputs/metrics).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to save baseline model weights (default: checkpoints).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max random training samples for Ridge regression (default: from config, <= 5M).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Ridge regularization penalty alpha (default: from config).",
    )
    return parser


def main() -> None:
    """CLI entrypoint for oceanscope_ml.models.baselines."""
    parser = build_cli()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    df_clim, df_ridge = run_baselines(
        config=config,
        store_path=args.store,
        out_dir=args.out_dir,
        models_dir=args.models_dir,
        max_train_samples=args.max_samples,
        ridge_alpha=args.alpha,
    )

    print("\n" + "=" * 80)
    print("  OCEANSCOPE AI - VALIDATION BASELINES COMPARISON (PRD_AI_ML §5.6)")
    print("=" * 80)
    print(
        f"{'Depth (m)':<12} {'Clim RMSE (°C)':<16} {'Ridge RMSE (°C)':<16} {'Ridge Skill (%)':<18}"
    )
    print("-" * 80)

    for d_idx, row_clim in df_clim.iterrows():
        row_ridge = df_ridge.iloc[d_idx]
        d_val = str(row_clim["depth_m"])
        c_rmse = float(row_clim["rmse_c"])
        r_rmse = float(row_ridge["rmse_c"])
        r_skill = float(row_ridge["skill_vs_clim_pct"])
        print(f"{d_val:<12} {c_rmse:<16.3f} {r_rmse:<16.3f} {r_skill:<24.1f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
