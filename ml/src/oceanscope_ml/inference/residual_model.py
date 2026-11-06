"""Conditional ARGO residual correction: a small gradient-boosted tree model of the model's error.

Usage (from ml/):

    python -m oceanscope_ml.inference.residual_model --config configs/experiment.yaml \
        --predictions outputs/predictions/predictions_<run_id>.zarr \
        --store <part stores...> [--stats stats.zarr] \
        --argo <argo_YYYY-MM.nc...> --holdout configs/argo_eval_floats.json \
        --run-id <run_id>_rm [--out-dir outputs]

``inference.argo_correct`` spreads the floats in the water on day t around them. This step
learns the model's *systematic* error instead: where, when and in which conditions it is too
warm or too cold. For every profile of a non-held-out float in the validation period the
residual r = T_argo - T_pred is taken on the 15 model levels (``argo_correct.residuals``), and a
LightGBM regressor learns r from:

- depth, the predicted temperature at that level, latitude, longitude, day-of-year (sin, cos)
- SST, SLA and wind speed at the cell on day t (from the processed store), when present
- great-circle distance (deg) and age (days) of the nearest *other* float's profile in the
  10 days up to t (in training the float's own profiles are left out, as a held-out float's
  are at inference)

Five-fold cross-validation by float (the folds never split a float) picks the number of trees
and reports the out-of-fold RMSE with and without the correction, overall and in the Bay of
Bengal at 50-300 m. The final model is fit on all non-held-out floats and its prediction is
added to ``temp_pred`` on every valid cell of every validation day. Held-out floats
(PRD_AI_ML §7) are never read; only validation dates are read or written.

``--features v2`` adds, aimed at the two weak spots the regime table found (warm anomalies and
the SW monsoon in the Bay of Bengal):

- ``near_res_c`` and ``near_wsum``: the weighted mean of the *other* floats' residuals at the
  same level in the 10 days up to t (Gaussian in great-circle distance, L = 1.5 deg, and
  exponential in age, tau = 5 days), and the sum of those weights. The tree learns how far to
  trust the nearby floats in each condition, where the OI step trusts them the same everywhere.
- ``pred_anom_c``: the predicted temperature minus the train-only climatology (needs
  ``--stats``), so the tree can see when the model calls a warm or cold anomaly.
- ``pred_dtdz`` (C/m) and ``pred_mld_m`` (first level 0.5 C colder than the ~10 m level):
  the shape of the model's own profile (thermocline, mixed layer). From the prediction only,
  never from GLORYS.
- ``sst_d14`` and ``sla_d28``: SST change over 14 days and SLA change over 28 days at the cell
  (NaN when the earlier day is not in the store).

``--max-depth M``: levels deeper than M metres are neither trained on nor corrected (on the
final half the correction made 643 m and 1062 m slightly worse).

Writes ``<out>/predictions/predictions_<run_id>.zarr`` and
``<out>/metrics/residual_model_cv_<run_id>.csv``. Every input is available in live mode too.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.spatial import cKDTree

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.argo import profile_levels, read_levels
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import check_no_test_leakage
from oceanscope_ml.evaluation.evaluate import BAY_OF_BENGAL_MIN_LON, EQUATORIAL_MAX_LAT
from oceanscope_ml.inference.argo_correct import Residuals, _day_number, residuals

logger = logging.getLogger("oceanscope_ml.inference.residual_model")

WINDOW_DAYS = 10
N_FOLDS = 5
FOLD_SEED = 42
BLOCK_DAYS = 16
FAR_DEG = 10.0  # distance feature when no other profile is in the window
SURFACE_VARS = ("sst", "sla", "u_wind", "v_wind")
FEATURE_SETS = ("v1", "v2")
NEAR_L_DEG = 1.5
NEAR_TAU_DAYS = 5.0
NEAR_MIN_WEIGHT = 1e-3  # below this the nearby-float residual is NaN (no float close enough)
MLD_REF_M = 10.0
MLD_DT_C = 0.5
LAGS_DAYS = {"sst": 14, "sla": 28}
POINT_BLOCK = 4096
PARAMS: dict[str, Any] = {
    "objective": "huber",
    "alpha": 1.0,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}
MAX_ROUNDS = 2000
EARLY_STOP = 50
CV_COLUMNS = ["group", "n_obs", "rmse_none_c", "rmse_corrected_c", "n_trees"]


def _surface_fields(surface: xr.Dataset | None, day: pd.Timestamp) -> dict[str, np.ndarray]:
    """[lat, lon] arrays of the surface features on ``day`` (empty without a surface store)."""
    if surface is None:
        return {}
    field = surface[[v for v in SURFACE_VARS if v in surface]].sel(time=day).load()
    out: dict[str, np.ndarray] = {}
    if "sst" in field:
        out["sst"] = np.asarray(field["sst"].values, dtype=float)
    if "sla" in field:
        out["sla"] = np.asarray(field["sla"].values, dtype=float)
    if "u_wind" in field and "v_wind" in field:
        u = np.asarray(field["u_wind"].values, dtype=float)
        v = np.asarray(field["v_wind"].values, dtype=float)
        out["wind"] = np.hypot(u, v)
    return out


def _unit(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _chord_deg(chord: np.ndarray) -> np.ndarray:
    return np.rad2deg(2 * np.arcsin(np.clip(chord / 2, 0, 1)))


def _nearest_other(
    res: Residuals, day: int, lat: np.ndarray, lon: np.ndarray, own: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance (deg) and age (days) of the nearest profile in the window
    up to ``day``; ``own`` (one float id per point) leaves each point's own float out."""
    sel = (res.day <= day) & (res.day > day - WINDOW_DAYS)
    dist = np.full(lat.size, FAR_DEG)
    age = np.full(lat.size, float(WINDOW_DAYS))
    if not sel.any() or lat.size == 0:
        return dist, age
    r_lat, r_lon = res.lat[sel], res.lon[sel]
    q, r = _unit(lat, lon), _unit(r_lat, r_lon)
    r_age = (day - res.day[sel]).astype(float)
    r_plat = res.platform[sel]
    if own is None:
        d, k = cKDTree(r).query(q)
        d = _chord_deg(d)
        return np.minimum(d, FAR_DEG), np.where(d < FAR_DEG, r_age[k], age)
    for p in np.unique(own):
        pts = own == p
        other = r_plat != p
        if not other.any():
            continue
        d, k = cKDTree(r[other]).query(q[pts])
        d = _chord_deg(d)
        dist[pts] = np.minimum(d, FAR_DEG)
        age[pts] = np.where(d < FAR_DEG, r_age[other][k], WINDOW_DAYS)
    return dist, age


def _surface_lagged(surface: xr.Dataset | None, day: pd.Timestamp) -> dict[str, np.ndarray]:
    """[lat, lon] change of SST over 14 days and SLA over 28 days (NaN where a day is missing)."""
    out: dict[str, np.ndarray] = {}
    if surface is None:
        return out
    times = pd.DatetimeIndex(surface["time"].values).normalize()
    for var, lag in LAGS_DAYS.items():
        if var not in surface:
            continue
        then = day - pd.Timedelta(days=lag)
        if day in times and then in times:
            now_v = np.asarray(surface[var].sel(time=day).values, dtype=float)
            then_v = np.asarray(surface[var].sel(time=then).values, dtype=float)
            out[f"{var}_d{lag}"] = now_v - then_v
        else:
            shape = (surface.sizes["lat"], surface.sizes["lon"])
            out[f"{var}_d{lag}"] = np.full(shape, np.nan)
    return out


def _clim_at(surface: xr.Dataset | None, day: pd.Timestamp) -> np.ndarray | None:
    """[depth, lat, lon] train-only climatology on ``day`` (None without a stats store)."""
    if surface is None or "clim" not in surface:
        return None
    clim = surface["clim"].isel(dayofyear=day.day_of_year - 1)
    return np.asarray(clim.transpose("depth", "lat", "lon").values, dtype=float)


def _near_residual(
    res: Residuals, day: int, lat: np.ndarray, lon: np.ndarray, own: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """[m, D] weighted mean of the residuals of the profiles in the window up to ``day`` and
    [m, D] sum of their weights. ``own`` (one float id per point) leaves each point's own float
    out, as a held-out float's profiles are left out at inference."""
    n_depth = res.values.shape[1]
    val = np.full((lat.size, n_depth), np.nan)
    wsum = np.zeros((lat.size, n_depth))
    sel = (res.day <= day) & (res.day > day - WINDOW_DAYS)
    if not sel.any() or lat.size == 0:
        return val, wsum
    r = _unit(res.lat[sel], res.lon[sel])
    decay = np.exp(-(day - res.day[sel]).astype(float) / NEAR_TAU_DAYS)
    r_val = res.values[sel]
    finite = np.isfinite(r_val)
    r_zero = np.where(finite, r_val, 0.0)
    r_plat = res.platform[sel]
    q_all = _unit(lat, lon)
    for b0 in range(0, lat.size, POINT_BLOCK):
        b1 = min(b0 + POINT_BLOCK, lat.size)
        d = np.rad2deg(np.arccos(np.clip(q_all[b0:b1] @ r.T, -1.0, 1.0)))
        w = np.exp(-(d**2) / (2.0 * NEAR_L_DEG**2)) * decay[None]
        if own is not None:
            w[own[b0:b1, None] == r_plat[None, :]] = 0.0
        num = w @ r_zero
        den = w @ finite.astype(float)
        wsum[b0:b1] = den
        val[b0:b1] = np.where(den > NEAR_MIN_WEIGHT, num / np.maximum(den, 1e-12), np.nan)
    return val, wsum


def _profile_shape(depths: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[m, D] vertical gradient (C/m) and [m] mixed-layer depth of the predicted profiles."""
    if depths.size > 1:
        dtdz = np.gradient(pred, depths, axis=1)
    else:
        dtdz = np.zeros_like(pred)
    ref = int(np.abs(depths - MLD_REF_M).argmin())
    colder = pred < (pred[:, ref] - MLD_DT_C)[:, None]
    colder[:, : ref + 1] = False
    has = colder.any(axis=1)
    mld = np.where(has, depths[colder.argmax(axis=1)], depths[-1])
    mld = np.where(np.isfinite(pred[:, ref]), mld, np.nan)
    return dtdz, mld


def _v2_columns(
    depths: np.ndarray,
    pred: np.ndarray,
    clim: np.ndarray | None,
    lagged: dict[str, np.ndarray],
    cells: tuple[np.ndarray, np.ndarray],
    near: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """The v2 feature columns (point-major, depth-minor) for m points."""
    n_depth = depths.size
    dtdz, mld = _profile_shape(depths, pred)
    cols: dict[str, np.ndarray] = {
        "near_res_c": near[0].reshape(-1),
        "near_wsum": near[1].reshape(-1),
        "pred_dtdz": dtdz.reshape(-1),
        "pred_mld_m": np.repeat(mld, n_depth),
    }
    if clim is not None:
        cols["pred_anom_c"] = (pred - clim[:, cells[0], cells[1]].T).reshape(-1)
    for name, grid in lagged.items():
        cols[name] = np.repeat(grid[cells[0], cells[1]], n_depth)
    return cols


def _features(
    depths: np.ndarray,
    pred: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    day: pd.Timestamp,
    surface: dict[str, np.ndarray],
    cells: tuple[np.ndarray, np.ndarray],
    dist: np.ndarray,
    age: np.ndarray,
    names: list[str],
    extra: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """[m * D, F] feature rows for m points (point-major, depth-minor)."""
    m, n_depth = pred.shape
    doy = 2 * np.pi * (day.day_of_year - 1) / 365.25
    cols: dict[str, np.ndarray] = {
        "depth_m": np.tile(depths, m),
        "pred_c": pred.reshape(-1),
        "lat": np.repeat(lat, n_depth),
        "lon": np.repeat(lon, n_depth),
        "doy_sin": np.full(m * n_depth, np.sin(doy)),
        "doy_cos": np.full(m * n_depth, np.cos(doy)),
        "near_dist_deg": np.repeat(dist, n_depth),
        "near_age_days": np.repeat(age, n_depth),
    }
    for name, grid in surface.items():
        cols[name] = np.repeat(grid[cells[0], cells[1]], n_depth)
    cols.update(extra or {})
    return np.column_stack([cols[n] for n in names]).astype(np.float32)


def feature_names(surface_names: list[str], extra_names: list[str] | None = None) -> list[str]:
    base = ["depth_m", "pred_c", "lat", "lon", "doy_sin", "doy_cos"]
    return base + surface_names + ["near_dist_deg", "near_age_days"] + list(extra_names or [])


def _day_features(
    features: str,
    res: Residuals,
    day_num: int,
    day: pd.Timestamp,
    depths: np.ndarray,
    model: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cells: tuple[np.ndarray, np.ndarray],
    surface: xr.Dataset | None,
    own: np.ndarray | None,
    names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Feature rows for m points on one day; the names are fixed by the first call."""
    surf = _surface_fields(surface, day)
    dist, age = _nearest_other(res, day_num, lat, lon, own)
    extra: dict[str, np.ndarray] = {}
    if features == "v2":
        near = _near_residual(res, day_num, lat, lon, own)
        extra = _v2_columns(depths, model, _clim_at(surface, day), _surface_lagged(surface, day),
                            cells, near)  # fmt: skip
    if names is None:
        names = feature_names(list(surf), list(extra))
    return _features(depths, model, lat, lon, day, surf, cells, dist, age, names, extra), names


def training_rows(
    res: Residuals,
    pred: xr.Dataset,
    surface: xr.Dataset | None,
    features: str = "v1",
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Feature matrix X, target r, float id and Bay-of-Bengal-band flag for every residual
    level, plus the feature names."""
    depths = np.asarray(pred["depth"].values, dtype=float)
    lat_grid = np.asarray(pred["lat"].values, dtype=float)
    lon_grid = np.asarray(pred["lon"].values, dtype=float)
    pred_days = _day_number(pred["time"].values)
    pos = {int(d): k for k, d in enumerate(pred_days)}
    i_lat = np.abs(res.lat[:, None] - lat_grid[None, :]).argmin(axis=1)
    i_lon = np.abs(res.lon[:, None] - lon_grid[None, :]).argmin(axis=1)
    xs, ys, gs, bs = [], [], [], []
    names: list[str] | None = None
    for d in np.unique(res.day):
        idx = np.flatnonzero(res.day == d)
        t = pos[int(d)]
        day = pd.Timestamp(pred["time"].values[t]).normalize()
        temp = np.asarray(pred["temp_pred"].isel(time=t).transpose("depth", "lat", "lon").values)
        model = temp[:, i_lat[idx], i_lon[idx]].T.astype(float)
        x, names = _day_features(features, res, int(d), day, depths, model, res.lat[idx],
                                 res.lon[idx], (i_lat[idx], i_lon[idx]), surface,
                                 res.platform[idx], names)  # fmt: skip
        y = res.values[idx].reshape(-1)
        band = ((res.lon[idx] >= BAY_OF_BENGAL_MIN_LON) & (res.lat[idx] >= EQUATORIAL_MAX_LAT))[
            :, None
        ] & ((depths >= 50) & (depths <= 300))[None, :]
        xs.append(x)
        ys.append(y)
        gs.append(np.repeat(res.platform[idx], depths.size))
        bs.append(band.reshape(-1))
    if names is None:
        raise ValueError("No residuals to train on")
    x, y, g, b = (np.concatenate(a) for a in (xs, ys, gs, bs))
    ok = np.isfinite(y) & np.isfinite(x[:, names.index("pred_c")])
    if max_depth_m is not None:
        ok &= x[:, names.index("depth_m")] <= max_depth_m
    return x[ok], y[ok], g[ok], b[ok], names


def _folds(groups: np.ndarray, n: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    floats = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(floats[rng.permutation(floats.size)], np.arange(floats.size) % n))
    return np.array([fold_of[g] for g in groups])


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def fit(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, band: np.ndarray, names: list[str]
) -> tuple[Any, pd.DataFrame]:
    """Cross-validate by float, then fit on everything; returns the booster and the CV table."""
    import lightgbm as lgb

    fold = _folds(groups)
    oof = np.zeros_like(y)
    rounds = []
    for k in range(int(fold.max()) + 1):
        tr, va = fold != k, fold == k
        model = lgb.train(
            PARAMS,
            lgb.Dataset(x[tr], y[tr], feature_name=names),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(x[va], y[va], feature_name=names)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        rounds.append(int(model.best_iteration or MAX_ROUNDS))
        oof[va] = model.predict(x[va], num_iteration=model.best_iteration)
    n_trees = max(1, int(np.median(rounds)))
    rows = []
    for name, sel in (("all", np.ones_like(band)), ("bob_50_300", band)):
        rows.append(
            {
                "group": name,
                "n_obs": int(sel.sum()),
                "rmse_none_c": _rmse(y[sel]),
                "rmse_corrected_c": _rmse(y[sel] - oof[sel]),
                "n_trees": n_trees,
            }
        )
    depth_col = x[:, names.index("depth_m")]
    for d in np.unique(depth_col):
        sel = depth_col == d
        rows.append(
            {
                "group": f"{float(d):g}",
                "n_obs": int(sel.sum()),
                "rmse_none_c": _rmse(y[sel]),
                "rmse_corrected_c": _rmse(y[sel] - oof[sel]),
                "n_trees": n_trees,
            }
        )
    table = pd.DataFrame(rows, columns=CV_COLUMNS)
    final = lgb.train(PARAMS, lgb.Dataset(x, y, feature_name=names), num_boost_round=n_trees)
    return final, table


def write_store(
    pred_path: Path,
    surface: xr.Dataset | None,
    res: Residuals,
    model: Any,
    names: list[str],
    run_id: str,
    dest: Path,
    features: str = "v1",
    max_depth_m: float | None = None,
) -> Path:
    """Copy the prediction store to ``dest`` with the learnt correction added on valid cells
    (none on levels deeper than ``max_depth_m``)."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(pred_path, dest)
    pred = xr.open_zarr(dest)
    dtype = pred["temp_pred"].dtype
    depths = np.asarray(pred["depth"].values, dtype=float)
    lat_g, lon_g = np.meshgrid(pred["lat"].values, pred["lon"].values, indexing="ij")
    i_g, j_g = np.meshgrid(np.arange(lat_g.shape[0]), np.arange(lat_g.shape[1]), indexing="ij")
    days = _day_number(pred["time"].values)
    n_time = days.size
    sq = np.zeros(depths.size)
    n_cells = 0
    deep = np.zeros(depths.size, dtype=bool) if max_depth_m is None else depths > max_depth_m
    for b0 in range(0, n_time, BLOCK_DAYS):
        b1 = min(b0 + BLOCK_DAYS, n_time)
        block = pred.isel(time=slice(b0, b1)).load()
        temp = block["temp_pred"].transpose("time", "depth", "lat", "lon").values.astype(float)
        cov = block["coverage"].values
        for k in range(b1 - b0):
            valid = cov[k] == 1
            if not valid.any():
                continue
            day = pd.Timestamp(block["time"].values[k]).normalize()
            lat, lon = lat_g[valid], lon_g[valid]
            p = temp[k][:, valid].T
            x, _ = _day_features(features, res, int(days[b0 + k]), day, depths, p, lat, lon,
                                 (i_g[valid], j_g[valid]), surface, None, names)  # fmt: skip
            c = model.predict(x).reshape(p.shape)
            c = np.where(np.isfinite(p) & ~deep[None, :], c, 0.0)
            temp[k][:, valid] += c.T
            sq += (c**2).sum(axis=0)
            n_cells += int(valid.sum())
        out = xr.Dataset({"temp_pred": (("time", "depth", "lat", "lon"), temp.astype(dtype))})
        out.to_zarr(dest, region={"time": slice(b0, b1)}, mode="r+")
        logger.info("Corrected %d / %d days", b1, n_time)
    group = zarr.open_group(str(dest), mode="r+")
    base_run = str(group.attrs.get("run_id", ""))
    group.attrs.update(
        {
            "run_id": run_id,
            "base_run_id": base_run,
            "residual_model": json.dumps(names),
            "residual_model_max_depth_m": "" if max_depth_m is None else float(max_depth_m),
        }
    )
    zarr.consolidate_metadata(str(dest))
    rms = np.sqrt(sq / max(n_cells, 1))
    logger.info("RMS correction per depth (C): %s", np.round(rms, 3).tolist())
    return dest


def _surface(store_paths: list[Path], stats_path: Path | None, pred: xr.Dataset) -> xr.Dataset:
    ds = open_parts(store_paths, stats_path)
    names = [v for v in SURFACE_VARS if v in ds.data_vars]
    missing = sorted(set(SURFACE_VARS) - set(names))
    if missing:
        logger.warning("Surface features missing from the store: %s", missing)
    times = pd.DatetimeIndex(pred["time"].values).normalize()
    start = times.min() - pd.Timedelta(days=max(LAGS_DAYS.values()))
    have = pd.DatetimeIndex(ds["time"].values)
    keep = have[(have >= start) & (have <= times.max())]  # lagged features need earlier days
    sub = ds[names].sel(time=keep)
    if "clim" in ds:
        sub["clim"] = ds["clim"]
    return sub.sel(lat=pred["lat"].values, lon=pred["lon"].values, method="nearest")


def run(
    config: ExperimentConfig,
    predictions: Path,
    argo_paths: list[Path],
    holdout: set[str],
    run_id: str,
    store_paths: list[Path] | None = None,
    stats_path: Path | None = None,
    out_dir: Path | None = None,
    features: str = "v1",
    max_depth_m: float | None = None,
) -> Path:
    """Fit the residual model on non-held-out floats and write the corrected store."""
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}")
    pred = xr.open_zarr(predictions)
    dates = pd.DatetimeIndex(pred["time"].values).normalize()
    check_no_test_leakage(dates, config.splits.test)
    val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
    if dates.size == 0 or dates.min() < val_start or dates.max() > val_end:
        raise ValueError("The prediction store must hold validation dates only")
    depths = np.asarray(pred["depth"].values, dtype=float)
    surface = _surface(store_paths, stats_path, pred) if store_paths else None

    levels = read_levels(argo_paths, val_start, val_end)
    profiles = profile_levels(levels, [float(d) for d in depths], holdout)
    res = residuals(profiles, pred)
    if np.isin(res.platform, list(holdout)).any():
        raise AssertionError("A held-out float reached the residual model")
    logger.info("Residuals: %d profiles from %d non-held-out floats", len(res),
                np.unique(res.platform).size)  # fmt: skip
    x, y, groups, band, names = training_rows(res, pred, surface, features, max_depth_m)
    logger.info("Training rows: %d, features: %s", y.size, names)
    model, table = fit(x, y, groups, band, names)
    for _, row in table.iterrows():
        logger.info("  CV %s: RMSE %.3f -> %.3f C (%d obs)", row["group"], row["rmse_none_c"],
                    row["rmse_corrected_c"], row["n_obs"])  # fmt: skip
    out = Path(out_dir or config.paths.outputs)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    table.round(6).to_csv(out / "metrics" / f"residual_model_cv_{run_id}.csv", index=False)
    importance = dict(zip(names, model.feature_importance("gain").round(1).tolist()))
    logger.info("Feature importance (gain): %s", importance)
    dest = out / "predictions" / f"predictions_{run_id}.zarr"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_store(predictions, surface, res, model, names, run_id, dest, features,
                       max_depth_m)  # fmt: skip


def _load_holdout(path: Path) -> set[str]:
    floats = json.loads(Path(path).read_text(encoding="utf-8")).get("held_out_floats") or []
    if not floats:
        raise ValueError(f"{path} lists no held-out floats (PRD_AI_ML §7)")
    return {str(f).strip() for f in floats}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tree-model ARGO residual correction.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--store", nargs="*", default=None, help="Part stores (surface features)")
    parser.add_argument("--stats", default=None)
    parser.add_argument("--argo", nargs="+", required=True, help="Monthly ARGO NetCDF files")
    parser.add_argument("--holdout", required=True, help="argo_eval_floats.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    parser.add_argument("--features", choices=FEATURE_SETS, default="v1")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="No training or correction below this depth (m)")  # fmt: skip
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    dest = run(
        load_config(args.config),
        Path(args.predictions),
        [Path(p) for p in args.argo],
        _load_holdout(Path(args.holdout)),
        args.run_id,
        store_paths=[Path(p) for p in args.store] if args.store else None,
        stats_path=Path(args.stats) if args.stats else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        features=args.features,
        max_depth_m=args.max_depth,
    )
    print(dest)


if __name__ == "__main__":
    main()
