"""Shared per-product reduction and regridding functions for OceanScope AI.

Implements PRD_AI_ML §3, §4.2, and §6.1.
Shared between oceanscope_ml.data.preprocess (one-month sample) and
oceanscope_ml.data.build_store (full multi-year streaming build).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import scipy.interpolate
import xarray as xr

T = TypeVar("T", xr.Dataset, xr.DataArray)

logger = logging.getLogger("oceanscope_ml.data.reduce")

EARTH_RADIUS_M = 6_371_000.0
AIR_DENSITY_KG_M3 = 1.22


@dataclass(frozen=True)
class TargetGrid:
    """Target 0.25-degree coordinate grid definitions."""

    lats: np.ndarray  # 1D array of latitude cell centres
    lons: np.ndarray  # 1D array of longitude cell centres
    resolution_deg: float = 0.25

    @classmethod
    def from_bounds(
        cls,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        resolution_deg: float,
    ) -> TargetGrid:
        """Construct target grid with cell centres matching PRD_AI_ML §2."""
        lons = np.arange(lon_min + resolution_deg / 2.0, lon_max, resolution_deg, dtype=np.float64)
        lats = np.arange(lat_min + resolution_deg / 2.0, lat_max, resolution_deg, dtype=np.float64)
        return cls(lats=lats, lons=lons, resolution_deg=resolution_deg)

    @property
    def lat_edges(self) -> np.ndarray:
        """Cell edge boundaries for latitudes."""
        half = self.resolution_deg / 2.0
        return np.append(self.lats - half, self.lats[-1] + half)

    @property
    def lon_edges(self) -> np.ndarray:
        """Cell edge boundaries for longitudes."""
        half = self.resolution_deg / 2.0
        return np.append(self.lons - half, self.lons[-1] + half)


def standardize_coords(ds: T) -> T:
    """Standardize coordinate names, order, and ranges across heterogeneous datasets.

    Rules applied:
    - Latitude names (latitude, nav_lat) -> lat, sorted ascending
    - Longitude names (longitude, nav_lon) -> lon, mapped to [-180, 180] convention
    - Depth names (deptht, lev, level, depth_levels) -> depth
    - Time names (time_counter, valid_time) -> time
    - Dim order: (..., lat, lon)
    """
    # 1. Check if dimensions need swap
    swap_map: dict[str, str] = {}
    if hasattr(ds, "dims"):
        for d in ds.dims:
            d_str = str(d)
            d_lower = d_str.lower()
            if d_lower in ["latitude", "nav_lat"] and "lat" in ds.coords and d_str != "lat":
                swap_map[d_str] = "lat"
            elif d_lower in ["longitude", "nav_lon"] and "lon" in ds.coords and d_str != "lon":
                swap_map[d_str] = "lon"
            elif (
                d_lower in ["deptht", "lev", "level", "depth_levels"]
                and "depth" in ds.coords
                and d_str != "depth"
            ):
                swap_map[d_str] = "depth"
            elif (
                d_lower in ["time_counter", "valid_time"]
                and "time" in ds.coords
                and d_str != "time"
            ):
                swap_map[d_str] = "time"
    if swap_map and hasattr(ds, "swap_dims"):
        try:
            ds = ds.swap_dims(swap_map)
        except (ValueError, KeyError) as e:
            logger.debug("swap_dims skipped: %s", e)

    # 2. Check rename for coordinates or dimensions
    rename_map: dict[str, str] = {}
    if hasattr(ds, "coords"):
        for c in ds.coords:
            c_str = str(c)
            c_lower = c_str.lower()
            if c_lower in ["latitude", "nav_lat"] and c_str != "lat" and "lat" not in ds.coords:
                rename_map[c_str] = "lat"
            elif c_lower in ["longitude", "nav_lon"] and c_str != "lon" and "lon" not in ds.coords:
                rename_map[c_str] = "lon"
            elif (
                c_lower in ["deptht", "lev", "level", "depth_levels"]
                and c_str != "depth"
                and "depth" not in ds.coords
            ):
                rename_map[c_str] = "depth"
            elif (
                c_lower in ["time_counter", "valid_time"]
                and c_str != "time"
                and "time" not in ds.coords
            ):
                rename_map[c_str] = "time"

    if hasattr(ds, "dims"):
        for d in ds.dims:
            d_str = str(d)
            d_lower = d_str.lower()
            if (
                d_lower in ["latitude", "nav_lat"]
                and d_str != "lat"
                and "lat" not in ds.dims
                and d_str not in rename_map
            ):
                rename_map[d_str] = "lat"
            elif (
                d_lower in ["longitude", "nav_lon"]
                and d_str != "lon"
                and "lon" not in ds.dims
                and d_str not in rename_map
            ):
                rename_map[d_str] = "lon"
            elif (
                d_lower in ["deptht", "lev", "level", "depth_levels"]
                and d_str != "depth"
                and "depth" not in ds.dims
                and d_str not in rename_map
            ):
                rename_map[d_str] = "depth"
            elif (
                d_lower in ["time_counter", "valid_time"]
                and d_str != "time"
                and "time" not in ds.dims
                and d_str not in rename_map
            ):
                rename_map[d_str] = "time"

    if rename_map:
        try:
            ds = ds.rename(rename_map)
        except (ValueError, KeyError) as e:
            logger.debug("rename skipped: %s", e)

    # Standardize longitude convention to -180..180 if in 0..360
    if "lon" in ds.coords:
        lon_vals = ds.coords["lon"].values
        if np.any(lon_vals > 180.0):
            norm_lons = ((lon_vals + 180.0) % 360.0) - 180.0
            ds = ds.assign_coords(lon=norm_lons)
            ds = ds.sortby("lon")

    # Ensure latitude is sorted ascending
    if "lat" in ds.coords:
        lat_vals = ds.coords["lat"].values
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            ds = ds.sortby("lat")

    # Ensure lat dimension appears before lon dimension
    if "lat" in ds.dims and "lon" in ds.dims:
        try:
            ds = ds.transpose(..., "lat", "lon")
        except (ValueError, KeyError) as e:
            logger.debug("transpose skipped: %s", e)

    return ds


def to_clean_datetime_index(times: Any) -> pd.DatetimeIndex:
    """Safely convert any times (including cftime objects) to normalized calendar dates."""
    if hasattr(times, "__iter__") and not isinstance(times, (str, bytes)):
        str_dates = [str(t)[:10] for t in times]
        return pd.to_datetime(str_dates).floor("D")
    return pd.to_datetime([str(times)[:10]]).floor("D")


def match_time_indices(
    src_times: np.ndarray,
    target_times: np.ndarray,
) -> list[int]:
    """Find index in src_times for each target time by exact calendar day.

    Returns -1 for any target day missing in src_times (PRD §0 rule 5: no silent gap filling).
    """
    src_days = to_clean_datetime_index(src_times)
    target_days = to_clean_datetime_index(target_times)
    mapping = {d: i for i, d in enumerate(src_days)}
    return [mapping.get(t, -1) for t in target_days]


def area_mean_regrid_2d(
    values: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_grid: TargetGrid,
    valid_frac_threshold: float = 0.5,
) -> np.ndarray:
    """Regrid 2D spatial field to target grid with >= 50% valid sub-cells rule.

    Parameters
    ----------
    values : np.ndarray, shape (n_src_lat, n_src_lon)
    src_lats : np.ndarray, shape (n_src_lat,)
    src_lons : np.ndarray, shape (n_src_lon,)
    target_grid : TargetGrid
    valid_frac_threshold : float, default 0.5 (>= 50% valid sub-cells)

    Returns
    -------
    np.ndarray, shape (n_target_lat, n_target_lon), float32
    """
    n_tgt_lat = len(target_grid.lats)
    n_tgt_lon = len(target_grid.lons)

    # Find bin indices for source coordinates
    lat_bins = np.digitize(src_lats, target_grid.lat_edges) - 1
    lon_bins = np.digitize(src_lons, target_grid.lon_edges) - 1

    # Keep only source coordinates falling inside the target grid boundaries
    valid_lat_idx = np.where((lat_bins >= 0) & (lat_bins < n_tgt_lat))[0]
    valid_lon_idx = np.where((lon_bins >= 0) & (lon_bins < n_tgt_lon))[0]

    if len(valid_lat_idx) == 0 or len(valid_lon_idx) == 0:
        return np.full((n_tgt_lat, n_tgt_lon), np.nan, dtype=np.float32)

    sub_lats_b = lat_bins[valid_lat_idx]
    sub_lons_b = lon_bins[valid_lon_idx]
    sub_vals = values[np.ix_(valid_lat_idx, valid_lon_idx)]

    # 2D grid of target flat cell indices for each sub-cell
    grid_lat_b, grid_lon_b = np.meshgrid(sub_lats_b, sub_lons_b, indexing="ij")
    flat_tgt_cells = grid_lat_b.ravel() * n_tgt_lon + grid_lon_b.ravel()
    flat_vals = sub_vals.ravel()

    # Total sub-cells mapped into each target cell
    total_counts = np.bincount(flat_tgt_cells, minlength=n_tgt_lat * n_tgt_lon)

    # Valid (non-NaN) sub-cells
    valid_mask = ~np.isnan(flat_vals)
    valid_counts = np.bincount(
        flat_tgt_cells[valid_mask],
        minlength=n_tgt_lat * n_tgt_lon,
    )

    sum_vals = np.bincount(
        flat_tgt_cells[valid_mask],
        weights=flat_vals[valid_mask],
        minlength=n_tgt_lat * n_tgt_lon,
    )

    # Valid fraction rule: valid_counts / total_counts >= threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        valid_frac = np.where(total_counts > 0, valid_counts / total_counts, 0.0)
        mean_vals = np.where(
            valid_frac >= valid_frac_threshold,
            sum_vals / np.maximum(valid_counts, 1),
            np.nan,
        )

    res = mean_vals.reshape((n_tgt_lat, n_tgt_lon)).astype(np.float32)

    # Fallback if source resolution is coarser than target resolution (e.g. 1/3 deg)
    # where some target cells receive 0 total sub-cells:
    if np.any(total_counts == 0) and len(src_lats) > 1 and len(src_lons) > 1:
        unfilled_mask = total_counts.reshape((n_tgt_lat, n_tgt_lon)) == 0
        if np.any(unfilled_mask):
            try:
                interp = scipy.interpolate.RegularGridInterpolator(
                    (src_lats, src_lons),
                    values,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                tgt_pts = np.meshgrid(target_grid.lats, target_grid.lons, indexing="ij")
                interp_vals = interp((tgt_pts[0][unfilled_mask], tgt_pts[1][unfilled_mask]))
                res[unfilled_mask] = interp_vals.astype(np.float32)
            except Exception as e:
                logger.debug("RegularGridInterpolator fallback skipped: %s", e)

    return res


def interpolate_depths_1d(
    values_by_depth: np.ndarray,
    src_depths: np.ndarray,
    target_depths: list[float],
) -> np.ndarray:
    """Interpolate 3D array (depth, lat, lon) along depth axis without extrapolation.

    Matches source depths within atol=0.05 to avoid float32 rounding discrepancies
    (e.g. GLORYS 1062.43994 m vs 1062.44 m; REVIEW_P3 R2). Clamps within 5 cm at
    boundaries (DECISIONS 2026-10-25 for GREP 0.494 m vs 0.506 m).
    """
    tgt_depth_arr = np.array(target_depths, dtype=np.float64)
    n_tgt = len(tgt_depth_arr)
    _, n_lat, n_lon = values_by_depth.shape

    # Fast path: check if every target depth has an exact match in src_depths within 0.05 m
    matched_indices: list[int] = []
    for td in tgt_depth_arr:
        diffs = np.abs(src_depths - td)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= 0.05:
            matched_indices.append(best_idx)
        else:
            matched_indices = []
            break

    if len(matched_indices) == n_tgt:
        return values_by_depth[matched_indices].astype(np.float32)

    out = np.full((n_tgt, n_lat, n_lon), np.nan, dtype=np.float32)
    min_d, max_d = float(np.min(src_depths)), float(np.max(src_depths))

    # Clamp boundary target depths within 5 cm (e.g. GREP 0.494 vs 0.506, GLORYS float32 max)
    eval_depths = tgt_depth_arr.copy()
    near_surface = (eval_depths < min_d) & (min_d - eval_depths <= 0.05)
    eval_depths[near_surface] = min_d
    near_bottom = (eval_depths > max_d) & (eval_depths - max_d <= 0.05)
    eval_depths[near_bottom] = max_d

    valid_tgt_mask = (eval_depths >= min_d) & (eval_depths <= max_d)
    if not np.any(valid_tgt_mask):
        return out

    interp_fn = scipy.interpolate.interp1d(
        src_depths,
        values_by_depth,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interpolated = interp_fn(eval_depths[valid_tgt_mask])
    out[valid_tgt_mask] = interpolated.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Per-Product Reduce Functions (PRD_AI_ML §3, §4.2)
# ---------------------------------------------------------------------------


def reduce_glorys12(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    depths_m: list[float],
) -> xr.Dataset:
    """Reduce GLORYS12 1/12° reanalysis thetao to target grid and 15 depths.

    Cell is valid if >= 50% native ocean sub-cells are valid (PRD_AI_ML §3).
    """
    ds = standardize_coords(ds)
    if "thetao" not in ds.data_vars:
        raise KeyError(
            f"Expected variable 'thetao' in GLORYS12 dataset, found: {list(ds.data_vars)}"
        )
    da = ds["thetao"]

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    src_depths = np.asarray(ds.coords["depth"].values, dtype=np.float64)

    time_vals = (
        ds.coords["time"].values
        if "time" in ds.coords
        else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
    )
    if np.ndim(time_vals) == 0:
        time_vals = np.array([time_vals])
    n_time = len(time_vals)
    n_depth = len(depths_m)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    out_arr = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    vals = da.values
    if vals.ndim == 3:  # (depth, lat, lon)
        vals = vals[np.newaxis, ...]
    elif vals.ndim == 2:  # (lat, lon)
        vals = vals[np.newaxis, np.newaxis, ...]

    # Match configured depths to GLORYS native levels with tolerance (atol=0.05)
    matched_indices: list[int] = []
    for td in depths_m:
        diffs = np.abs(src_depths - td)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= 0.05:
            matched_indices.append(best_idx)
        else:
            matched_indices = []
            break
    direct_depth_slicing = len(matched_indices) == n_depth

    for t_idx in range(n_time):
        t_slice = vals[t_idx]
        if direct_depth_slicing:
            # Regrid only the 15 selected depth slices directly (REVIEW_P3 R2)
            t_slice_15 = t_slice[matched_indices]
            for d_idx in range(n_depth):
                out_arr[t_idx, d_idx] = area_mean_regrid_2d(
                    t_slice_15[d_idx],
                    src_lats,
                    src_lons,
                    target_grid,
                    valid_frac_threshold=0.5,
                )
        else:
            regrid_by_depth = np.full((len(src_depths), n_lat, n_lon), np.nan, dtype=np.float32)
            for d_idx in range(len(src_depths)):
                regrid_by_depth[d_idx] = area_mean_regrid_2d(
                    t_slice[d_idx],
                    src_lats,
                    src_lons,
                    target_grid,
                    valid_frac_threshold=0.5,
                )
            out_arr[t_idx] = interpolate_depths_1d(regrid_by_depth, src_depths, depths_m)

    surface_layer = out_arr[:, 0, :, :]
    ocean_mask = np.any(~np.isnan(surface_layer), axis=0)
    target_valid = ~np.isnan(out_arr)

    return xr.Dataset(
        data_vars={
            "thetao": (("time", "depth", "lat", "lon"), out_arr),
            "ocean_mask": (("lat", "lon"), ocean_mask),
            "target_valid": (("time", "depth", "lat", "lon"), target_valid),
        },
        coords={
            "time": time_vals,
            "depth": np.array(depths_m, dtype=np.float64),
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )


def reduce_sla(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce DUACS L4 SLA to 0.25° target grid with >= 50% valid rule."""
    ds = standardize_coords(ds)
    if "sla" not in ds.data_vars:
        raise KeyError(f"Expected variable 'sla' in SLA dataset, found: {list(ds.data_vars)}")
    da = ds["sla"]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    if target_time is not None and "time" in da.coords:
        time_vals = target_time
        time_indices = match_time_indices(da.coords["time"].values, time_vals)
    else:
        time_vals = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        if np.ndim(time_vals) == 0:
            time_vals = np.array([time_vals])
        time_indices = list(range(len(time_vals)))

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx, src_idx in enumerate(time_indices):
        if src_idx < 0 or src_idx >= vals.shape[0]:
            continue
        out_arr[t_idx] = area_mean_regrid_2d(
            vals[src_idx],
            src_lats,
            src_lons,
            target_grid,
            valid_frac_threshold=0.5,
        )

    return xr.Dataset(
        data_vars={"sla": (("time", "lat", "lon"), out_arr)},
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_adt(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce DUACS L4 ADT to 0.25° target grid with >= 50% valid rule."""
    ds = standardize_coords(ds)
    if "adt" not in ds.data_vars:
        raise KeyError(f"Expected variable 'adt' in ADT dataset, found: {list(ds.data_vars)}")
    da = ds["adt"]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    if target_time is not None and "time" in da.coords:
        time_vals = target_time
        time_indices = match_time_indices(da.coords["time"].values, time_vals)
    else:
        time_vals = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        if np.ndim(time_vals) == 0:
            time_vals = np.array([time_vals])
        time_indices = list(range(len(time_vals)))

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx, src_idx in enumerate(time_indices):
        if src_idx < 0 or src_idx >= vals.shape[0]:
            continue
        out_arr[t_idx] = area_mean_regrid_2d(
            vals[src_idx],
            src_lats,
            src_lons,
            target_grid,
            valid_frac_threshold=0.5,
        )

    return xr.Dataset(
        data_vars={"adt": (("time", "lat", "lon"), out_arr)},
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_sst(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce OSTIA L4 SST to 0.25° target grid, converting Kelvin to °C."""
    ds = standardize_coords(ds)
    var_name = None
    for cand in ["analysed_sst", "sst"]:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise KeyError(
            "Expected SST variable ('analysed_sst' or 'sst') not found in dataset: "
            f"{list(ds.data_vars)}"
        )

    da = ds[var_name]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    if target_time is not None and "time" in da.coords:
        time_vals = target_time
        time_indices = match_time_indices(da.coords["time"].values, time_vals)
    else:
        time_vals = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        if np.ndim(time_vals) == 0:
            time_vals = np.array([time_vals])
        time_indices = list(range(len(time_vals)))

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    # Convert Kelvin to Celsius (PRD_AI_ML §3, REVIEW_P3 B3)
    units_attr = str(da.attrs.get("units", "")).lower()
    is_kelvin = "kelvin" in units_attr or units_attr == "k"
    if not is_kelvin and units_attr == "":
        valid_sample = vals[~np.isnan(vals)]
        if len(valid_sample) > 0 and float(np.mean(valid_sample[:1000])) > 100.0:
            is_kelvin = True

    if is_kelvin:
        vals = vals - 273.15

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx, src_idx in enumerate(time_indices):
        if src_idx < 0 or src_idx >= vals.shape[0]:
            continue
        out_arr[t_idx] = area_mean_regrid_2d(
            vals[src_idx],
            src_lats,
            src_lons,
            target_grid,
            valid_frac_threshold=0.5,
        )

    return xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), out_arr)},
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_sss(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce MULTIOBS SSS to 0.25° target grid with >= 50% valid rule."""
    ds = standardize_coords(ds)
    var_name = None
    for cand in ["sos", "sss", "so"]:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise KeyError(
            f"Expected SSS variable ('sos' or 'sss') not found in dataset: {list(ds.data_vars)}"
        )

    da = ds[var_name]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    if target_time is not None and "time" in da.coords:
        time_vals = target_time
        time_indices = match_time_indices(da.coords["time"].values, time_vals)
    else:
        time_vals = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        if np.ndim(time_vals) == 0:
            time_vals = np.array([time_vals])
        time_indices = list(range(len(time_vals)))

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx, src_idx in enumerate(time_indices):
        if src_idx < 0 or src_idx >= vals.shape[0]:
            continue
        out_arr[t_idx] = area_mean_regrid_2d(
            vals[src_idx],
            src_lats,
            src_lons,
            target_grid,
            valid_frac_threshold=0.5,
        )

    return xr.Dataset(
        data_vars={"sss": (("time", "lat", "lon"), out_arr)},
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_currents(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce OSCAR surface currents (u, v) to 0.25° target grid."""
    ds = standardize_coords(ds)
    u_var = None
    v_var = None
    for cand_u in ["u", "u_current", "ug"]:
        if cand_u in ds.data_vars:
            u_var = cand_u
            break
    for cand_v in ["v", "v_current", "vg"]:
        if cand_v in ds.data_vars:
            v_var = cand_v
            break
    if u_var is None or v_var is None:
        raise KeyError(
            "Expected surface currents variables ('u'/'v') not found in dataset: "
            f"{list(ds.data_vars)}"
        )

    da_u = ds[u_var]
    da_v = ds[v_var]
    if "depth" in da_u.dims:
        da_u = da_u.squeeze("depth")
        da_v = da_v.squeeze("depth")

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    if target_time is not None and "time" in da_u.coords:
        time_vals = target_time
        time_indices = match_time_indices(da_u.coords["time"].values, time_vals)
    else:
        time_vals = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        if np.ndim(time_vals) == 0:
            time_vals = np.array([time_vals])
        time_indices = list(range(len(time_vals)))

    u_vals = da_u.values
    v_vals = da_v.values
    while u_vals.ndim > 3:
        u_vals = np.squeeze(u_vals, axis=1)
        v_vals = np.squeeze(v_vals, axis=1)
    if u_vals.ndim == 2:
        u_vals = u_vals[np.newaxis, ...]
        v_vals = v_vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    regrid_u = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    regrid_v = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx, src_idx in enumerate(time_indices):
        if src_idx < 0 or src_idx >= u_vals.shape[0]:
            continue
        regrid_u[t_idx] = area_mean_regrid_2d(
            u_vals[src_idx],
            src_lats,
            src_lons,
            target_grid,
            valid_frac_threshold=0.5,
        )
        regrid_v[t_idx] = area_mean_regrid_2d(
            v_vals[src_idx],
            src_lats,
            src_lons,
            target_grid,
            valid_frac_threshold=0.5,
        )

    return xr.Dataset(
        data_vars={
            "u_current": (("time", "lat", "lon"), regrid_u),
            "v_current": (("time", "lat", "lon"), regrid_v),
        },
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def compute_wind_stress_and_curl(
    u10: np.ndarray,
    v10: np.ndarray,
    target_grid: TargetGrid,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute wind stress vector and spherical curl from 10m wind components (PRD_AI_ML §3).

    Bulk formula:
    rho_air = 1.22 kg/m³
    Cd per Large & Pond (1981):
      Cd = 1.2e-3 for |U| < 11 m/s
      Cd = (0.49 + 0.065 * |U|) * 1e-3 for 11 <= |U| <= 25 m/s
      Cd capped at 25 m/s value above 25 m/s
    tau = rho_air * Cd * |U| * U

    Spherical curl:
    curl(tau) = 1/(R * cos(phi)) * d(tau_y)/d(lon) - 1/R * d(tau_x)/d(lat)

    Per REVIEW_P3 B4, wind stress curl is masked at the coast and land (requires cell
    and all 4 stencil neighbors to be ocean).
    """
    n_time, n_lat, n_lon = u10.shape
    curl_out = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    speed = np.sqrt(u10**2 + v10**2)

    cd = np.where(
        speed < 11.0,
        1.2e-3,
        np.where(
            speed <= 25.0,
            (0.49 + 0.065 * speed) * 1e-3,
            (0.49 + 0.065 * 25.0) * 1e-3,
        ),
    )

    tau_x = (AIR_DENSITY_KG_M3 * cd * speed * u10).astype(np.float32)
    tau_y = (AIR_DENSITY_KG_M3 * cd * speed * v10).astype(np.float32)

    lat_rad = np.radians(target_grid.lats)
    cos_lat = np.cos(lat_rad)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-4, 1e-4, cos_lat)

    delta_lat_rad = np.radians(target_grid.resolution_deg)
    delta_lon_rad = np.radians(target_grid.resolution_deg)

    # Valid ocean stencil mask for coast masking (REVIEW_P3 B4)
    valid_ocean_interior: np.ndarray | None = None
    if ocean_mask is not None:
        om = np.asarray(ocean_mask, dtype=bool)
        valid_ocean_interior = (
            om[1:-1, 1:-1] & om[:-2, 1:-1] & om[2:, 1:-1] & om[1:-1, :-2] & om[1:-1, 2:]
        )

    for t_idx in range(n_time):
        tx = tau_x[t_idx]
        ty = tau_y[t_idx]

        d_ty_d_lon = (ty[:, 2:] - ty[:, :-2]) / (2.0 * delta_lon_rad)
        d_tx_d_lat = (tx[2:, :] - tx[:-2, :]) / (2.0 * delta_lat_rad)

        term1 = d_ty_d_lon[1:-1, :] / (EARTH_RADIUS_M * cos_lat_safe[1:-1, np.newaxis])
        term2 = d_tx_d_lat[:, 1:-1] / EARTH_RADIUS_M

        curl_interior = term1 - term2
        if valid_ocean_interior is not None:
            curl_interior = np.where(valid_ocean_interior, curl_interior, np.nan)

        curl_out[t_idx, 1:-1, 1:-1] = curl_interior.astype(np.float32)

    return u10.astype(np.float32), v10.astype(np.float32), curl_out


def reduce_era5_winds(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
    ocean_mask: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce ERA5 10m winds to 0.25° grid, computing stress and curl (PRD_AI_ML §3).

    Requires >= 18 of 24 valid hours for daily means when processing hourly files.
    """
    ds = standardize_coords(ds)
    u_var = None
    v_var = None
    for cand_u in ["u10", "10m_u_component_of_wind", "u_wind", "u"]:
        if cand_u in ds.data_vars:
            u_var = cand_u
            break
    for cand_v in ["v10", "10m_v_component_of_wind", "v_wind", "v"]:
        if cand_v in ds.data_vars:
            v_var = cand_v
            break
    if u_var is None or v_var is None:
        raise KeyError(
            f"Expected wind variables ('u10'/'v10') not found in dataset: {list(ds.data_vars)}"
        )

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    time_vals = (
        target_time
        if target_time is not None
        else (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
    )
    if np.ndim(time_vals) == 0:
        time_vals = np.array([time_vals])

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    regrid_u = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    regrid_v = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    is_hourly = False
    if "time" in ds.coords and len(ds.coords["time"]) > 1:
        time_diffs = np.diff(pd.to_datetime(ds.coords["time"].values[:10]))
        median_hours = pd.Series(time_diffs).median().total_seconds() / 3600.0
        if median_hours < 12.0:
            is_hourly = True

    if is_hourly and "time" in ds.coords:
        src_days = to_clean_datetime_index(ds.coords["time"].values)
        target_days = to_clean_datetime_index(time_vals)
        for t_idx, t in enumerate(target_days):
            day_mask = src_days == t
            n_hours = int(np.sum(day_mask))
            if n_hours >= 18:
                u_slice = ds[u_var].isel(time=np.where(day_mask)[0]).values
                v_slice = ds[v_var].isel(time=np.where(day_mask)[0]).values
                while u_slice.ndim > 3:
                    u_slice = np.squeeze(u_slice, axis=1)
                    v_slice = np.squeeze(v_slice, axis=1)
                u_mean = np.nanmean(u_slice, axis=0)
                v_mean = np.nanmean(v_slice, axis=0)
                regrid_u[t_idx] = area_mean_regrid_2d(u_mean, src_lats, src_lons, target_grid)
                regrid_v[t_idx] = area_mean_regrid_2d(v_mean, src_lats, src_lons, target_grid)
    else:
        src_times = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        time_indices = match_time_indices(src_times, time_vals)
        u_data = ds[u_var].values
        v_data = ds[v_var].values
        while u_data.ndim > 3:
            u_data = np.squeeze(u_data, axis=1)
            v_data = np.squeeze(v_data, axis=1)
        if u_data.ndim == 2:
            u_data = u_data[np.newaxis, ...]
            v_data = v_data[np.newaxis, ...]

        for t_idx, src_idx in enumerate(time_indices):
            if src_idx < 0 or src_idx >= u_data.shape[0]:
                continue
            u_t = u_data[src_idx]
            v_t = v_data[src_idx]
            regrid_u[t_idx] = area_mean_regrid_2d(u_t, src_lats, src_lons, target_grid)
            regrid_v[t_idx] = area_mean_regrid_2d(v_t, src_lats, src_lons, target_grid)

    u_wind, v_wind, curl = compute_wind_stress_and_curl(
        regrid_u, regrid_v, target_grid, ocean_mask=ocean_mask
    )

    return xr.Dataset(
        data_vars={
            "u_wind": (("time", "lat", "lon"), u_wind),
            "v_wind": (("time", "lat", "lon"), v_wind),
            "wind_stress_curl": (("time", "lat", "lon"), curl),
        },
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_era5_fluxes(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce ERA5 net heat flux and precipitation to 0.25° target grid.

    Net heat flux Q_net = ssr + str + sshf + slhf (PRD_AI_ML §3).
    ERA5 hourly accum fields (J m**-2) are converted to W m**-2 (divided by 3600 s)
    and daily averaged requiring >= 18 of 24 valid hours.
    Precipitation tp (hourly meters) is daily summed requiring >= 18 of 24 valid hours.
    """
    ds = standardize_coords(ds)
    has_components = all(k in ds.data_vars for k in ["ssr", "str", "sshf", "slhf"])
    if not has_components and "net_heat_flux" not in ds.data_vars:
        raise KeyError(
            "Expected flux components ('ssr', 'str', 'sshf', 'slhf') or 'net_heat_flux' "
            f"in dataset: {list(ds.data_vars)}"
        )

    tp_var = None
    for cand in ["tp", "precip", "total_precipitation"]:
        if cand in ds.data_vars:
            tp_var = cand
            break
    if tp_var is None:
        raise KeyError(
            f"Expected precipitation ('tp' or 'precip') in dataset: {list(ds.data_vars)}"
        )

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    time_vals = (
        target_time
        if target_time is not None
        else (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
    )
    if np.ndim(time_vals) == 0:
        time_vals = np.array([time_vals])

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    out_heat = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    out_precip = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    is_hourly = False
    if "time" in ds.coords and len(ds.coords["time"]) > 1:
        time_diffs = np.diff(pd.to_datetime(ds.coords["time"].values[:10]))
        median_hours = pd.Series(time_diffs).median().total_seconds() / 3600.0
        if median_hours < 12.0:
            is_hourly = True

    is_joules = False
    if has_components:
        ssr_units = str(ds["ssr"].attrs.get("units", "")).lower()
        if "j" in ssr_units:
            is_joules = True

    if is_hourly and "time" in ds.coords:
        src_days = to_clean_datetime_index(ds.coords["time"].values)
        target_days = to_clean_datetime_index(time_vals)
        for t_idx, t in enumerate(target_days):
            day_mask = src_days == t
            n_hours = int(np.sum(day_mask))
            if n_hours >= 18:
                if has_components:
                    ssr_vals = ds["ssr"].isel(time=np.where(day_mask)[0]).values
                    str_vals = ds["str"].isel(time=np.where(day_mask)[0]).values
                    sshf_vals = ds["sshf"].isel(time=np.where(day_mask)[0]).values
                    slhf_vals = ds["slhf"].isel(time=np.where(day_mask)[0]).values
                    q_hour = ssr_vals + str_vals + sshf_vals + slhf_vals
                    if is_joules:
                        q_hour = q_hour / 3600.0
                    q_day = np.nanmean(q_hour, axis=0)
                else:
                    q_vals = ds["net_heat_flux"].isel(time=np.where(day_mask)[0]).values
                    q_day = np.nanmean(q_vals, axis=0)

                tp_vals = ds[tp_var].isel(time=np.where(day_mask)[0]).values
                valid_hours_count = np.sum(np.isfinite(tp_vals), axis=0)
                # Cell is valid only if valid_hours >= 18; scale daily sum to 24 hours
                tp_day = np.where(
                    valid_hours_count >= 18,
                    np.nansum(tp_vals, axis=0) * (24.0 / np.maximum(valid_hours_count, 1)),
                    np.nan,
                )

                while q_day.ndim > 2:
                    q_day = np.squeeze(q_day, axis=0)
                while tp_day.ndim > 2:
                    tp_day = np.squeeze(tp_day, axis=0)

                out_heat[t_idx] = area_mean_regrid_2d(
                    q_day.astype(np.float32), src_lats, src_lons, target_grid
                )
                out_precip[t_idx] = area_mean_regrid_2d(
                    tp_day.astype(np.float32), src_lats, src_lons, target_grid
                )
    else:
        src_times = (
            ds.coords["time"].values
            if "time" in ds.coords
            else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        )
        time_indices = match_time_indices(src_times, time_vals)
        if has_components:
            q_all = ds["ssr"].values + ds["str"].values + ds["sshf"].values + ds["slhf"].values
            if is_joules:
                q_all = q_all / 3600.0
        else:
            q_all = ds["net_heat_flux"].values

        tp_all = ds[tp_var].values
        while q_all.ndim > 3:
            q_all = np.squeeze(q_all, axis=1)
            tp_all = np.squeeze(tp_all, axis=1)
        if q_all.ndim == 2:
            q_all = q_all[np.newaxis, ...]
            tp_all = tp_all[np.newaxis, ...]

        for t_idx, src_idx in enumerate(time_indices):
            if src_idx < 0 or src_idx >= q_all.shape[0]:
                continue
            q_t = q_all[src_idx]
            tp_t = tp_all[src_idx]
            out_heat[t_idx] = area_mean_regrid_2d(
                q_t.astype(np.float32), src_lats, src_lons, target_grid
            )
            out_precip[t_idx] = area_mean_regrid_2d(
                tp_t.astype(np.float32), src_lats, src_lons, target_grid
            )

    return xr.Dataset(
        data_vars={
            "net_heat_flux": (("time", "lat", "lon"), out_heat),
            "precip": (("time", "lat", "lon"), out_precip),
        },
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_gebco(ds: xr.Dataset, target_grid: TargetGrid) -> xr.Dataset:
    """Reduce GEBCO 15 arc-sec bathymetry to 0.25° grid, storing log(1 + depth)."""
    ds = standardize_coords(ds)
    var_name = None
    for cand in ["elevation", "z", "bathymetry"]:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise KeyError(
            "Expected bathymetry variable ('elevation' or 'z') in GEBCO dataset, "
            f"found: {list(ds.data_vars)}"
        )

    # Spatial subset before loading large global grids into memory
    lat_min = float(target_grid.lats.min()) - 0.5
    lat_max = float(target_grid.lats.max()) + 0.5
    lon_min = float(target_grid.lons.min()) - 0.5
    lon_max = float(target_grid.lons.max()) + 0.5

    if "lat" in ds.coords and len(ds.coords["lat"]) > len(target_grid.lats) * 4:
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    elev = ds[var_name].values.astype(np.float32)
    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)

    # Ocean depth in meters: depth = max(0, -elevation)
    depth = np.maximum(0.0, -elev)

    # Area-mean regrid to target grid
    regrid_depth = area_mean_regrid_2d(depth, src_lats, src_lons, target_grid)
    log_depth = np.log1p(np.maximum(0.0, regrid_depth)).astype(np.float32)

    return xr.Dataset(
        data_vars={"bathymetry": (("lat", "lon"), log_depth)},
        coords={"lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_woa(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    depths_m: list[float],
    month: int | None = None,
) -> xr.Dataset:
    """Reduce World Ocean Atlas (WOA23) monthly climatology to 15 depths and 0.25° grid.

    Always returns (month, depth, lat, lon) without any numeric time dimension (REVIEW_P3 R1).
    """
    ds = standardize_coords(ds)
    if "time" in ds.dims:
        ds = ds.squeeze("time", drop=True)
    if "time" in ds.coords:
        ds = ds.drop_vars("time", errors="ignore")

    var_name = None
    for cand in ["t_an", "temperature", "t_mn"]:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise KeyError(
            f"Expected temperature variable ('t_an') in WOA dataset, found: {list(ds.data_vars)}"
        )

    da = ds[var_name]
    while da.ndim > 3:
        da = da.squeeze(axis=0)

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    src_depths = np.asarray(ds.coords["depth"].values, dtype=np.float64)

    vals = da.values
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    regrid_by_depth = np.full((len(src_depths), n_lat, n_lon), np.nan, dtype=np.float32)
    for d_idx in range(len(src_depths)):
        regrid_by_depth[d_idx] = area_mean_regrid_2d(
            vals[d_idx],
            src_lats,
            src_lons,
            target_grid,
        )
    interp_vals = interpolate_depths_1d(regrid_by_depth, src_depths, depths_m)
    out_arr = interp_vals[np.newaxis, ...]  # (1, depth, lat, lon)

    if month is not None:
        month_val = int(month)
    elif "month" in ds.coords and len(ds.coords["month"].values) > 0:
        month_val = int(ds.coords["month"].values[0])
    else:
        month_val = 1

    return xr.Dataset(
        data_vars={"woa_clim_profile": (("month", "depth", "lat", "lon"), out_arr)},
        coords={
            "month": [month_val],
            "depth": np.array(depths_m, dtype=np.float64),
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )


def reduce_grep(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    depths_m: list[float],
) -> xr.Dataset:
    """Reduce GREP ensemble mean thetao_mean to 15 depths and target 0.25° grid."""
    ds = standardize_coords(ds)
    var_name = None
    for cand in ["thetao_mean", "thetao", "temperature"]:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise KeyError(
            f"Expected variable ('thetao_mean') in GREP dataset, found: {list(ds.data_vars)}"
        )

    da = ds[var_name]
    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    src_depths = np.asarray(ds.coords["depth"].values, dtype=np.float64)
    time_vals = (
        ds.coords["time"].values
        if "time" in ds.coords
        else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
    )
    if np.ndim(time_vals) == 0:
        time_vals = np.array([time_vals])

    vals = da.values
    if vals.ndim == 3:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_depth = len(depths_m)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    out_arr = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        regrid_by_depth = np.full((len(src_depths), n_lat, n_lon), np.nan, dtype=np.float32)
        for d_idx in range(len(src_depths)):
            regrid_by_depth[d_idx] = area_mean_regrid_2d(
                vals[t_idx, d_idx],
                src_lats,
                src_lons,
                target_grid,
                valid_frac_threshold=0.5,
            )
        out_arr[t_idx] = interpolate_depths_1d(regrid_by_depth, src_depths, depths_m)

    return xr.Dataset(
        data_vars={"grep_mean": (("time", "depth", "lat", "lon"), out_arr)},
        coords={
            "time": time_vals,
            "depth": np.array(depths_m, dtype=np.float64),
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )


def reduce_armor3d(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    depths_m: list[float],
) -> xr.Dataset:
    """Reduce ARMOR3D observation-based 3D temperature to 15 depths and target 0.25° grid."""
    ds = standardize_coords(ds)
    var_name = None
    for cand in ["to", "temperature", "thetao"]:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise KeyError(f"Expected variable ('to') in ARMOR3D dataset, found: {list(ds.data_vars)}")

    da = ds[var_name]
    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    src_depths = np.asarray(ds.coords["depth"].values, dtype=np.float64)
    time_vals = (
        ds.coords["time"].values
        if "time" in ds.coords
        else np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
    )
    if np.ndim(time_vals) == 0:
        time_vals = np.array([time_vals])

    vals = da.values
    if vals.ndim == 3:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_depth = len(depths_m)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    out_arr = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        regrid_by_depth = np.full((len(src_depths), n_lat, n_lon), np.nan, dtype=np.float32)
        for d_idx in range(len(src_depths)):
            regrid_by_depth[d_idx] = area_mean_regrid_2d(
                vals[t_idx, d_idx],
                src_lats,
                src_lons,
                target_grid,
                valid_frac_threshold=0.5,
            )
        out_arr[t_idx] = interpolate_depths_1d(regrid_by_depth, src_depths, depths_m)

    return xr.Dataset(
        data_vars={"armor3d": (("time", "depth", "lat", "lon"), out_arr)},
        coords={
            "time": time_vals,
            "depth": np.array(depths_m, dtype=np.float64),
            "lat": target_grid.lats,
            "lon": target_grid.lons,
        },
    )
