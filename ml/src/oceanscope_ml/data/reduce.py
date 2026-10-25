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
        lon_min: float = 40.0,
        lon_max: float = 100.0,
        lat_min: float = 0.0,
        lat_max: float = 30.0,
        resolution_deg: float = 0.25,
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
    """Standardize dimension and coordinate names to (time, depth, lat, lon)."""
    # 1. Check if coords exist that need to be swapped with dims (e.g. lon on longitude dim)
    swap_map: dict[str, str] = {}
    if hasattr(ds, "dims") and hasattr(ds, "coords"):
        for dim in ds.dims:
            d_str = str(dim)
            d_lower = d_str.lower()
            if d_lower in ["longitude", "nav_lon"] and "lon" in ds.coords and d_str != "lon":
                swap_map[d_str] = "lon"
            elif d_lower in ["latitude", "nav_lat"] and "lat" in ds.coords and d_str != "lat":
                swap_map[d_str] = "lat"
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
    if swap_map:
        try:
            ds = ds.swap_dims(swap_map)
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

    return ds


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
        # Interpolate missing interior target cells using regular grid interpolator
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
    """Interpolate 3D array (depth, lat, lon) along depth axis without extrapolation."""
    tgt_depth_arr = np.array(target_depths, dtype=np.float64)
    n_tgt = len(tgt_depth_arr)
    _, n_lat, n_lon = values_by_depth.shape

    # Fast path if source depths exactly match target depths
    if len(src_depths) == n_tgt and np.allclose(src_depths, tgt_depth_arr, atol=0.01):
        return values_by_depth.astype(np.float32)

    # 1D vertical interpolation per spatial cell
    out = np.full((n_tgt, n_lat, n_lon), np.nan, dtype=np.float32)
    # Clip valid range: no extrapolation beyond min/max native depth
    min_d, max_d = float(np.min(src_depths)), float(np.max(src_depths))
    valid_tgt_mask = (tgt_depth_arr >= min_d) & (tgt_depth_arr <= max_d)

    interp_fn = scipy.interpolate.interp1d(
        src_depths,
        values_by_depth,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interpolated = interp_fn(tgt_depth_arr[valid_tgt_mask])
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
    var_name = "thetao" if "thetao" in ds.data_vars else list(ds.data_vars.keys())[0]
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
    n_time = len(time_vals)
    n_depth = len(depths_m)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    # 4D output array: (time, depth, lat, lon)
    out_arr = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    # Reshape source to (time, depth, lat, lon)
    vals = da.values
    if vals.ndim == 3:  # (depth, lat, lon)
        vals = vals[np.newaxis, ...]
    elif vals.ndim == 2:  # (lat, lon)
        vals = vals[np.newaxis, np.newaxis, ...]

    for t_idx in range(n_time):
        t_slice = vals[t_idx]  # (src_depth, src_lat, src_lon)
        # Regrid each native depth layer spatially first
        regrid_by_depth = np.full((len(src_depths), n_lat, n_lon), np.nan, dtype=np.float32)
        for d_idx in range(len(src_depths)):
            regrid_by_depth[d_idx] = area_mean_regrid_2d(
                t_slice[d_idx],
                src_lats,
                src_lons,
                target_grid,
                valid_frac_threshold=0.5,
            )
        # Vertically interpolate/select the 15 configured depths
        out_arr[t_idx] = interpolate_depths_1d(regrid_by_depth, src_depths, depths_m)

    # Derive ocean_mask: 1 where GLORYS surface level is valid (PRD §5.1, §6.1)
    # Check over all time slices or the first time slice
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
    var_name = "sla" if "sla" in ds.data_vars else list(ds.data_vars.keys())[0]
    da = ds[var_name]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    if target_time is not None and "time" in da.coords:
        try:
            da = da.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

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

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        t_val = vals[t_idx] if t_idx < vals.shape[0] else vals[-1]
        out_arr[t_idx] = area_mean_regrid_2d(
            t_val,
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
    var_name = "adt" if "adt" in ds.data_vars else list(ds.data_vars.keys())[0]
    da = ds[var_name]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    if target_time is not None and "time" in da.coords:
        try:
            da = da.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

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

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        t_val = vals[t_idx] if t_idx < vals.shape[0] else vals[-1]
        out_arr[t_idx] = area_mean_regrid_2d(
            t_val,
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
    var_name = (
        "analysed_sst"
        if "analysed_sst" in ds.data_vars
        else ("sst" if "sst" in ds.data_vars else list(ds.data_vars.keys())[0])
    )
    da = ds[var_name]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    if target_time is not None and "time" in da.coords:
        try:
            da = da.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

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

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    # Convert Kelvin to Celsius if necessary (PRD_AI_ML §3)
    sample_valid = vals[~np.isnan(vals)]
    if len(sample_valid) > 0 and np.mean(sample_valid[:1000]) > 100.0:
        vals = vals - 273.15

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        t_val = vals[t_idx] if t_idx < vals.shape[0] else vals[-1]
        out_arr[t_idx] = area_mean_regrid_2d(
            t_val,
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
    var_name = (
        "sos"
        if "sos" in ds.data_vars
        else ("sss" if "sss" in ds.data_vars else list(ds.data_vars.keys())[0])
    )
    da = ds[var_name]
    if "depth" in da.dims:
        da = da.squeeze("depth")

    if target_time is not None and "time" in da.coords:
        try:
            da = da.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

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

    vals = da.values
    while vals.ndim > 3:
        vals = np.squeeze(vals, axis=1)
    if vals.ndim == 2:
        vals = vals[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    out_arr = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        t_val = vals[t_idx] if t_idx < vals.shape[0] else vals[-1]
        out_arr[t_idx] = area_mean_regrid_2d(
            t_val,
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
    u_name = "u" if "u" in ds.data_vars else ("u_current" if "u_current" in ds.data_vars else None)
    v_name = "v" if "v" in ds.data_vars else ("v_current" if "v_current" in ds.data_vars else None)

    if u_name is None or v_name is None:
        keys = list(ds.data_vars.keys())
        u_name = str(keys[0])
        v_name = str(keys[1]) if len(keys) > 1 else str(keys[0])

    da_u = ds[u_name]
    da_v = ds[v_name]
    if "depth" in da_u.dims:
        da_u = da_u.squeeze("depth")
        da_v = da_v.squeeze("depth")

    if target_time is not None and "time" in da_u.coords:
        try:
            da_u = da_u.sel(time=pd.to_datetime(target_time), method="nearest")
            da_v = da_v.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

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

    out_u = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    out_v = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        u_t = u_vals[t_idx] if t_idx < u_vals.shape[0] else u_vals[-1]
        v_t = v_vals[t_idx] if t_idx < v_vals.shape[0] else v_vals[-1]
        out_u[t_idx] = area_mean_regrid_2d(u_t, src_lats, src_lons, target_grid)
        out_v[t_idx] = area_mean_regrid_2d(v_t, src_lats, src_lons, target_grid)

    return xr.Dataset(
        data_vars={
            "u_current": (("time", "lat", "lon"), out_u),
            "v_current": (("time", "lat", "lon"), out_v),
        },
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def compute_wind_stress_and_curl(
    u10: np.ndarray,
    v10: np.ndarray,
    target_grid: TargetGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute wind stress (tau_x, tau_y) and spherical curl using Large & Pond (1981).

    Parameters
    ----------
    u10, v10 : np.ndarray, shape (n_time, n_lat, n_lon)
    target_grid : TargetGrid

    Returns
    -------
    tuple of (u_wind, v_wind, wind_stress_curl), all shape (n_time, n_lat, n_lon)
    """
    n_time, n_lat, n_lon = u10.shape
    curl_out = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    # Wind speed
    speed = np.sqrt(u10**2 + v10**2)

    # Drag coefficient Cd per Large & Pond (1981)
    cd = np.where(
        speed < 11.0,
        1.2e-3,
        np.where(
            speed <= 25.0,
            (0.49 + 0.065 * speed) * 1e-3,
            (0.49 + 0.065 * 25.0) * 1e-3,
        ),
    )

    # Surface wind stress vector tau = rho_air * Cd * |U| * U
    tau_x = (AIR_DENSITY_KG_M3 * cd * speed * u10).astype(np.float32)
    tau_y = (AIR_DENSITY_KG_M3 * cd * speed * v10).astype(np.float32)

    # Spherical coordinates curl(tau) = 1/(R cos phi) * d(tau_y)/d_lambda - 1/R * d(tau_x)/d_phi
    lat_rad = np.radians(target_grid.lats)  # (n_lat,)
    cos_lat = np.cos(lat_rad)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-4, 1e-4, cos_lat)  # avoid div by zero at poles

    delta_lat_rad = np.radians(target_grid.resolution_deg)
    delta_lon_rad = np.radians(target_grid.resolution_deg)

    for t_idx in range(n_time):
        tx = tau_x[t_idx]  # (n_lat, n_lon)
        ty = tau_y[t_idx]

        # Centred differences on interior points
        d_ty_d_lon = (ty[:, 2:] - ty[:, :-2]) / (2.0 * delta_lon_rad)  # (n_lat, n_lon-2)
        d_tx_d_lat = (tx[2:, :] - tx[:-2, :]) / (2.0 * delta_lat_rad)  # (n_lat-2, n_lon)

        # Interior region (1:-1, 1:-1)
        term1 = d_ty_d_lon[1:-1, :] / (EARTH_RADIUS_M * cos_lat_safe[1:-1, np.newaxis])
        term2 = d_tx_d_lat[:, 1:-1] / EARTH_RADIUS_M

        curl_interior = term1 - term2
        curl_out[t_idx, 1:-1, 1:-1] = curl_interior.astype(np.float32)

    return u10.astype(np.float32), v10.astype(np.float32), curl_out


def reduce_era5_winds(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    target_time: np.ndarray | None = None,
) -> xr.Dataset:
    """Reduce ERA5 10m winds to 0.25° grid, computing stress and curl (PRD_AI_ML §3)."""
    ds = standardize_coords(ds)
    if target_time is not None and "time" in ds.coords:
        try:
            t_min = pd.to_datetime(target_time).min() - pd.Timedelta(days=1)
            t_max = pd.to_datetime(target_time).max() + pd.Timedelta(days=1)
            ds = ds.sel(time=slice(t_min, t_max))
        except Exception:
            pass

    if "time" in ds.coords and len(ds.coords["time"]) > 60:
        try:
            ds = ds.resample(time="1D").mean()
        except Exception:
            pass

    if target_time is not None and "time" in ds.coords:
        try:
            ds = ds.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

    u_var = (
        "u10"
        if "u10" in ds.data_vars
        else ("10m_u_component_of_wind" if "10m_u_component_of_wind" in ds.data_vars else "u_wind")
    )
    v_var = (
        "v10"
        if "v10" in ds.data_vars
        else ("10m_v_component_of_wind" if "10m_v_component_of_wind" in ds.data_vars else "v_wind")
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

    u_data = ds[u_var].values
    v_data = ds[v_var].values
    while u_data.ndim > 3:
        u_data = np.squeeze(u_data, axis=1)
        v_data = np.squeeze(v_data, axis=1)
    if u_data.ndim == 2:
        u_data = u_data[np.newaxis, ...]
        v_data = v_data[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    regrid_u = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    regrid_v = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        u_t = u_data[t_idx] if t_idx < u_data.shape[0] else u_data[-1]
        v_t = v_data[t_idx] if t_idx < v_data.shape[0] else v_data[-1]
        regrid_u[t_idx] = area_mean_regrid_2d(u_t, src_lats, src_lons, target_grid)
        regrid_v[t_idx] = area_mean_regrid_2d(v_t, src_lats, src_lons, target_grid)

    u_wind, v_wind, curl = compute_wind_stress_and_curl(regrid_u, regrid_v, target_grid)

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
    """Reduce ERA5 net heat flux and precipitation to 0.25° target grid."""
    ds = standardize_coords(ds)
    if target_time is not None and "time" in ds.coords:
        try:
            t_min = pd.to_datetime(target_time).min() - pd.Timedelta(days=1)
            t_max = pd.to_datetime(target_time).max() + pd.Timedelta(days=1)
            ds = ds.sel(time=slice(t_min, t_max))
        except Exception:
            pass

    if "time" in ds.coords and len(ds.coords["time"]) > 60:
        try:
            ds = ds.resample(time="1D").mean()
        except Exception:
            pass

    if target_time is not None and "time" in ds.coords:
        try:
            ds = ds.sel(time=pd.to_datetime(target_time), method="nearest")
        except Exception:
            pass

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

    # Net heat flux Q_net = ssr + str + sshf + slhf
    has_components = all(k in ds.data_vars for k in ["ssr", "str", "sshf", "slhf"])
    if has_components:
        heat_flux = (
            ds["ssr"].values + ds["str"].values + ds["sshf"].values + ds["slhf"].values
        ).astype(np.float32)
    elif "net_heat_flux" in ds.data_vars:
        heat_flux = ds["net_heat_flux"].values.astype(np.float32)
    else:
        heat_flux = np.zeros_like(ds[list(ds.data_vars.keys())[0]].values, dtype=np.float32)

    tp_var = (
        "tp"
        if "tp" in ds.data_vars
        else ("precip" if "precip" in ds.data_vars else list(ds.data_vars.keys())[0])
    )
    precip = ds[tp_var].values.astype(np.float32)

    while heat_flux.ndim > 3:
        heat_flux = np.squeeze(heat_flux, axis=1)
        precip = np.squeeze(precip, axis=1)
    if heat_flux.ndim == 2:
        heat_flux = heat_flux[np.newaxis, ...]
        precip = precip[np.newaxis, ...]

    n_time = len(time_vals)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    out_hf = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    out_tp = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    for t_idx in range(n_time):
        hf_t = heat_flux[t_idx] if t_idx < heat_flux.shape[0] else heat_flux[-1]
        tp_t = precip[t_idx] if t_idx < precip.shape[0] else precip[-1]
        out_hf[t_idx] = area_mean_regrid_2d(hf_t, src_lats, src_lons, target_grid)
        out_tp[t_idx] = area_mean_regrid_2d(tp_t, src_lats, src_lons, target_grid)

    return xr.Dataset(
        data_vars={
            "net_heat_flux": (("time", "lat", "lon"), out_hf),
            "precip": (("time", "lat", "lon"), out_tp),
        },
        coords={"time": time_vals, "lat": target_grid.lats, "lon": target_grid.lons},
    )


def reduce_gebco(ds: xr.Dataset, target_grid: TargetGrid) -> xr.Dataset:
    """Reduce GEBCO 15 arc-sec bathymetry to 0.25° grid, storing log(1 + depth)."""
    ds = standardize_coords(ds)
    var_name = (
        "elevation"
        if "elevation" in ds.data_vars
        else ("z" if "z" in ds.data_vars else list(ds.data_vars.keys())[0])
    )
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
) -> xr.Dataset:
    """Reduce World Ocean Atlas (WOA23) monthly climatology to 15 depths and 0.25° grid."""
    ds = standardize_coords(ds)
    var_name = (
        "t_an"
        if "t_an" in ds.data_vars
        else ("temperature" if "temperature" in ds.data_vars else list(ds.data_vars.keys())[0])
    )
    da = ds[var_name]

    src_lats = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    src_lons = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    src_depths = np.asarray(ds.coords["depth"].values, dtype=np.float64)

    vals = da.values  # could be (depth, lat, lon) or (time/month, depth, lat, lon)
    if vals.ndim == 3:
        vals = vals[np.newaxis, ...]

    n_slices = vals.shape[0]
    n_depth = len(depths_m)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)

    out_arr = np.full((n_slices, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

    for s_idx in range(n_slices):
        regrid_by_depth = np.full((len(src_depths), n_lat, n_lon), np.nan, dtype=np.float32)
        for d_idx in range(len(src_depths)):
            regrid_by_depth[d_idx] = area_mean_regrid_2d(
                vals[s_idx, d_idx],
                src_lats,
                src_lons,
                target_grid,
            )
        out_arr[s_idx] = interpolate_depths_1d(regrid_by_depth, src_depths, depths_m)

    coords: dict[str, Any] = {
        "depth": np.array(depths_m, dtype=np.float64),
        "lat": target_grid.lats,
        "lon": target_grid.lons,
    }
    if "time" in ds.coords:
        coords["time"] = ds.coords["time"].values
        dims = ("time", "depth", "lat", "lon")
    elif "month" in ds.coords:
        coords["month"] = ds.coords["month"].values
        dims = ("month", "depth", "lat", "lon")
    else:
        coords["month"] = np.arange(1, n_slices + 1)
        dims = ("month", "depth", "lat", "lon")

    return xr.Dataset(
        data_vars={"woa_clim_profile": (dims, out_arr)},
        coords=coords,
    )


def reduce_grep(
    ds: xr.Dataset,
    target_grid: TargetGrid,
    depths_m: list[float],
) -> xr.Dataset:
    """Reduce GREP ensemble mean thetao_mean to 15 depths and target 0.25° grid."""
    ds = standardize_coords(ds)
    var_name = "thetao_mean" if "thetao_mean" in ds.data_vars else list(ds.data_vars.keys())[0]
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
