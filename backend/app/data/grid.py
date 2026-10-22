"""Grid coordinate snapping, slicing, and area-weighting utilities (PRD_BACKEND §4, §5)."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from app.errors import OutOfRegionError


def round_half_up(val: float) -> int:
    """Deterministic round-half-up on float index (PRD_BACKEND §9 Test 3)."""
    return int(math.floor(val + 0.5))


def snap_coordinate(
    val: float,
    coords: np.ndarray,
    min_bound: float,
    max_bound: float,
    tolerance: float = 1e-4,
) -> tuple[int, float]:
    """Snap a coordinate to the nearest grid cell centre with round-half-up.

    Raises OutOfRegionError if val is outside [min_bound, max_bound].
    Returns (index, snapped_coordinate).
    """
    if val < min_bound - tolerance or val > max_bound + tolerance:
        raise OutOfRegionError(
            f"Coordinate {val} is outside valid region [{min_bound}, {max_bound}]"
        )

    # Resolution from coords
    if len(coords) < 2:
        return 0, float(coords[0])

    res = float(coords[1] - coords[0])
    first_c = float(coords[0])

    frac_idx = (val - first_c) / res
    idx = round_half_up(frac_idx)
    idx = max(0, min(len(coords) - 1, idx))

    return idx, round(float(coords[idx]), 3)


def get_region_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    region: Literal["arabian_sea", "bay_of_bengal", "equatorial"] | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Create 2D boolean mask for a predefined region or bounding box.

    Returns 2D bool array of shape (n_lat, n_lon).
    Raises OutOfRegionError if bounding box has no intersection or 0 cells.
    """
    lat_2d, lon_2d = np.meshgrid(lats, lons, indexing="ij")

    if region == "arabian_sea":
        # Arabian Sea: lon < 78.0 (PRD_AI_ML §6.3)
        return np.asarray(lon_2d < 78.0, dtype=bool)
    elif region == "bay_of_bengal":
        # Bay of Bengal: lon >= 78.0 and lat >= 5.0
        return np.asarray((lon_2d >= 78.0) & (lat_2d >= 5.0), dtype=bool)
    elif region == "equatorial":
        # Equatorial: lat < 5.0
        return np.asarray(lat_2d < 5.0, dtype=bool)

    # Explicit bounding box
    if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
        raise OutOfRegionError("Provide either a valid region name or a full bounding box")

    if lat_min > lat_max or lon_min > lon_max:
        raise OutOfRegionError("Invalid bounding box: min bound exceeds max bound")

    mask = (
        (lat_2d >= lat_min - tolerance)
        & (lat_2d <= lat_max + tolerance)
        & (lon_2d >= lon_min - tolerance)
        & (lon_2d <= lon_max + tolerance)
    )

    if not np.any(mask):
        raise OutOfRegionError(
            f"Bounding box [{lat_min}, {lat_max}, {lon_min}, {lon_max}] contains no grid cells"
        )

    return np.asarray(mask, dtype=bool)


def compute_area_weights(lats: np.ndarray, n_lon: int) -> np.ndarray:
    """Compute 2D cos(lat) area weights for regional aggregation."""
    cos_lats = np.cos(np.radians(lats))
    return np.asarray(np.repeat(cos_lats[:, np.newaxis], n_lon, axis=1), dtype=float)
