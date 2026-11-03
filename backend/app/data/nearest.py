"""Beach mode geometry: the first ocean cell along a compass bearing (PRD_BACKEND §4.17).

Pure functions over the coverage grid. Coordinates are used only for the calculation and are
never logged or stored here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.data.grid import round_half_up

EARTH_RADIUS_KM = 6371.0088
MIN_OFFSHORE_KM = 5.0
RAY_STEP_KM = 2.0


@dataclass(frozen=True)
class OceanHit:
    lat_index: int
    lon_index: int
    lat: float
    lon: float
    distance_km: float
    bearing_deg: float


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray | float, lon2: np.ndarray | float):  # type: ignore[no-untyped-def]
    """Great-circle distance in km (vectorised over the second point)."""
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - math.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2, 0 = north, clockwise, in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def destination(
    lat: float, lon: float, bearing_deg: float, distance_km: float
) -> tuple[float, float]:
    """Point reached from (lat, lon) after `distance_km` along a great circle."""
    d = distance_km / EARTH_RADIUS_KM
    b = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2)
    )
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def km_outside_bbox(lat: float, lon: float, bbox: dict[str, float]) -> float:
    """Distance from the point to the nearest point of the bbox (0 inside)."""
    clat = min(max(lat, bbox["lat_min"]), bbox["lat_max"])
    clon = min(max(lon, bbox["lon_min"]), bbox["lon_max"])
    return float(haversine_km(lat, lon, clat, clon))


def _cell_index(val: float, coords: np.ndarray) -> int | None:
    res = float(coords[1] - coords[0])
    idx = round_half_up((val - float(coords[0])) / res)
    if idx < 0 or idx >= len(coords) or abs(val - float(coords[idx])) > abs(res) / 2 + 1e-9:
        return None
    return idx


def _hit(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray, r: int, c: int) -> OceanHit:
    clat, clon = float(lats[r]), float(lons[c])
    return OceanHit(
        lat_index=r,
        lon_index=c,
        lat=round(clat, 3),
        lon=round(clon, 3),
        distance_km=round(float(haversine_km(lat, lon, clat, clon)), 1),
        bearing_deg=round(initial_bearing_deg(lat, lon, clat, clon), 1),
    )


def ocean_along_bearing(
    lat: float,
    lon: float,
    bearing_deg: float,
    max_km: float,
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> OceanHit | None:
    """First ocean cell whose centre is at least 5 km away, walking the ray in 2 km steps."""
    steps = int(math.floor((max_km - MIN_OFFSHORE_KM) / RAY_STEP_KM)) + 1
    for k in range(max(steps, 0)):
        dist = MIN_OFFSHORE_KM + k * RAY_STEP_KM
        plat, plon = destination(lat, lon, bearing_deg, dist)
        r, c = _cell_index(plat, lats), _cell_index(plon, lons)
        if r is None or c is None or not ocean[r, c]:
            continue
        hit = _hit(lat, lon, lats, lons, r, c)
        if MIN_OFFSHORE_KM <= hit.distance_km <= max_km:
            return hit
    return None


def nearest_ocean(
    lat: float, lon: float, max_km: float, ocean: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> OceanHit | None:
    """Closest ocean cell centre at least 5 km away and within `max_km`."""
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    dist = np.asarray(haversine_km(lat, lon, lat2d, lon2d), dtype=float)
    ok = ocean & (dist >= MIN_OFFSHORE_KM) & (dist <= max_km)
    if not ok.any():
        return None
    masked = np.where(ok, dist, np.inf)
    r, c = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return _hit(lat, lon, lats, lons, int(r), int(c))
