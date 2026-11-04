"""Beach mode: GET /nearest-ocean (PRD_BACKEND §4.17).

The user's location is used only for this request. It is never stored, and logs carry nothing
finer than 1° (here: no coordinates at all).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, Query, Response

import app.data.store as store_module
from app.data import glasses, nearest
from app.errors import APIError, DateUnavailableError
from app.meta.config import beach_available
from app.schemas.models import NearestOceanResponse

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_OUTSIDE_REGION_KM = 500.0


def locate(
    lat: float,
    lon: float,
    bearing_deg: float | None,
    max_km: float,
    min_depth_m: float | None,
    feature: str,
) -> tuple[nearest.OceanHit, str, dict[str, Any]] | None:
    """The ocean cell in view on the latest available date, or None (PRD_BACKEND §4.17).

    Raises 422 `out_of_region` far outside the region and 422 `depth_out_of_range` for a
    `min_depth_m` below the deepest model level. Never logs the location.
    """
    store = store_module.store
    store._ensure_loaded()
    manifest = store.loader.manifest
    assert manifest is not None

    if nearest.km_outside_bbox(lat, lon, manifest["region"]) > MAX_OUTSIDE_REGION_KM:
        logger.info("feature=%s status=422 reason=out_of_region", feature)
        raise APIError(
            422, "out_of_region", "This location is more than 500 km outside the covered region"
        )
    depths = [float(d) for d in manifest["depths_m"]]
    if min_depth_m is not None and min_depth_m > depths[-1] + 1e-9:
        raise APIError(
            422,
            "depth_out_of_range",
            f"min_depth_m must be at most the deepest model level ({depths[-1]:g} m)",
        )

    date = str(manifest["available_dates"]["end"])
    ds = store.get_predictions_dataset()
    try:
        day = ds.sel(time=date)
        coverage = day["coverage"].values.astype(int)
    except KeyError:
        raise DateUnavailableError(f"Date {date} is unavailable in dataset") from None
    lats = np.asarray(ds.coords["lat"].values, dtype=float)
    lons = np.asarray(ds.coords["lon"].values, dtype=float)
    ocean = coverage == 1
    if min_depth_m is not None:
        pred = np.asarray(day["temp_pred"].transpose("depth", "lat", "lon").values, dtype=float)
        ocean = ocean & glasses.deep_enough(pred, depths, min_depth_m)

    if bearing_deg is None:
        hit = nearest.nearest_ocean(lat, lon, max_km, ocean, lats, lons)
    else:
        hit = nearest.ocean_along_bearing(lat, lon, bearing_deg, max_km, ocean, lats, lons)
    return None if hit is None else (hit, date, manifest)


@router.get(
    "/nearest-ocean",
    response_model=NearestOceanResponse,
    tags=["beach"],
    summary="First ocean cell in view from the user's location (beach mode)",
    description=(
        "With a compass bearing, returns the first valid ocean cell at least 5 km away along "
        "that bearing, up to max_km; without one, the nearest such cell. Uses coverage on the "
        "latest available date. With min_depth_m, only cells with a predicted value at that "
        "depth count. The location is not stored or logged."
    ),
)
def get_nearest_ocean(
    response: Response,
    lat: float = Query(..., ge=-90, le=90, description="User latitude"),
    lon: float = Query(..., ge=-180, le=180, description="User longitude"),
    bearing_deg: float | None = Query(
        None, ge=0, lt=360, description="Compass bearing, 0 = north, clockwise"
    ),
    max_km: float = Query(300.0, gt=0, le=500, description="How far to look"),
    min_depth_m: float | None = Query(
        None,
        ge=0,
        description=(
            "Only cells with a predicted value at this depth (water at least this deep); "
            "must not be deeper than the deepest model level"
        ),
    ),
) -> NearestOceanResponse:
    """Find the ocean cell the user is looking at."""
    response.headers["Cache-Control"] = "no-store"
    if not beach_available():
        raise APIError(503, "beach_disabled", "Beach mode is not available right now")
    found = locate(lat, lon, bearing_deg, max_km, min_depth_m, feature="beach")
    if found is None:
        logger.info("feature=beach status=404 bearing=%s", bearing_deg is not None)
        raise APIError(404, "no_ocean_in_view", "No ocean cell found in that direction")
    hit, date, manifest = found
    logger.info("feature=beach status=200 bearing=%s", bearing_deg is not None)
    return NearestOceanResponse(
        lat=hit.lat,
        lon=hit.lon,
        distance_km=hit.distance_km,
        bearing_deg=hit.bearing_deg,
        date=date,
        run_id=manifest["active_run_id"],
        synthetic=manifest["synthetic"],
    )
