"""Data field query endpoints: map, profile, timeseries, region-summary (PRD_BACKEND §4.3-4.6)."""

from __future__ import annotations

import hashlib
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from app.data.store import store
from app.schemas.models import (
    MapResponse,
    ProfileResponse,
    RegionSummaryResponse,
    TimeseriesResponse,
)

router = APIRouter()


def _set_cache_headers(response: Response, run_id: str, request: Request) -> None:
    """Set Cache-Control: public, max-age=86400 and ETag (PRD_BACKEND §4)."""
    response.headers["Cache-Control"] = "public, max-age=86400"
    query_str = str(request.url.query)
    etag_hash = hashlib.sha256(f"{run_id}:{request.url.path}:{query_str}".encode()).hexdigest()[:16]
    response.headers["ETag"] = f'"{etag_hash}"'


@router.get(
    "/map",
    response_model=MapResponse,
    tags=["data"],
    summary="2D horizontal temperature field slice",
    description="Returns 2D grid slice for a given date, depth, and field (pred, glorys, error).",
)
def get_map(
    request: Request,
    response: Response,
    date: str = Query(..., description="Date formatted as YYYY-MM-DD"),
    depth_m: float | None = Query(None, description="Target depth in meters"),
    depth_index: int | None = Query(None, description="Depth index 0-14"),
    field: Literal["pred", "glorys", "error"] = Query("pred", description="Field to return"),
) -> MapResponse:
    """Query 2D grid slice."""
    res = store.get_map(date=date, depth_m=depth_m, depth_index=depth_index, field=field)
    _set_cache_headers(response, res.run_id, request)
    return res


@router.get(
    "/profile",
    response_model=ProfileResponse,
    tags=["data"],
    summary="Vertical temperature profile at a point",
    description="Returns 15-depth vertical temperature profile for a given lat, lon, and date.",
)
def get_profile(
    request: Request,
    response: Response,
    lat: float = Query(..., description="Latitude in degrees north"),
    lon: float = Query(..., description="Longitude in degrees east"),
    date: str = Query(..., description="Date formatted as YYYY-MM-DD"),
) -> ProfileResponse:
    """Query vertical temperature profile."""
    res = store.get_profile(lat=lat, lon=lon, date=date)
    _set_cache_headers(response, res.run_id, request)
    return res


@router.get(
    "/timeseries",
    response_model=TimeseriesResponse,
    tags=["data"],
    summary="Time series of temperature at a point and depth",
    description="Returns daily time series for up to 366 days at a specific grid cell and depth.",
)
def get_timeseries(
    request: Request,
    response: Response,
    lat: float = Query(..., description="Latitude in degrees north"),
    lon: float = Query(..., description="Longitude in degrees east"),
    depth_m: float | None = Query(None, description="Target depth in meters"),
    depth_index: int | None = Query(None, description="Depth index 0-14"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
) -> TimeseriesResponse:
    """Query time series at a location."""
    res = store.get_timeseries(
        lat=lat,
        lon=lon,
        depth_m=depth_m,
        depth_index=depth_index,
        start=start,
        end=end,
    )
    _set_cache_headers(response, res.run_id, request)
    return res


@router.get(
    "/region-summary",
    response_model=RegionSummaryResponse,
    tags=["data"],
    summary="Area-weighted regional summary statistics",
    description="Computes area-weighted (cos(lat)) mean, min, max, std over a sub-region or bbox.",
)
def get_region_summary(
    request: Request,
    response: Response,
    date: str = Query(..., description="Date formatted as YYYY-MM-DD"),
    depth_m: float | None = Query(None, description="Target depth in meters"),
    depth_index: int | None = Query(None, description="Depth index 0-14"),
    lat_min: float | None = Query(None, description="Bbox minimum latitude"),
    lat_max: float | None = Query(None, description="Bbox maximum latitude"),
    lon_min: float | None = Query(None, description="Bbox minimum longitude"),
    lon_max: float | None = Query(None, description="Bbox maximum longitude"),
    region: Literal["arabian_sea", "bay_of_bengal", "equatorial"] | None = Query(
        None, description="Predefined region name"
    ),
    field: Literal["pred", "glorys", "error"] = Query("pred", description="Field to summarize"),
) -> RegionSummaryResponse:
    """Compute regional aggregate statistics."""
    res = store.get_region_summary(
        date=date,
        depth_m=depth_m,
        depth_index=depth_index,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        region=region,
        field=field,
    )
    _set_cache_headers(response, res.run_id, request)
    return res
