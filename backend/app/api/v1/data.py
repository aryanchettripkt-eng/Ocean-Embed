"""Data field query endpoints: map, profile, timeseries, region-summary (PRD_BACKEND §4.3-4.6)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.models import (
    MapResponse,
    ProfileResponse,
    RegionSummaryResponse,
    TimeseriesResponse,
)

router = APIRouter()


@router.get(
    "/map",
    response_model=MapResponse,
    tags=["data"],
    summary="2D horizontal temperature field slice",
    description="Returns 2D grid slice for a given date, depth, and field (pred, glorys, error).",
)
def get_map(
    date: str = Query(..., description="Date formatted as YYYY-MM-DD"),
    depth_m: float | None = Query(None, description="Target depth in meters"),
    depth_index: int | None = Query(None, description="Depth index 0-14"),
    field: Literal["pred", "glorys", "error"] = Query("pred", description="Field to return"),
) -> MapResponse:
    """Query 2D grid slice."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B2",
    )


@router.get(
    "/profile",
    response_model=ProfileResponse,
    tags=["data"],
    summary="Vertical temperature profile at a point",
    description="Returns 15-depth vertical temperature profile for a given lat, lon, and date.",
)
def get_profile(
    lat: float = Query(..., description="Latitude in degrees north"),
    lon: float = Query(..., description="Longitude in degrees east"),
    date: str = Query(..., description="Date formatted as YYYY-MM-DD"),
) -> ProfileResponse:
    """Query vertical temperature profile."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B2",
    )


@router.get(
    "/timeseries",
    response_model=TimeseriesResponse,
    tags=["data"],
    summary="Time series of temperature at a point and depth",
    description="Returns daily time series for up to 366 days at a specific grid cell and depth.",
)
def get_timeseries(
    lat: float = Query(..., description="Latitude in degrees north"),
    lon: float = Query(..., description="Longitude in degrees east"),
    depth_m: float | None = Query(None, description="Target depth in meters"),
    depth_index: int | None = Query(None, description="Depth index 0-14"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
) -> TimeseriesResponse:
    """Query time series at a location."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B2",
    )


@router.get(
    "/region-summary",
    response_model=RegionSummaryResponse,
    tags=["data"],
    summary="Area-weighted regional summary statistics",
    description=(
        "Computes area-weighted (cos(lat)) mean, min, max, std over a sub-region or bounding box."
    ),
)
def get_region_summary(
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
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B2",
    )
