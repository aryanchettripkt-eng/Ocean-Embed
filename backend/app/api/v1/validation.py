"""Evaluation metrics endpoints: /metrics, /argo, /argo/profiles, /headline."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.models import (
    ArgoProfilesResponse,
    ArgoResponse,
    HeadlineResponse,
    MetricsResponse,
)

router = APIRouter()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["validation"],
    summary="Model and baseline validation metrics",
    description="Returns precomputed evaluation metrics grouped by depth, season, or region.",
)
def get_metrics(
    split: Literal["val", "test"] = Query("val", description="Evaluation split"),
    group_by: Literal["depth", "season", "region"] = Query(
        "depth", description="Grouping criterion"
    ),
) -> MetricsResponse:
    """Query model evaluation metrics."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B3",
    )


@router.get(
    "/argo",
    response_model=ArgoResponse,
    tags=["validation"],
    summary="Independent ARGO float matchup metrics",
    description=(
        "Returns held-out float validation metrics comparing model "
        "and competitors against ARGO observations."
    ),
)
def get_argo() -> ArgoResponse:
    """Query ARGO validation metrics."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B3",
    )


@router.get(
    "/argo/profiles",
    response_model=ArgoProfilesResponse,
    tags=["validation"],
    summary="Matched ARGO float profile coordinates",
    description="Returns coordinates of matched ARGO float profiles for map marker rendering.",
)
def get_argo_profiles(
    date: str | None = Query(None, description="Filter profiles by date YYYY-MM-DD"),
    lat_min: float | None = Query(None, description="Filter minimum latitude"),
    lat_max: float | None = Query(None, description="Filter maximum latitude"),
    lon_min: float | None = Query(None, description="Filter minimum longitude"),
    lon_max: float | None = Query(None, description="Filter maximum longitude"),
) -> ArgoProfilesResponse:
    """Query matched ARGO float profile locations."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B3",
    )


@router.get(
    "/headline",
    response_model=HeadlineResponse,
    tags=["validation"],
    summary="Pre-registered primary success criterion table",
    description=(
        "Returns the pre-registered headline comparison in the Bay of Bengal "
        "50-300m with float-bootstrap CIs."
    ),
)
def get_headline() -> HeadlineResponse:
    """Query headline success criterion table."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B3",
    )
