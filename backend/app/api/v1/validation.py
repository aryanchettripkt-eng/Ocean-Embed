"""Evaluation metrics endpoints: /metrics, /argo, /argo/profiles, /headline."""

from __future__ import annotations

import hashlib
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from app.data.store import store
from app.schemas.models import (
    ArgoProfilesResponse,
    ArgoResponse,
    HeadlineResponse,
    MetricsResponse,
)

router = APIRouter()


def _set_cache_headers(response: Response, run_id: str, request: Request) -> None:
    """Set Cache-Control: public, max-age=86400 and ETag (PRD_BACKEND §4)."""
    response.headers["Cache-Control"] = "public, max-age=86400"
    query_str = str(request.url.query)
    etag_hash = hashlib.sha256(f"{run_id}:{request.url.path}:{query_str}".encode()).hexdigest()[:16]
    response.headers["ETag"] = f'"{etag_hash}"'


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["validation"],
    summary="Model and baseline validation metrics",
    description="Returns precomputed evaluation metrics grouped by depth, season, or region.",
)
def get_metrics(
    request: Request,
    response: Response,
    split: Literal["val", "test"] = Query("val", description="Evaluation split"),
    group_by: Literal["depth", "season", "region"] = Query(
        "depth", description="Grouping criterion"
    ),
) -> MetricsResponse:
    """Query model evaluation metrics."""
    res = store.get_metrics(split=split, group_by=group_by)
    _set_cache_headers(response, res.run_id, request)
    return res


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
def get_argo(
    request: Request,
    response: Response,
) -> ArgoResponse:
    """Query ARGO validation metrics."""
    res = store.get_argo()
    _set_cache_headers(response, res.run_id, request)
    return res


@router.get(
    "/argo/profiles",
    response_model=ArgoProfilesResponse,
    tags=["validation"],
    summary="Matched ARGO float profile coordinates",
    description="Returns coordinates of matched ARGO float profiles for map marker rendering.",
)
def get_argo_profiles(
    request: Request,
    response: Response,
    date: str | None = Query(None, description="Filter profiles by date YYYY-MM-DD"),
    lat_min: float | None = Query(None, description="Filter minimum latitude"),
    lat_max: float | None = Query(None, description="Filter maximum latitude"),
    lon_min: float | None = Query(None, description="Filter minimum longitude"),
    lon_max: float | None = Query(None, description="Filter maximum longitude"),
) -> ArgoProfilesResponse:
    """Query matched ARGO float profile locations."""
    res = store.get_argo_profiles(
        date=date,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    _set_cache_headers(response, res.run_id, request)
    return res


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
def get_headline(
    request: Request,
    response: Response,
) -> HeadlineResponse:
    """Query headline success criterion table."""
    res = store.get_headline()
    _set_cache_headers(response, res.run_id, request)
    return res
