"""Curated events endpoint: GET /events for the time machine (PRD_BACKEND §4.14)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response

import app.data.store as store_module
from app.data.events import EventsFileError, load_events, validate_events, visible_events
from app.errors import APIError
from app.schemas.models import EventItem, EventsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/events",
    response_model=EventsResponse,
    tags=["events"],
    summary="Curated ocean events for the time machine",
    description=(
        "Returns the committed, human-approved event list. Events with a date in the sealed test "
        "period are hidden until the owner runs the final test. Suggested depths are snapped to "
        "model depths. Contains no model numbers."
    ),
)
def get_events(response: Response) -> EventsResponse:
    """List the curated events that the time machine may show."""
    store_module.store._ensure_loaded()
    manifest = store_module.store.loader.manifest
    assert manifest is not None
    try:
        cleaned = validate_events(
            load_events(),
            region=manifest["region"],
            depths_m=[float(d) for d in manifest["depths_m"]],
            available=manifest["available_dates"],
        )
    except EventsFileError:
        logger.error("feature=events status=503 reason=invalid_events_file")
        raise APIError(
            503, "events_unavailable", "The event list is unavailable right now"
        ) from None
    visible = visible_events(
        cleaned,
        test_range=manifest["splits"]["test"],
        test_evaluated=bool(manifest.get("test_evaluated", False)),
    )
    # Data endpoint: cacheable while the inputs are the same (PRD_BACKEND §4.12).
    response.headers["Cache-Control"] = "public, max-age=300"
    return EventsResponse(
        events=[EventItem(**e) for e in visible],
        run_id=manifest["active_run_id"],
        synthetic=manifest["synthetic"],
    )
