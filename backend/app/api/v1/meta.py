"""System and dataset metadata endpoints (PRD_BACKEND §4.1-4.2)."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.data.loader import loader
from app.data.store import store
from app.errors import DataNotLoadedError
from app.schemas.models import HealthResponse, MetaResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description="Returns 200 OK and active run_id when data is loaded, 503 otherwise.",
)
def get_health() -> HealthResponse:
    """Check backend operational health and dataset loading status."""
    if loader.is_loaded:
        return HealthResponse(
            status="ok",
            run_id=loader.active_run_id,
            data_loaded=True,
        )
    raise DataNotLoadedError(loader.load_error or "Data store is not loaded")


@router.get(
    "/meta",
    response_model=MetaResponse,
    tags=["metadata"],
    summary="Dataset and model metadata",
    description=(
        "Returns active run metadata, coordinate ranges, depths, "
        "color scales, scientific caveats, and chat assistant configuration."
    ),
)
def get_meta(response: Response) -> MetaResponse:
    """Retrieve complete metadata for the active model run and prediction store."""
    meta = store.get_meta()
    # Cache-Control: public, max-age=300 (PRD_BACKEND Section 4)
    response.headers["Cache-Control"] = "public, max-age=300"
    return meta
