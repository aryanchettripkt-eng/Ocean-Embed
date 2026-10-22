"""OceanScope AI API FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_v1_router
from app.api.v1.meta import get_health, get_meta
from app.errors import (
    APIError,
    api_error_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.schemas.models import MetaResponse
from app.settings import settings

app = FastAPI(
    title="OceanScope AI API",
    description="Subsurface ocean temperature reconstruction API for the North Indian Ocean",
    version="0.1.0",
)

# Exception handlers enforcing PRD_BACKEND Section 4 JSON error envelope
app.add_exception_handler(APIError, api_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]

# CORS middleware
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware for responses > 1 KB
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Mount API v1 router
app.include_router(api_v1_router)


# Root level endpoints for health and meta
@app.get("/health", tags=["system"], summary="Root health check")
def health() -> JSONResponse:
    """Convenience root health check delegating to /api/v1/health."""
    return get_health()


@app.get("/meta", response_model=MetaResponse, tags=["metadata"], summary="Root metadata")
def meta(response: Response) -> MetaResponse:
    """Convenience root metadata delegating to /api/v1/meta."""
    return get_meta(response)
