"""OceanScope API FastAPI application entry point."""

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.settings import settings

app = FastAPI(
    title="OceanScope AI API",
    description="Subsurface ocean temperature reconstruction API for the North Indian Ocean",
    version="0.1.0",
)

origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "run_id": "initial_skeleton",
        "data_loaded": True,
    }


@app.get("/api/v1/health", tags=["system"])
def api_v1_health() -> dict[str, Any]:
    """Base path health check endpoint."""
    return health()
