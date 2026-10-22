"""OceanScope API FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.data.loader import loader
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
def health() -> JSONResponse:
    """Health check endpoint returning 200 when data is loaded, 503 otherwise."""
    if loader.is_loaded:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "run_id": loader.active_run_id,
                "data_loaded": True,
            },
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "run_id": None,
            "data_loaded": False,
            "detail": loader.load_error,
        },
    )


@app.get("/api/v1/health", tags=["system"])
def api_v1_health() -> JSONResponse:
    """Base path health check endpoint."""
    return health()
