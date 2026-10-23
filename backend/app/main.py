import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_v1_router
from app.data.loader import loader
from app.errors import (
    APIError,
    api_error_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initial loading and periodic manifest reload."""
    loader.load()

    reload_task = None
    if settings.RELOAD_INTERVAL_MIN > 0:

        async def periodic_manifest_reload() -> None:
            while True:
                await asyncio.sleep(settings.RELOAD_INTERVAL_MIN * 60)
                try:
                    logger.info("Executing periodic manifest reload check...")
                    loader.load()
                except Exception as e:
                    logger.error("Periodic manifest reload failed: %s", e)

        reload_task = asyncio.create_task(periodic_manifest_reload())

    yield

    if reload_task is not None:
        reload_task.cancel()
        try:
            await reload_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="OceanScope AI API",
    description="Subsurface ocean temperature reconstruction API for the North Indian Ocean",
    version="0.1.0",
    lifespan=lifespan,
)

# Exception handlers enforcing PRD_BACKEND Section 4 JSON error envelope
app.add_exception_handler(APIError, api_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]

# CORS middleware
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
allow_all = "*" in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else origins,
    allow_origin_regex=r"^https://.*\.vercel\.app$" if not allow_all else None,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware for responses > 1 KB
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Mount API v1 router
app.include_router(api_v1_router)
