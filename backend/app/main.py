import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_v1_router, root_router
from app.data.events import EventsFileError, check_events_file
from app.data.loader import loader
from app.errors import (
    APIError,
    api_error_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.meta.budget import flush_budget
from app.meta.client import warm_up_chat
from app.settings import settings

logger = logging.getLogger(__name__)

# Paths whose query string carries a user's location (PRD_BACKEND §4.17). The access log must
# not record it at better than 1°, so the query is dropped from those lines.
_LOCATION_PATHS = ("/nearest-ocean",)


class RedactLocationFilter(logging.Filter):
    """Removes the query string from uvicorn access-log lines for location endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path, sep, _ = args[2].partition("?")
            if sep and path.endswith(_LOCATION_PATHS):
                record.args = (*args[:2], f"{path}?<redacted>", *args[3:])
        return True


logging.getLogger("uvicorn.access").addFilter(RedactLocationFilter())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initial loading and periodic manifest reload.

    The first load (which may download the dataset from HF) runs in a worker thread, so
    uvicorn binds its port at once. Render fails a deploy whose port is not open within
    its scan window. Until the load finishes, /health and /meta return 503.
    """

    async def load_then_reload() -> None:
        try:
            await asyncio.to_thread(loader.load)
        except Exception as e:
            logger.error("Initial data load failed: %s", e)
        else:
            # Check the curated events file against the manifest (PRD_BACKEND §4.14). The load
            # now runs after the port is bound, so a bad file cannot stop startup any more; it
            # is logged here and /events answers 503 events_unavailable until it is fixed.
            if loader.manifest is not None:
                try:
                    check_events_file(loader.manifest)
                except EventsFileError as e:
                    logger.error("events.json is invalid: %s", e)
        if settings.RELOAD_INTERVAL_MIN <= 0:
            return
        while True:
            await asyncio.sleep(settings.RELOAD_INTERVAL_MIN * 60)
            try:
                logger.info("Executing periodic manifest reload check...")
                await asyncio.to_thread(loader.load)
            except Exception as e:
                logger.error("Periodic manifest reload failed: %s", e)

    load_task = asyncio.create_task(load_then_reload())
    # Wake the Meta chat model in the background so the first guide question isn't slow.
    warmup_task = asyncio.create_task(asyncio.to_thread(warm_up_chat))

    yield

    load_task.cancel()
    warmup_task.cancel()
    for task in (load_task, warmup_task):
        try:
            await task
        except asyncio.CancelledError:
            pass
    # Save any spend not yet pushed to the budget ledger (app/meta/budget.py).
    try:
        await asyncio.to_thread(flush_budget)
    except Exception as e:
        logger.error("Budget ledger flush failed: %s", e)


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
    # The API uses no cookies or auth headers, and the frontend never sends credentials.
    # Allowing them would let any *.vercel.app page (the regex above) make credentialed calls.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Response:
    """Headers for every response: no MIME sniffing (postcard bytes come from an upstream
    model), no framing of API responses, and no Referer leaking query strings."""
    response: Response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


# GZip middleware for responses > 1 KB
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Mount API router at /api/v1 (canonical schema) and at / (convenience fallback)
app.include_router(api_v1_router)
app.include_router(root_router, include_in_schema=False)
