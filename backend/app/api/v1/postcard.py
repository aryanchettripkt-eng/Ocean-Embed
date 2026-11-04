"""Ocean postcards: POST /postcard and GET /postcard/<hash>.webp (PRD_BACKEND §4.15)."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

import app.data.store as store_module
from app.data.events import EventsFileError, load_events, validate_events, visible_events
from app.errors import APIError
from app.meta import client as client_module
from app.meta import postcard as postcard_module
from app.meta.client import MetaUpstreamError
from app.meta.config import postcard_available, postcard_provider_label
from app.meta.limits import FeatureLimiter, client_ip
from app.meta.prompts import POSTCARD_PROMPT_VERSION
from app.schemas.caveats import POSTCARD_CAVEAT
from app.schemas.models import PostcardCaption, PostcardRequest, PostcardResponse
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

TIMEOUT_S = 45.0
CAPTION_SOURCE = "Model prediction"

_limiter: FeatureLimiter | None = None


def get_limiter() -> FeatureLimiter:
    global _limiter
    if _limiter is None:
        _limiter = FeatureLimiter(
            settings.POSTCARD_RATE_LIMIT,
            settings.POSTCARD_DAILY_CAP,
            settings.POSTCARD_IP_DAILY_CAP,
        )
    return _limiter


def _require_enabled() -> None:
    if not postcard_available():
        raise APIError(503, "postcard_disabled", "Postcards are not available right now")


def _find_event(event_id: str, manifest: dict[str, object]) -> dict[str, object]:
    try:
        events = validate_events(
            load_events(),
            region=manifest["region"],  # type: ignore[arg-type]
            depths_m=[float(d) for d in manifest["depths_m"]],  # type: ignore[attr-defined]
            available=manifest["available_dates"],  # type: ignore[arg-type]
        )
    except EventsFileError:
        raise APIError(503, "events_unavailable", "The event list is unavailable") from None
    visible = visible_events(
        events,
        test_range=manifest["splits"]["test"],  # type: ignore[index]
        test_evaluated=bool(manifest.get("test_evaluated", False)),
    )
    for event in visible:
        if event["id"] == event_id:
            return event
    raise APIError(404, "event_not_found", "No curated event has that id")


@router.post(
    "/postcard",
    response_model=PostcardResponse,
    tags=["postcard"],
    summary="Ocean postcard (Muse Image)",
    description=(
        "Makes an artistic postcard of the current view. The image prompt comes only from a "
        "fixed template filled with region names, wording computed from the model grid and the "
        "curated event title; no user text. Identical requests are served from a cache. The "
        "caption numbers come from the model outputs and are never drawn into the image."
    ),
)
def post_postcard(body: PostcardRequest, request: Request, response: Response) -> PostcardResponse:
    """Make (or fetch from cache) one postcard."""
    response.headers["Cache-Control"] = "no-store"
    _require_enabled()
    get_limiter().check(client_ip(request))

    store = store_module.store
    store._ensure_loaded()
    manifest = store.loader.manifest
    assert manifest is not None

    store.ensure_not_sealed(body.date)
    event = _find_event(body.event_id, manifest) if body.event_id else None
    model_tag = "mock" if settings.META_CLIENT == "mock" else (settings.META_IMAGE_MODEL or "")
    plan = postcard_module.plan_postcard(
        store,
        date=body.date,
        depth_m=body.depth_m,
        style=body.style,
        event=event,
        model_tag=model_tag,
    )

    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    cached = True
    with postcard_module.key_lock(plan.cache_key):
        if not postcard_module.cached_path(plan.cache_key).exists():
            cached = False
            try:
                image = client_module.build_client().image(plan.prompt, timeout_s=TIMEOUT_S)
            except MetaUpstreamError:
                logger.warning(
                    "feature=postcard status=502 latency_ms=%d request_id=%s",
                    (time.monotonic() - started) * 1000,
                    request_id,
                )
                raise APIError(
                    502, "upstream_error", "Postcards are unavailable right now"
                ) from None
            postcard_module.write_cached(plan.cache_key, image)

    logger.info(
        "feature=postcard status=200 latency_ms=%d request_id=%s cached=%s",
        (time.monotonic() - started) * 1000,
        request_id,
        cached,
    )
    return PostcardResponse(
        image_url=f"/api/v1/postcard/{plan.cache_key}.webp",
        caption=PostcardCaption(
            date=body.date,
            depth_m=plan.depth_m,
            mean_c=plan.mean_c,
            source=CAPTION_SOURCE,
        ),
        label=POSTCARD_CAVEAT,
        prompt_version=POSTCARD_PROMPT_VERSION,
        provider_label=postcard_provider_label(),
        run_id=manifest["active_run_id"],
        synthetic=manifest["synthetic"],
    )


@router.get(
    "/postcard/{name}",
    tags=["postcard"],
    summary="A cached postcard image",
    description="Serves a postcard made by POST /postcard. Names are content hashes.",
    response_class=FileResponse,
    responses={200: {"content": {"image/webp": {}}}},
)
def get_postcard_image(name: str) -> FileResponse:
    """Return one cached postcard image."""
    _require_enabled()
    key, dot, ext = name.partition(".")
    if dot != "." or ext != "webp" or not postcard_module.is_cache_key(key):
        raise APIError(404, "postcard_not_found", "No postcard with that name")
    path = postcard_module.cached_path(key)
    if not path.is_file():
        raise APIError(404, "postcard_not_found", "No postcard with that name")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )
