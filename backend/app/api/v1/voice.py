"""Voice input endpoint: POST /transcribe (PRD_BACKEND §4.13, §6.5)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.routing import APIRoute

import app.data.store as store_module
from app.errors import APIError
from app.meta import client as client_module
from app.meta.client import MetaUpstreamError
from app.meta.config import voice_available, voice_provider_label
from app.meta.limits import FeatureLimiter, client_ip
from app.schemas.models import TranscribeResponse
from app.settings import settings

logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 1_000_000
# Allowance for multipart framing. The whole body stays under Starlette's 1 MiB in-memory spool
# limit, so an accepted upload never rolls over to a temporary file on disk (PRD_BACKEND §4.13).
MAX_BODY_BYTES = MAX_AUDIO_BYTES + 8_192
ALLOWED_AUDIO_TYPES = frozenset({"audio/webm", "audio/mp4", "audio/ogg"})
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$")
TIMEOUT_S = 20.0

_limiter: FeatureLimiter | None = None


def get_limiter() -> FeatureLimiter:
    global _limiter
    if _limiter is None:
        _limiter = FeatureLimiter(
            settings.VOICE_RATE_LIMIT, settings.VOICE_DAILY_CAP, settings.VOICE_IP_DAILY_CAP
        )
    return _limiter


class _BoundedRoute(APIRoute):
    """Checks the flag, the rate limit and the declared body length before FastAPI reads the
    body, so an oversized upload is refused without being parsed or spooled."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def bounded(request: Request) -> Response:
            if not voice_available():
                raise APIError(503, "voice_disabled", "Voice input is not available right now")
            get_limiter().check(client_ip(request))
            length = request.headers.get("content-length")
            if length is None:
                raise APIError(411, "length_required", "Send the upload with a Content-Length")
            if not length.isdigit() or int(length) > MAX_BODY_BYTES:
                raise APIError(413, "audio_too_large", "Audio must be 1 MB or less")
            response = await handler(request)
            response.headers["Cache-Control"] = "no-store"
            return response

        return bounded


router = APIRouter(route_class=_BoundedRoute)


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    tags=["voice"],
    summary="Voice input (Muse Voice Transcribe)",
    description=(
        "Transcribes one short clip (30 s or less, 1 MB or less; WebM/Opus, MP4 or Ogg). The audio "
        "is forwarded and discarded in memory; the browser shows the text so the user can edit it "
        "before it is sent to /chat."
    ),
)
def post_transcribe(
    audio: UploadFile = File(..., description="Clip of 30 s or less, 1 MB or less"),
    lang: str = Form("en", description="BCP-47 language tag"),
) -> TranscribeResponse:
    """Turn one spoken question into text."""
    mime = (audio.content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_AUDIO_TYPES:
        raise APIError(415, "unsupported_audio", "Audio must be WebM/Opus, MP4 or Ogg")
    if not _LANG_RE.match(lang):
        raise APIError(422, "invalid_lang", "lang must be a BCP-47 tag such as en or hi-IN")
    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise APIError(413, "audio_too_large", "Audio must be 1 MB or less")
    if not data:
        raise APIError(422, "empty_audio", "The audio upload is empty")

    store_module.store._ensure_loaded()
    manifest = store_module.store.loader.manifest
    assert manifest is not None

    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        result = client_module.build_client().transcribe(data, mime, lang, timeout_s=TIMEOUT_S)
    except MetaUpstreamError:
        logger.warning(
            "feature=voice status=502 latency_ms=%d request_id=%s",
            (time.monotonic() - started) * 1000,
            request_id,
        )
        raise APIError(502, "upstream_error", "Transcription is unavailable right now") from None

    logger.info(
        "feature=voice status=200 latency_ms=%d request_id=%s bytes=%d",
        (time.monotonic() - started) * 1000,
        request_id,
        len(data),
    )
    return TranscribeResponse(
        text=result.text,
        duration_s=result.duration_s,
        provider_label=voice_provider_label(),
        run_id=manifest["active_run_id"],
        synthetic=manifest["synthetic"],
    )
