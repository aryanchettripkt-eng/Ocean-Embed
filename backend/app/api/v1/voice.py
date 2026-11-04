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
# Muse Voice Transcribe takes only mono 16-bit PCM WAV at 16 or 24 kHz, so the browser converts
# its recording before upload (30 s at 16 kHz is 960 kB, inside the 1 MB cap).
ALLOWED_AUDIO_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})
WAV_RATES = frozenset({16_000, 24_000})
WAV_MESSAGE = "Audio must be mono 16-bit PCM WAV at 16 or 24 kHz"
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*$")
TIMEOUT_S = 20.0


def is_supported_wav(data: bytes) -> bool:
    """True for a RIFF/WAVE file whose `fmt ` chunk is mono 16-bit PCM at 16 or 24 kHz."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if chunk_id == b"fmt ":
            if size < 16 or pos + 8 + 16 > len(data):
                return False
            fmt = data[pos + 8 : pos + 24]
            audio_format = int.from_bytes(fmt[0:2], "little")
            channels = int.from_bytes(fmt[2:4], "little")
            rate = int.from_bytes(fmt[4:8], "little")
            bits = int.from_bytes(fmt[14:16], "little")
            return audio_format == 1 and channels == 1 and bits == 16 and rate in WAV_RATES
        pos += 8 + size + (size & 1)
    return False


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
        "Transcribes one short clip (30 s or less, 1 MB or less; mono 16-bit PCM WAV at 16 or "
        "24 kHz, which the browser converts its recording to). The audio "
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
        raise APIError(415, "unsupported_audio", WAV_MESSAGE)
    if not _LANG_RE.match(lang):
        raise APIError(422, "invalid_lang", "lang must be a BCP-47 tag such as en or hi-IN")
    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise APIError(413, "audio_too_large", "Audio must be 1 MB or less")
    if not data:
        raise APIError(422, "empty_audio", "The audio upload is empty")
    if not is_supported_wav(data):
        raise APIError(415, "unsupported_audio", WAV_MESSAGE)

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
