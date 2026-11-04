"""Glasses ocean guide: POST /glasses/ask (PRD_BACKEND §4.18).

One request from the Android glasses app: a spoken (or typed) question plus the phone's location
and compass heading. The answer is built from the prediction store alone, so every number is
exact and no chat model is called. Audio goes to Muse Voice Transcribe and is then discarded.
The location is used only for this request: never stored, never logged.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.routing import APIRoute

import app.data.store as store_module
from app.api.v1.beach import locate
from app.api.v1.voice import (
    _LANG_RE,
    ALLOWED_AUDIO_TYPES,
    MAX_AUDIO_BYTES,
    MAX_BODY_BYTES,
    TIMEOUT_S,
    WAV_MESSAGE,
    is_supported_wav,
)
from app.data import glasses
from app.errors import APIError
from app.meta import client as client_module
from app.meta.client import MetaUpstreamError
from app.meta.config import glasses_available, voice_available
from app.meta.limits import FeatureLimiter, client_ip
from app.schemas.caveats import GLORYS_CAVEAT, LIVE_CAVEAT, SYNTHETIC_CAVEAT
from app.schemas.models import GlassesAskResponse
from app.settings import settings

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 300

_limiter: FeatureLimiter | None = None


def get_limiter() -> FeatureLimiter:
    global _limiter
    if _limiter is None:
        _limiter = FeatureLimiter(
            settings.GLASSES_RATE_LIMIT, settings.GLASSES_DAILY_CAP, settings.GLASSES_IP_DAILY_CAP
        )
    return _limiter


class _BoundedRoute(APIRoute):
    """Flag, rate limit and declared body length are checked before the body is read."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def bounded(request: Request) -> Response:
            if not glasses_available():
                raise APIError(
                    503, "glasses_disabled", "The glasses guide is not available right now"
                )
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


def _transcribe(audio: UploadFile, lang: str) -> str:
    if not voice_available():
        raise APIError(503, "voice_disabled", "Voice input is not available right now")
    mime = (audio.content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_AUDIO_TYPES:
        raise APIError(415, "unsupported_audio", WAV_MESSAGE)
    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise APIError(413, "audio_too_large", "Audio must be 1 MB or less")
    if not data:
        raise APIError(422, "empty_audio", "The audio upload is empty")
    if not is_supported_wav(data):
        raise APIError(415, "unsupported_audio", WAV_MESSAGE)
    try:
        result = client_module.build_client().transcribe(data, mime, lang, timeout_s=TIMEOUT_S)
    except MetaUpstreamError:
        raise APIError(502, "upstream_error", "Transcription is unavailable right now") from None
    return result.text


@router.post(
    "/glasses/ask",
    response_model=GlassesAskResponse,
    tags=["glasses"],
    summary="Glasses ocean guide: temperature where the user is looking",
    description=(
        "Takes a spoken question (mono 16-bit PCM WAV at 16 or 24 kHz, 1 MB or less) or typed "
        "text, plus the phone's location and compass heading. Finds the first ocean cell along "
        "the heading that has water at the asked depth (as GET /nearest-ocean with min_depth_m), "
        "reads its profile on the latest available date, blends linearly between the two "
        "model levels around the asked depth, and returns a short spoken answer. No chat "
        "model is called; the location is not stored or logged."
    ),
)
def post_glasses_ask(
    lat: float = Form(..., ge=-90, le=90, description="User latitude"),
    lon: float = Form(..., ge=-180, le=180, description="User longitude"),
    bearing_deg: float | None = Form(
        None, ge=0, lt=360, description="Compass heading, true north, clockwise"
    ),
    heading_accuracy_deg: float | None = Form(
        None, ge=0, le=180, description="Compass accuracy reported by the phone, if known"
    ),
    max_km: float = Form(300.0, gt=0, le=500, description="How far to look"),
    text: str | None = Form(None, description="Typed question (used when no audio is sent)"),
    audio: UploadFile | None = File(None, description="Spoken question, 30 s or less"),
    lang: str = Form("en", description="BCP-47 language tag for transcription"),
) -> GlassesAskResponse:
    """Answer "how warm is it out there at N metres?" for the direction the user faces."""
    if not _LANG_RE.match(lang):
        raise APIError(422, "invalid_lang", "lang must be a BCP-47 tag such as en or hi-IN")
    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    source = "audio" if audio is not None else "text"
    question = (_transcribe(audio, lang) if audio is not None else (text or "")).strip()
    if not question:
        raise APIError(422, "empty_question", "Send a question as audio or text")
    question = question[:MAX_QUESTION_CHARS]

    store = store_module.store
    store._ensure_loaded()
    manifest = store.loader.manifest
    assert manifest is not None
    depths = [float(d) for d in manifest["depths_m"]]
    deepest_model = depths[-1]

    asked = glasses.parse_depth_m(question)
    target = depths[0] if asked is None else min(max(asked, 0.0), deepest_model)

    too_shallow = False
    found = locate(lat, lon, bearing_deg, max_km, target, feature="glasses")
    if found is None:
        too_shallow = True
        found = locate(lat, lon, bearing_deg, max_km, None, feature="glasses")
    if found is None:
        logger.info("feature=glasses status=404 source=%s request_id=%s", source, request_id)
        raise APIError(404, "no_ocean_in_view", "No ocean cell found in that direction")
    hit, date, _ = found

    profile = store.get_profile(hit.lat, hit.lon, date)
    value = (
        glasses.deepest_value(profile.depths_m, profile.pred)
        if too_shallow
        else glasses.value_at_depth(profile.depths_m, profile.pred, target)
    )
    if value is None:
        raise APIError(404, "no_ocean_in_view", "No ocean cell found in that direction")
    # Only a question with a depth can be "too shallow" (a depthless one asks for the surface).
    too_shallow = too_shallow and asked is not None

    synthetic = bool(manifest["synthetic"])
    answer = glasses.spoken_answer(
        glasses.AnswerParts(
            distance_km=hit.distance_km,
            bearing_deg=hit.bearing_deg,
            date=date,
            value=value,
            depth_asked_m=asked,
            deepest_model_m=deepest_model,
            too_shallow=too_shallow,
            max_km=max_km,
            heading_accuracy_deg=heading_accuracy_deg,
            synthetic=synthetic,
        )
    )
    caveats = [GLORYS_CAVEAT]
    if profile.split == "live":
        caveats.append(LIVE_CAVEAT)
    if synthetic:
        caveats.append(SYNTHETIC_CAVEAT)

    logger.info(
        "feature=glasses status=200 source=%s depth_asked=%s too_shallow=%s latency_ms=%d "
        "request_id=%s",
        source,
        asked is not None,
        too_shallow,
        (time.monotonic() - started) * 1000,
        request_id,
    )
    return GlassesAskResponse(
        answer_text=answer,
        question=question,
        temperature_c=value.temperature_c,
        depth_asked_m=asked,
        depth_used_m=value.depth_used_m,
        levels_m=value.levels_m,
        interpolated=value.interpolated,
        too_shallow=too_shallow,
        lat=hit.lat,
        lon=hit.lon,
        distance_km=hit.distance_km,
        bearing_deg=hit.bearing_deg,
        direction=glasses.compass_word(hit.bearing_deg),
        date=date,
        split=profile.split,
        caveats=caveats,
        run_id=manifest["active_run_id"],
        synthetic=synthetic,
    )
