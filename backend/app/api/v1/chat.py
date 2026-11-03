"""Ocean guide endpoint: /chat (PRD_BACKEND §4.10, §6)."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Request, Response

import app.data.store as store_module
from app.errors import APIError, ChatDisabledError
from app.meta import client as client_module
from app.meta.client import MetaUpstreamError
from app.meta.config import chat_available, provider_label
from app.meta.guide import run_guide
from app.meta.limits import FeatureLimiter, client_ip
from app.schemas.models import ChatRequest, ChatResponse
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

_limiter: FeatureLimiter | None = None


def get_limiter() -> FeatureLimiter:
    global _limiter
    if _limiter is None:
        _limiter = FeatureLimiter(settings.CHAT_RATE_LIMIT, settings.CHAT_DAILY_CAP)
    return _limiter


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Ocean guide (Muse Spark via the Meta Model API)",
    description=(
        "Answers questions with data tools over the OutputStore and moves the explorer with "
        "backend-validated UI actions. Every number in the answer is checked against tool "
        "results; if one is not, the answer is replaced by the raw results (grounded=false)."
    ),
)
def post_chat(body: ChatRequest, request: Request, response: Response) -> ChatResponse:
    """Ask the ocean guide a question."""
    response.headers["Cache-Control"] = "no-store"
    if not chat_available():
        raise ChatDisabledError()
    get_limiter().check(client_ip(request))

    store = store_module.store
    store._ensure_loaded()
    assert store.loader.manifest is not None
    manifest = store.loader.manifest

    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        result = run_guide(body, client_module.build_client(), store)
    except MetaUpstreamError:
        logger.warning(
            "feature=chat status=502 latency_ms=%d request_id=%s",
            (time.monotonic() - started) * 1000,
            request_id,
        )
        raise APIError(502, "upstream_error", "The guide is unavailable right now") from None

    logger.info(
        "feature=chat status=200 latency_ms=%d request_id=%s rounds=%d grounded=%s",
        (time.monotonic() - started) * 1000,
        request_id,
        result.rounds,
        result.grounded,
    )
    return ChatResponse(
        answer=result.answer,
        tool_calls=result.tool_calls,
        ui_actions=result.ui_actions,
        grounded=result.grounded,
        provider_label=provider_label(),
        run_id=manifest["active_run_id"],
        synthetic=manifest["synthetic"],
    )
