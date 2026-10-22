"""Chat assistant endpoint: /chat (PRD_BACKEND §4.10, §6)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.models import ChatRequest, ChatResponse
from app.settings import settings

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="LLM oceanographic data assistant",
    description=(
        "Conversational assistant grounded in OutputStore queries "
        "with tool calling and number-grounding guards."
    ),
)
def post_chat(request: ChatRequest) -> ChatResponse:
    """Interact with the OceanScope AI assistant."""
    if not settings.CHAT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "chat_disabled",
                    "message": "Chat assistant is disabled",
                    "details": None,
                }
            },
        )
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Endpoint will be implemented in P1-B5",
    )
