"""Feature availability for the Meta features (PRD_BACKEND §4.12)."""

from __future__ import annotations

from typing import Literal

from app.meta.budget import budget_configured
from app.settings import settings

MOCK_PROVIDER_LABEL = "Offline mock guide (no Meta model)"
VOICE_PROVIDER_LABEL = "Muse Voice Transcribe via Meta Model API"
MOCK_VOICE_PROVIDER_LABEL = "Offline mock transcription (no Meta model)"
POSTCARD_PROVIDER_LABEL = "Muse Image via Meta Model API"
MOCK_POSTCARD_PROVIDER_LABEL = "Offline mock postcard (no Meta model)"
SAM_PROVIDER_LABEL = "SAM 3.1 via Meta Model API"
MOCK_SAM_PROVIDER_LABEL = "Offline mock segmentation (no Meta model)"
THRESHOLD_PROVIDER_LABEL = "Similar-temperature selection (no AI model)"


def chat_available() -> bool:
    """True when the guide is enabled and its client is fully configured."""
    if not settings.CHAT_ENABLED:
        return False
    if settings.META_CLIENT == "mock":
        return True
    return bool(
        settings.META_API_BASE_URL
        and settings.META_API_KEY
        and settings.META_CHAT_MODEL
        and budget_configured("chat")
    )


def voice_available() -> bool:
    """True when voice input is enabled and its client is fully configured (PRD_BACKEND §4.12)."""
    if not settings.VOICE_ENABLED:
        return False
    if settings.META_CLIENT == "mock":
        return True
    return bool(
        settings.META_API_BASE_URL
        and settings.META_API_KEY
        and settings.META_TRANSCRIBE_MODEL
        and budget_configured("transcribe")
    )


def postcard_available() -> bool:
    """True when postcards are enabled and the image model is configured (PRD_BACKEND §4.15)."""
    if not settings.POSTCARD_ENABLED:
        return False
    if settings.META_CLIENT == "mock":
        return True
    return bool(
        settings.META_API_BASE_URL
        and settings.META_API_KEY
        and settings.META_IMAGE_MODEL
        and budget_configured("image")
    )


def segment_available() -> bool:
    """True when the lasso is enabled. It never needs the Meta API: without SAM it uses the
    deterministic threshold method (PRD_BACKEND §4.16)."""
    return settings.SEGMENT_ENABLED


def sam_configured() -> bool:
    """True when SAM may be called: provider is `sam` and the client is mock or fully set up."""
    if settings.SEGMENT_PROVIDER != "sam":
        return False
    if settings.META_CLIENT == "mock":
        return True
    return bool(
        settings.META_API_BASE_URL
        and settings.META_API_KEY
        and settings.META_SEGMENT_MODEL
        and budget_configured("segment")
    )


def segment_method() -> Literal["sam", "threshold"]:
    """The method the lasso tries first, as reported in /meta.features.segment_method."""
    return "sam" if sam_configured() else "threshold"


def beach_available() -> bool:
    """Beach mode is a pure data lookup (PRD_BACKEND §4.17); only its flag gates it."""
    return settings.BEACH_ENABLED


def glasses_available() -> bool:
    """The glasses guide answers from the data alone (PRD_BACKEND §4.18); only its flag gates
    it. Spoken (audio) questions also need voice input to be available."""
    return settings.GLASSES_ENABLED


def provider_label() -> str:
    """Display name of the model behind the guide, returned to the UI."""
    if settings.META_CLIENT == "mock":
        return MOCK_PROVIDER_LABEL
    return settings.PROVIDER_LABEL


def voice_provider_label() -> str:
    """Display name of the transcription model, returned with every /transcribe response."""
    if settings.META_CLIENT == "mock":
        return MOCK_VOICE_PROVIDER_LABEL
    return VOICE_PROVIDER_LABEL


def postcard_provider_label() -> str:
    """Display name of the image model, returned with every /postcard response."""
    if settings.META_CLIENT == "mock":
        return MOCK_POSTCARD_PROVIDER_LABEL
    return POSTCARD_PROVIDER_LABEL


def segment_provider_label(method: Literal["sam", "threshold"]) -> str:
    """Display name of what drew the selection, returned with every /segment response."""
    if method == "threshold":
        return THRESHOLD_PROVIDER_LABEL
    if settings.META_CLIENT == "mock":
        return MOCK_SAM_PROVIDER_LABEL
    return SAM_PROVIDER_LABEL
