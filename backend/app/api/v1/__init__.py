"""API v1 router bundle for OceanScope AI backend."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.beach import router as beach_router
from app.api.v1.chat import router as chat_router
from app.api.v1.data import router as data_router
from app.api.v1.events import router as events_router
from app.api.v1.glasses import router as glasses_router
from app.api.v1.meta import router as meta_router
from app.api.v1.postcard import router as postcard_router
from app.api.v1.segment import router as segment_router
from app.api.v1.validation import router as validation_router
from app.api.v1.voice import router as voice_router

api_v1_router = APIRouter(prefix="/api/v1")
root_router = APIRouter()

# Mount sub-routers to both prefix and root
for router in (api_v1_router, root_router):
    router.include_router(meta_router)
    router.include_router(data_router)
    router.include_router(validation_router)
    router.include_router(chat_router)
    router.include_router(voice_router)
    router.include_router(events_router)
    router.include_router(postcard_router)
    router.include_router(segment_router)
    router.include_router(beach_router)
    router.include_router(glasses_router)
