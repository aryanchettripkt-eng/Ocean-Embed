"""API v1 router bundle for OceanScope AI backend."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.data import router as data_router
from app.api.v1.meta import router as meta_router
from app.api.v1.validation import router as validation_router

api_v1_router = APIRouter(prefix="/api/v1")
root_router = APIRouter()

# Mount sub-routers to both prefix and root
for router in (api_v1_router, root_router):
    router.include_router(meta_router)
    router.include_router(data_router)
    router.include_router(validation_router)
    router.include_router(chat_router)
