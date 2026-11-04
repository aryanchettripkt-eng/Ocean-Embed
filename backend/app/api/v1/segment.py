"""Lasso: POST /segment (PRD_BACKEND §4.16).

SAM 3.1 (via the Meta Model API) draws the shape on a PNG we render from our own grid; the
statistics always come from the numeric grid. If SAM is off, fails, times out or returns an
empty mask, the deterministic threshold method is used, and the response says which one ran.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

import numpy as np
from cachetools import TTLCache
from fastapi import APIRouter, Request, Response

import app.data.store as store_module
from app.data.grid import snap_coordinate
from app.data.store import area_weighted_stats
from app.errors import APIError
from app.meta import client as client_module
from app.meta import segment as seg
from app.meta.client import MetaUpstreamError
from app.meta.config import sam_configured, segment_available, segment_provider_label
from app.meta.limits import FeatureLimiter, client_ip
from app.schemas.caveats import SEGMENT_CAVEAT
from app.schemas.models import SegmentMask, SegmentRequest, SegmentResponse, SegmentStats
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Whole SAM call, well under the frontend's 20 s lasso abort (LASSO_TIMEOUT_MS), so a slow SAM
# reply falls back to the threshold method and the browser still gets an answer.
SAM_TIMEOUT_S = 6.0

_limiter: FeatureLimiter | None = None
# Identical inputs give identical answers, so repeated taps do not call SAM again (§4.12).
_cache: TTLCache[tuple[object, ...], SegmentResponse] = TTLCache(maxsize=256, ttl=3600)
_cache_lock = threading.Lock()


def get_limiter() -> FeatureLimiter:
    global _limiter
    if _limiter is None:
        _limiter = FeatureLimiter(
            settings.SEGMENT_RATE_LIMIT, settings.SEGMENT_DAILY_CAP, settings.SEGMENT_IP_DAILY_CAP
        )
    return _limiter


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


@router.post(
    "/segment",
    response_model=SegmentResponse,
    tags=["segment"],
    summary="Lasso a region from a tapped point (SAM 3.1, threshold fallback)",
    description=(
        "Selects the area around a tapped point on our own rendered grid. Statistics are "
        "area-weighted over the selected ocean cells, with the same code as /region-summary. "
        "`method` says whether SAM or the similar-temperature fallback made the selection."
    ),
)
def post_segment(body: SegmentRequest, request: Request, response: Response) -> SegmentResponse:
    """Select the region around one tapped point."""
    if not segment_available():
        raise APIError(503, "segment_disabled", "The lasso is not available right now")
    get_limiter().check(client_ip(request))

    store = store_module.store
    store._ensure_loaded()
    manifest = store.loader.manifest
    assert manifest is not None

    store.ensure_not_sealed(body.date)
    grid = store.get_field_grid(body.date, depth_m=body.depth_m, field=body.field)
    reg = manifest["region"]
    r0, snapped_lat = snap_coordinate(body.lat, grid.lats, reg["lat_min"], reg["lat_max"])
    c0, snapped_lon = snap_coordinate(body.lon, grid.lons, reg["lon_min"], reg["lon_max"])
    valid = (grid.coverage == 1) & ~np.isnan(grid.values)
    if not valid[r0, c0]:
        raise APIError(422, "point_not_ocean", "The tapped point has no ocean value to select")

    use_sam = sam_configured()
    key = (
        manifest["active_run_id"],
        id(manifest),
        grid.date,
        grid.depth_m,
        grid.field,
        r0,
        c0,
        use_sam,
        settings.META_CLIENT,
        settings.SEGMENT_THRESHOLD_C,
    )
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        response.headers["Cache-Control"] = "private, max-age=300"
        return hit

    request_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    cells: np.ndarray | None = None
    capped = False
    method: str = "threshold"
    if use_sam:
        cells = _try_sam(store, grid, valid, (r0, c0), request_id)
        if cells is not None:
            method = "sam"
    if cells is None:
        cells, capped = seg.threshold_select(
            grid.values, valid, (r0, c0), settings.SEGMENT_THRESHOLD_C
        )

    res_deg = float(manifest["grid_resolution_deg"])
    stats = area_weighted_stats(grid.values, cells, grid.lats)
    outline_cells = seg.connected_component(cells, (r0, c0))
    result = SegmentResponse(
        method="sam" if method == "sam" else "threshold",
        date=grid.date,
        depth_m=grid.depth_m,
        field=body.field,
        lat=snapped_lat,
        lon=snapped_lon,
        mask=SegmentMask(lat_index_ranges=seg.mask_to_ranges(cells)),
        stats=SegmentStats(
            mean=stats.mean,
            min=stats.min,
            max=stats.max,
            std=stats.std,
            n_valid=stats.n_valid,
            area_km2=seg.area_km2(cells, grid.lats, res_deg),
        ),
        outline=seg.outline_polygon(outline_cells, grid.lats, grid.lons, res_deg),
        capped=capped,
        caveat=SEGMENT_CAVEAT,
        provider_label=segment_provider_label("sam" if method == "sam" else "threshold"),
        run_id=manifest["active_run_id"],
        synthetic=manifest["synthetic"],
    )
    logger.info(
        "feature=segment status=200 latency_ms=%d request_id=%s method=%s cells=%d",
        (time.monotonic() - started) * 1000,
        request_id,
        method,
        stats.n_valid,
    )
    with _cache_lock:
        _cache[key] = result
    response.headers["Cache-Control"] = "private, max-age=300"
    return result


def _try_sam(
    store: store_module.OutputStore,
    grid: store_module.FieldGrid,
    valid: np.ndarray,
    seed: tuple[int, int],
    request_id: str,
) -> np.ndarray | None:
    """SAM's selection intersected with valid ocean cells, or None to fall back."""
    ranges = {c.depth_m: c for c in store.get_meta().color_ranges}
    color = ranges.get(grid.depth_m)
    if color is not None:
        vmin, vmax = color.vmin, color.vmax
    else:
        vmin, vmax = float(np.nanmin(grid.values)), float(np.nanmax(grid.values))
    rendered = seg.render_grid_png(grid.values, grid.coverage, vmin, vmax)
    try:
        mask = client_module.build_client().segment(
            rendered.png, rendered.pixel_of(*seed), timeout_s=SAM_TIMEOUT_S
        )
    except MetaUpstreamError:
        logger.warning("feature=segment sam=failed fallback=threshold request_id=%s", request_id)
        return None
    cells: np.ndarray = rendered.mask_to_cells(np.asarray(mask, dtype=bool)) & valid
    if not cells.any():
        logger.info("feature=segment sam=empty fallback=threshold request_id=%s", request_id)
        return None
    return cells
