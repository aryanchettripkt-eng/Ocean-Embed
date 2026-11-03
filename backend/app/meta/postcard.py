"""Ocean postcards: descriptors, the fixed prompt and the image cache (PRD_BACKEND §4.15).

The image prompt is built only from the fixed template in `prompts.py`, filled with region names,
qualitative descriptors computed here from the model grid, and a curated event title. The caption
numbers come from the same grid and are overlaid as text by the frontend, never drawn.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.data.grid import get_region_mask
from app.data.store import OutputStore, area_weighted_stats
from app.meta.prompts import POSTCARD_PROMPT_VERSION, build_postcard_prompt
from app.settings import settings

REGION_NAMES: dict[str, str] = {
    "arabian_sea": "the Arabian Sea",
    "bay_of_bengal": "the Bay of Bengal",
    "equatorial": "the equatorial Indian Ocean",
}
WHOLE_REGION_NAME = "the North Indian Ocean"

# Thresholds that turn grid means into words for the picture. They only choose wording; the
# numbers shown to people are the caption values, not these.
SURFACE_WORDS = (
    (28.0, "very warm surface water"),
    (26.0, "warm surface water"),
    (22.0, "mild surface water"),
)
SURFACE_DEFAULT = "cool surface water"
DEPTH_WORDS = ((25.0, "warm water at depth"), (15.0, "temperate water at depth"))
DEPTH_DEFAULT = "cold water at depth"
THERMOCLINE_WORDS = ((8.0, "a sharp thermocline"), (4.0, "a gradual thermocline"))
THERMOCLINE_DEFAULT = "a weak thermocline"
PATCHY_STD_C = 1.5
THERMOCLINE_TOP_M = 50.0
THERMOCLINE_BOTTOM_M = 200.0

_NAME_RE_LEN = 32  # hex characters in a cache key


@dataclass(frozen=True)
class PostcardPlan:
    prompt: str
    depth_m: float
    mean_c: float | None
    cache_key: str


def _words(value: float, table: tuple[tuple[float, str], ...], default: str) -> str:
    for threshold, word in table:
        if value >= threshold:
            return word
    return default


def _nearest_depth(depths: list[float], target: float) -> float:
    return min(depths, key=lambda d: abs(d - target))


def plan_postcard(
    store: OutputStore,
    *,
    date: str,
    depth_m: float,
    style: str,
    event: dict[str, Any] | None,
    model_tag: str,
) -> PostcardPlan:
    """Compute descriptors and the caption mean, fill the template, and derive the cache key."""
    manifest = store.loader.manifest
    assert manifest is not None
    depths = [float(d) for d in manifest["depths_m"]]

    chosen = store.get_field_grid(date, depth_m=depth_m, field="pred")
    if event is not None:
        bbox = event["bbox"]
    else:
        bbox = manifest["region"]
    area = get_region_mask(
        chosen.lats,
        chosen.lons,
        lat_min=bbox["lat_min"],
        lat_max=bbox["lat_max"],
        lon_min=bbox["lon_min"],
        lon_max=bbox["lon_max"],
    )
    ocean = (chosen.coverage == 1) & area

    def mean_at(d: float) -> float | None:
        grid = chosen if d == chosen.depth_m else store.get_field_grid(date, d, field="pred")
        return area_weighted_stats(grid.values, ocean, grid.lats).mean

    chosen_stats = area_weighted_stats(chosen.values, ocean, chosen.lats)
    descriptors: list[str] = []
    surface = mean_at(depths[0])
    if surface is not None:
        descriptors.append(_words(surface, SURFACE_WORDS, SURFACE_DEFAULT))
    top = mean_at(_nearest_depth(depths, THERMOCLINE_TOP_M))
    bottom = mean_at(_nearest_depth(depths, THERMOCLINE_BOTTOM_M))
    if top is not None and bottom is not None:
        descriptors.append(_words(top - bottom, THERMOCLINE_WORDS, THERMOCLINE_DEFAULT))
    if chosen.depth_m != depths[0] and chosen_stats.mean is not None:
        descriptors.append(_words(chosen_stats.mean, DEPTH_WORDS, DEPTH_DEFAULT))
    if chosen_stats.std is not None:
        descriptors.append(
            "patchy temperatures" if chosen_stats.std >= PATCHY_STD_C else "even temperatures"
        )
    if not descriptors:
        descriptors.append("quiet water")

    if event is None:
        regions = [WHOLE_REGION_NAME]
    else:
        regions = [
            name
            for key, name in REGION_NAMES.items()
            if np.any(ocean & get_region_mask(chosen.lats, chosen.lons, region=key))  # type: ignore[arg-type]
        ] or [WHOLE_REGION_NAME]

    prompt = build_postcard_prompt(
        style, regions, descriptors, str(event["title"]) if event else None
    )
    key_src = json.dumps(
        {
            "run_id": manifest["active_run_id"],
            "prompt_version": POSTCARD_PROMPT_VERSION,
            "model": model_tag,
            "date": date,
            "depth_m": chosen.depth_m,
            "field": "pred",
            "event_id": event["id"] if event else None,
            "style": style,
            "prompt": prompt,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:_NAME_RE_LEN]
    return PostcardPlan(
        prompt=prompt, depth_m=chosen.depth_m, mean_c=chosen_stats.mean, cache_key=key
    )


def cache_dir() -> Path:
    """Postcard cache directory (created on demand), outside the repository."""
    configured = settings.POSTCARD_CACHE_DIR.strip()
    base = Path(configured) if configured else Path(tempfile.gettempdir()) / "oceanscope-postcards"
    base.mkdir(parents=True, exist_ok=True)
    return base


def is_cache_key(name: str) -> bool:
    return len(name) == _NAME_RE_LEN and all(ch in "0123456789abcdef" for ch in name)


def cached_path(key: str) -> Path:
    return cache_dir() / f"{key}.webp"


def write_cached(key: str, data: bytes) -> None:
    """Atomic write so a reader never sees a half-written image."""
    final = cached_path(key)
    fd, tmp = tempfile.mkstemp(dir=final.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, final)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def key_lock(key: str) -> threading.Lock:
    """One lock per cache key so identical concurrent requests call the model once."""
    # Bounded in practice by POSTCARD_DAILY_CAP distinct requests a day per worker.
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())
