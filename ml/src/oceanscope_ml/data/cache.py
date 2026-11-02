"""Fast local training cache over the processed store (PRD_AI_ML §5.1, §5.2, §5.5).

Reading one sample from the Zarr store decompresses whole 32-day chunks of every input and
target variable, about 0.4 s per sample on the real grid. The cache rewrites the train and
validation dates once into plain float16 ``.npy`` files read through memory maps, so a
sample costs a few small slices.

What the cache holds (time axis = the store's first date up to the end of the last trainable
split; test dates are never written, PRD §0 rule 3):
- each dynamic input variable, already z-scored with ``norm_stats`` (raw float16 would
  underflow on small fields such as wind-stress curl, ~1e-7);
- ``y`` = (thetao - clim) / sigma_d, NaN where the target is invalid (``target_valid``
  folded in);
- ``grep_y`` the same for the GREP mean, when the store has it.

The cache never computes statistics: it only applies ``norm_stats``, ``clim`` and ``sigma_d``
from the store. It is rebuilt when the stores, the stats or the cut-off date change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.data.transform import check_no_test_leakage

logger = logging.getLogger("oceanscope_ml.data.cache")

CACHE_VERSION = 1
BLOCK_DAYS = 64
META_FILE = "meta.json"


def cache_end_date(config: Any) -> pd.Timestamp:
    """Last date the cache may hold: the end of train or val, whichever is later."""
    end = max(pd.Timestamp(config.splits.train[1]), pd.Timestamp(config.splits.val[1]))
    check_no_test_leakage([end], config.splits.test)
    return end


def _fingerprint(ds: xr.Dataset, input_vars: list[str], end: pd.Timestamp) -> str:
    """Hash of everything the cached values depend on."""
    times = pd.to_datetime(ds["time"].values)
    payload = {
        "version": CACHE_VERSION,
        "parts": ds.attrs.get("parts", []),
        "built_years": ds.attrs.get("built_years", []),
        "norm_stats": ds.attrs.get("norm_stats"),
        "first": str(times[0].date()),
        "end": str(end.date()),
        "inputs": input_vars,
        "has_grep": "grep_mean" in ds.data_vars,
        "shape": [int(ds.sizes[d]) for d in ("depth", "lat", "lon")],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _n_cached_days(ds: xr.Dataset, end: pd.Timestamp) -> int:
    times = pd.to_datetime(ds["time"].values)
    n = int(np.searchsorted(times.values, end.to_datetime64(), side="right"))
    if n == 0:
        raise ValueError(f"The store has no dates on or before {end.date()}")
    return n


def build_cache(
    ds: xr.Dataset,
    config: Any,
    input_vars: list[str],
    cache_dir: Path,
) -> Path:
    """Write the cache for ``ds`` (opened with ``parts.open_parts``) unless it is current."""
    end = cache_end_date(config)
    fingerprint = _fingerprint(ds, input_vars, end)
    meta_path = cache_dir / META_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            logger.info("Training cache at %s is current; reusing it.", cache_dir)
            return cache_dir
        logger.info("Training cache at %s is out of date; rebuilding it.", cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    n_t = _n_cached_days(ds, end)
    times = pd.to_datetime(ds["time"].values[:n_t])
    check_no_test_leakage(times, config.splits.test)
    n_d, n_lat, n_lon = (int(ds.sizes[d]) for d in ("depth", "lat", "lon"))
    stats = ds.attrs["norm_stats"]
    sigma = np.asarray(stats["sigma_d"], dtype=np.float64)[None, :, None, None]
    clim = np.asarray(ds["clim"].values, dtype=np.float64)  # (366, depth, lat, lon)
    has_grep = "grep_mean" in ds.data_vars
    has_valid = "target_valid" in ds.data_vars

    def open_out(name: str, shape: tuple[int, ...]) -> np.memmap:
        return np.lib.format.open_memmap(
            cache_dir / f"{name}.npy", mode="w+", dtype=np.float16, shape=shape
        )

    outs: dict[str, np.memmap] = {v: open_out(v, (n_t, n_lat, n_lon)) for v in input_vars}
    outs["y"] = open_out("y", (n_t, n_d, n_lat, n_lon))
    if has_grep:
        outs["grep_y"] = open_out("grep_y", (n_t, n_d, n_lat, n_lon))

    logger.info(
        "Building training cache at %s: %d days (%s to %s), %d input variables%s...",
        cache_dir,
        n_t,
        times[0].date(),
        times[-1].date(),
        len(input_vars),
        " + GREP" if has_grep else "",
    )
    for t0 in range(0, n_t, BLOCK_DAYS):
        t1 = min(t0 + BLOCK_DAYS, n_t)
        block = ds.isel(time=slice(t0, t1))
        for v in input_vars:
            entry = stats["channels"][v]
            raw = np.asarray(block[v].values, dtype=np.float64)
            outs[v][t0:t1] = ((raw - entry["mean"]) / entry["std"]).astype(np.float16)

        clim_block = clim[times[t0:t1].day_of_year.values - 1]
        y = (np.asarray(block["thetao"].values, dtype=np.float64) - clim_block) / sigma
        if has_valid:
            y = np.where(np.asarray(block["target_valid"].values, dtype=bool), y, np.nan)
        outs["y"][t0:t1] = y.astype(np.float16)
        if has_grep:
            g = (np.asarray(block["grep_mean"].values, dtype=np.float64) - clim_block) / sigma
            outs["grep_y"][t0:t1] = g.astype(np.float16)
        logger.info("Cache: %d / %d days written", t1, n_t)

    for arr in outs.values():
        arr.flush()
    meta = {
        "fingerprint": fingerprint,
        "n_days": n_t,
        "first": str(times[0].date()),
        "last": str(times[-1].date()),
        "inputs": input_vars,
        "has_grep": has_grep,
    }
    # meta.json goes last, so an interrupted build is never mistaken for a finished one.
    meta_path.write_text(json.dumps(meta, indent=2))
    size_gb = sum(f.stat().st_size for f in cache_dir.glob("*.npy")) / 1e9
    logger.info("Training cache ready: %.1f GB", size_gb)
    return cache_dir


def open_cache(cache_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the cache metadata and read-only memory maps of its arrays."""
    meta = json.loads((cache_dir / META_FILE).read_text())
    names = [*meta["inputs"], "y"] + (["grep_y"] if meta["has_grep"] else [])
    arrays = {n: np.load(cache_dir / f"{n}.npy", mmap_mode="r") for n in names}
    return meta, arrays
