"""Recent ARGO profiles for live mode's ARGO input channels (PRD_AI_ML §5.1, §7, §12).

Runs with ARGO input channels (main2_ai, main3 and an ensemble holding them) need the profiles
of the 10 days up to the target day. They come from the same Argo GDAC ERDDAP table as the
training profiles (``data.argo``), which also serves real-time profiles about a day after a
float surfaces. The same QC filter (flags 1-2), held-out list and depth interpolation are used,
so live input channels are built exactly like validation ones.

Days in the sealed test period are refused (PRD_AI_ML §0); days after it are allowed.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.argo import (
    ERDDAP_CSV,
    MAX_PRES_DBAR,
    VARIABLES,
    csv_to_dataset,
    profile_levels,
    read_levels,
)

logger = logging.getLogger("oceanscope_ml.live.argo")

# fetch(start, end) -> one row per profile on the model depths (profile_levels layout)
ArgoFetcher = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


def refuse_test_overlap(start: pd.Timestamp, end: pd.Timestamp, config: ExperimentConfig) -> None:
    test_start, test_end = (pd.Timestamp(d) for d in config.splits.test)
    if start <= test_end and end >= test_start:
        raise ValueError(
            f"ARGO days {start.date()}..{end.date()} touch the sealed test period (PRD_AI_ML §0)"
        )


def range_query(start: pd.Timestamp, end: pd.Timestamp, config: ExperimentConfig) -> str:
    region = config.region
    query = (
        f"{','.join(VARIABLES)}"
        f"&time>={start.strftime('%Y-%m-%dT00:00:00Z')}"
        f"&time<={end.strftime('%Y-%m-%dT23:59:59Z')}"
        f"&latitude>={region.lat_min}&latitude<={region.lat_max}"
        f"&longitude>={region.lon_min}&longitude<={region.lon_max}&pres<={MAX_PRES_DBAR}"
    )
    return f"{ERDDAP_CSV}?{urllib.parse.quote(query, safe='&,=')}"


def _download(url: str, retries: int, opener: Callable[[str], str] | None) -> str:
    if opener is not None:
        return opener(url)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return str(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # ERDDAP's answer when no rows match
                return ""
            logger.warning("ARGO ERDDAP: HTTP %s, retry %d", exc.code, attempt + 1)
        except Exception as exc:  # noqa: BLE001 - network errors of any kind are retried
            logger.warning("ARGO ERDDAP: %s, retry %d", exc, attempt + 1)
        time.sleep(10 * 2**attempt)
    raise RuntimeError(f"ERDDAP did not answer: {url}")


def fetch_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: ExperimentConfig,
    out_dir: Path,
    retries: int = 5,
    opener: Callable[[str], str] | None = None,
) -> Path:
    """Download ARGO levels for [start, end] to ``out_dir`` (kept and reused if present)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    refuse_test_overlap(start, end, config)
    path = Path(out_dir) / f"argo_live_{start.date()}_{end.date()}.nc"
    if path.exists():
        return path
    ds = csv_to_dataset(_download(range_query(start, end, config), retries, opener))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    ds.to_netcdf(tmp)
    tmp.rename(path)
    n_floats = len(set(ds["platform_number"].values.tolist()))
    logger.info(
        "ARGO %s..%s: %d levels, %d floats", start.date(), end.date(), ds.sizes["row"], n_floats
    )
    return path


def live_profiles(
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: ExperimentConfig,
    out_dir: Path,
    holdout: set[str],
    opener: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Profiles of non-held-out floats in [start, end] on the model depths."""
    if not holdout:
        raise ValueError("No held-out float list: live ARGO inputs must leave them out (§7)")
    path = fetch_range(start, end, config, out_dir, opener=opener)
    levels = read_levels([path], pd.Timestamp(start), pd.Timestamp(end))
    return profile_levels(levels, list(config.depths_m), holdout)


def default_fetch(config: ExperimentConfig, out_dir: Path, holdout: set[str]) -> ArgoFetcher:
    def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return live_profiles(start, end, config, out_dir, holdout)

    return fetch
