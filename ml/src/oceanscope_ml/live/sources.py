"""Check the live input sources before P9-1 (PRD_AI_ML §12, P9_LIVE_RUNBOOK step 1).

For every product in ``live.sources`` (and, with ``--all``, ``live_check.reprocessed`` and the
GLORYS check product), ask its service whether the dataset ID exists, whether it has the
variables we read, and what its newest day is. Nothing is downloaded except, for CDS, one
day over a 1° box, because CDS has no way to ask for a dataset's newest day.

The point is to replace "confirm the IDs by hand with copernicusmarine describe" with one
command that prints a table and fails (non-zero exit) when any source is wrong.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("oceanscope_ml.live.sources")

# probe(spec, today, bbox) -> SourceCheck fields (newest_day, variables_found, note)
Probe = Callable[[dict[str, Any], pd.Timestamp, tuple[float, float, float, float]], dict[str, Any]]


@dataclass
class SourceCheck:
    group: str  # "live", "reprocessed" or "glorys"
    product: str
    service: str
    dataset_id: str
    ok: bool = False
    newest_day: str | None = None
    lag_days: int | None = None
    missing_variables: list[str] = field(default_factory=list)
    note: str = ""


def specs_to_check(live_cfg: dict[str, Any], include_check: bool) -> list[tuple[str, str, dict]]:
    """(group, product, spec) for each source in the live config."""
    out: list[tuple[str, str, dict]] = []
    for product, spec in dict((live_cfg.get("live") or {}).get("sources") or {}).items():
        out.append(("live", product, dict(spec)))
    if include_check:
        check = dict(live_cfg.get("live_check") or {})
        for product, spec in dict(check.get("reprocessed") or {}).items():
            out.append(("reprocessed", product, dict(spec)))
        if check.get("glorys"):
            out.append(("glorys", "glorys", dict(check["glorys"])))
    if not out:
        raise ValueError("The live config has no sources to check")
    return out


def check_sources(
    live_cfg: dict[str, Any],
    today: pd.Timestamp,
    bbox: tuple[float, float, float, float],
    probes: dict[str, Probe],
    *,
    include_check: bool = False,
    max_lag_days: int | None = None,
) -> list[SourceCheck]:
    """Run one probe per source. A probe error marks that source failed; the rest still run.

    ``max_lag_days`` (live sources only): a newest day older than this is a failure, since a
    live run could never find a complete day.
    """
    results = []
    for group, product, spec in specs_to_check(live_cfg, include_check):
        service = str(spec.get("service", ""))
        res = SourceCheck(group, product, service, str(spec.get("dataset_id", "")))
        results.append(res)
        probe = probes.get(service)
        if probe is None:
            res.note = f"unknown service {service!r}"
            continue
        try:
            found = probe(spec, today, bbox)
        except Exception as e:  # report and carry on with the other sources
            res.note = f"{type(e).__name__}: {e}"[:300]
            continue
        newest = found.get("newest_day")
        if newest is not None:
            newest_ts = pd.Timestamp(newest).normalize()
            res.newest_day = str(newest_ts.date())
            res.lag_days = int((today.normalize() - newest_ts).days)
        wanted = [str(v) for v in spec.get("variables") or []]
        have = found.get("variables")
        if have is not None and wanted:
            res.missing_variables = [v for v in wanted if v not in set(have)]
        res.note = str(found.get("note", ""))
        res.ok = res.newest_day is not None and not res.missing_variables
        if res.newest_day is None and not res.note:
            res.note = "no days found"
        if res.missing_variables:
            res.note = (res.note + "; " if res.note else "") + "variables not in the dataset"
        if (
            res.ok
            and group == "live"
            and max_lag_days is not None
            and res.lag_days is not None
            and res.lag_days > max_lag_days
        ):
            res.ok = False
            res.note = f"newest day is {res.lag_days} days old (max_lag_days {max_lag_days})"
    return results


def markdown(results: list[SourceCheck], today: pd.Timestamp) -> str:
    lines = [
        f"## Live sources check ({today.date()})",
        "",
        "| Group | Product | Service | Dataset ID | OK | Newest day | Lag (days) | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        missing = f" missing: {', '.join(r.missing_variables)}" if r.missing_variables else ""
        lines.append(
            f"| {r.group} | {r.product} | {r.service} | `{r.dataset_id}` | "
            f"{'yes' if r.ok else '**no**'} | {r.newest_day or '-'} | "
            f"{'-' if r.lag_days is None else r.lag_days} | {r.note}{missing} |"
        )
    bad = [r for r in results if not r.ok]
    lines += ["", f"{len(results) - len(bad)} of {len(results)} sources OK."]
    return "\n".join(lines) + "\n"


def write_report(results: list[SourceCheck], today: pd.Timestamp, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"today": str(today.date()), "sources": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )


# --- Real probes (network and credentials; not run in the tests) ---------------------------


def probe_copernicus(
    spec: dict[str, Any], today: pd.Timestamp, bbox: tuple[float, float, float, float]
) -> dict[str, Any]:
    """Open the dataset lazily (metadata only) and read its time axis and variables."""
    import copernicusmarine

    lon_min, lon_max, lat_min, lat_max = bbox
    kwargs: dict[str, Any] = {
        "dataset_id": spec["dataset_id"],
        "minimum_longitude": lon_min,
        "maximum_longitude": lon_min + 1.0,
        "minimum_latitude": lat_min,
        "maximum_latitude": lat_min + 1.0,
    }
    if spec.get("dataset_version"):
        kwargs["dataset_version"] = str(spec["dataset_version"])
    if spec.get("max_depth") is not None:
        kwargs["maximum_depth"] = float(spec["max_depth"])
    ds = copernicusmarine.open_dataset(**kwargs)
    times = pd.to_datetime(ds["time"].values)
    return {
        "newest_day": times.max() if len(times) else None,
        "variables": list(ds.data_vars),
        "note": f"from {times.min().date()}" if len(times) else "",
    }


def probe_earthdata(
    spec: dict[str, Any], today: pd.Timestamp, bbox: tuple[float, float, float, float]
) -> dict[str, Any]:
    """Search the last 60 days of granules (metadata only) and take the newest one's end."""
    import earthaccess

    earthaccess.login(strategy="environment")
    start = today - pd.Timedelta(days=60)
    results = earthaccess.search_data(
        short_name=spec["dataset_id"],
        temporal=(str(start.date()), str(today.date())),
        bounding_box=(bbox[0], bbox[2], bbox[1], bbox[3]),
    )
    if not results:
        # Maybe the collection exists but stopped earlier, or the name is wrong.
        collections = earthaccess.search_datasets(short_name=spec["dataset_id"])
        if not collections:
            return {"newest_day": None, "note": "no collection with this short name"}
        return {"newest_day": None, "note": "collection exists but has no granules in 60 days"}
    ends = []
    for g in results:
        rng = (g.get("umm", {}).get("TemporalExtent", {}) or {}).get("RangeDateTime", {}) or {}
        stamp = rng.get("EndingDateTime") or rng.get("BeginningDateTime")
        if stamp:
            ts = pd.Timestamp(stamp)
            ends.append(ts.tz_convert(None) if ts.tzinfo is not None else ts)
    newest = max(ends) if ends else None
    # A day's granule ends at 00:00 of the next day; count it as that day.
    if newest is not None and newest == newest.normalize():
        newest = newest - pd.Timedelta(days=1)
    return {"newest_day": newest, "note": f"{len(results)} granules in 60 days"}


def probe_cds(
    spec: dict[str, Any],
    today: pd.Timestamp,
    bbox: tuple[float, float, float, float],
    *,
    lags: tuple[int, ...] = (5, 6, 7, 8),
) -> dict[str, Any]:
    """Ask CDS for one day over a 1° box, newest first; the first day that works is the answer."""
    import cdsapi

    from oceanscope_ml.data.build_store import ERA5_DAILY_VARIABLES, _open_cds_download

    client = cdsapi.Client()
    lon_min, _, lat_min, _ = bbox
    errors = []
    with tempfile.TemporaryDirectory(prefix="oceanscope-cds-") as tmp:
        for lag in lags:
            day = today.normalize() - pd.Timedelta(days=lag)
            target = Path(tmp) / f"{day.date()}.nc"
            request = {
                "product_type": "reanalysis",
                "year": str(day.year),
                "month": f"{day.month:02d}",
                "day": [f"{day.day:02d}"],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": [lat_min + 1.0, lon_min, lat_min, lon_min + 1.0],
                "format": "netcdf",
                "variable": ERA5_DAILY_VARIABLES,
            }
            try:
                client.retrieve(spec["dataset_id"], request, str(target))
                ds = _open_cds_download(target)
            except Exception as e:
                errors.append(f"{day.date()}: {type(e).__name__}")
                logger.info("CDS %s not available: %s", day.date(), e)
                continue
            return {
                "newest_day": day,
                "variables": None,
                "note": f"{len(ds.data_vars)} variables" + (f"; tried {errors}" if errors else ""),
            }
    return {"newest_day": None, "note": f"no day in lags {list(lags)}: {errors}"}


def probe_arco(
    spec: dict[str, Any], today: pd.Timestamp, bbox: tuple[float, float, float, float]
) -> dict[str, Any]:
    """Read ARCO ERA5's metadata: the last ERA5T day (``valid_time_stop_era5t``)."""
    import xarray as xr

    from oceanscope_ml.data.build_store import ARCO_ERA5_URL, ARCO_ERA5_VARIABLES

    ds = xr.open_zarr(ARCO_ERA5_URL, chunks=None, consolidated=True, zarr_format=2)
    stop = ds.attrs.get("valid_time_stop_era5t") or ds.attrs.get("valid_time_stop")
    missing = [v for v in ARCO_ERA5_VARIABLES if v not in ds.data_vars]
    return {
        "newest_day": stop,
        "variables": None,
        "note": f"final ERA5 to {ds.attrs.get('valid_time_stop')}"
        + (f"; missing {missing}" if missing else ""),
    }


REAL_PROBES: dict[str, Probe] = {
    "copernicusmarine": probe_copernicus,
    "earthdata": probe_earthdata,
    "cds": probe_cds,
    "arco": probe_arco,
}
