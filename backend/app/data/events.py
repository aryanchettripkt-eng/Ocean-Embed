"""Curated events for the time machine (PRD_BACKEND §4.14).

`events.json` is written and approved by people only (PHASES P7-E). It is validated at startup
against the active manifest and the server fails fast if it is wrong. A missing file means no
events yet. `GET /events` and the guide's event tools both read this module.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVENTS_PATH = Path(__file__).resolve().parent / "events.json"

# A suggested depth snaps to the nearest model depth when it is within this distance.
DEPTH_SNAP_ABS_M = 1.0
DEPTH_SNAP_REL = 0.10

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FIELDS = {
    "id",
    "title",
    "kind",
    "start",
    "end",
    "before",
    "after",
    "bbox",
    "suggested_depths_m",
    "summary",
    "source_url",
    "source_name",
}
_BBOX_FIELDS = {"lat_min", "lat_max", "lon_min", "lon_max"}


class EventsFileError(ValueError):
    """The curated events file fails validation. The server must not start with it."""


def load_events(path: Path | None = None) -> list[dict[str, Any]]:
    """Return the raw events, or an empty list if the file does not exist yet."""
    target = path or EVENTS_PATH
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise EventsFileError(f"{target.name} is not valid JSON") from exc
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise EventsFileError(f"{target.name} must be an object with an 'events' list")
    return [e for e in events if isinstance(e, dict)]


def validate_events(
    raw: list[dict[str, Any]],
    *,
    region: dict[str, Any],
    depths_m: list[float],
    available: dict[str, str],
) -> list[dict[str, Any]]:
    """Check every event and return clean copies with suggested depths snapped to model depths.

    Raises EventsFileError on the first problem, with the event id and the rule it broke.
    """
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for index, event in enumerate(raw):
        where = str(event.get("id", f"#{index}"))
        if set(event) != _FIELDS:
            missing = sorted(_FIELDS - set(event))
            extra = sorted(set(event) - _FIELDS)
            raise EventsFileError(
                f"{where}: fields must match the schema (missing {missing}, extra {extra})"
            )
        if not isinstance(event["id"], str) or not _ID_RE.match(event["id"]):
            raise EventsFileError(f"{where}: id must be lowercase words joined by hyphens")
        if event["id"] in seen:
            raise EventsFileError(f"{where}: duplicate id")
        seen.add(event["id"])

        for key in ("title", "kind", "source_name", "summary"):
            if not isinstance(event[key], str) or not event[key].strip():
                raise EventsFileError(f"{where}: {key} must be a non-empty string")
        if re.search(r"\d", event["summary"]):
            raise EventsFileError(f"{where}: summary must contain no numbers")
        url = event["source_url"]
        if not isinstance(url, str) or not url.startswith("https://") or re.search(r"\s", url):
            raise EventsFileError(f"{where}: source_url must be an https link")

        dates = {k: _parse_date(event[k], where, k) for k in ("before", "start", "end", "after")}
        if not (dates["before"] < dates["start"] <= dates["end"] < dates["after"]):
            raise EventsFileError(f"{where}: dates must satisfy before < start <= end < after")
        lo, hi = date.fromisoformat(available["start"]), date.fromisoformat(available["end"])
        for k, d in dates.items():
            if not lo <= d <= hi:
                raise EventsFileError(f"{where}: {k} lies outside the available dates")

        bbox = event["bbox"]
        if not isinstance(bbox, dict) or set(bbox) != _BBOX_FIELDS:
            raise EventsFileError(f"{where}: bbox must have lat_min, lat_max, lon_min, lon_max")
        if not all(isinstance(v, int | float) for v in bbox.values()):
            raise EventsFileError(f"{where}: bbox values must be numbers")
        if bbox["lat_min"] >= bbox["lat_max"] or bbox["lon_min"] >= bbox["lon_max"]:
            raise EventsFileError(f"{where}: bbox minimum must be below its maximum")
        if (
            bbox["lat_min"] < region["lat_min"]
            or bbox["lat_max"] > region["lat_max"]
            or bbox["lon_min"] < region["lon_min"]
            or bbox["lon_max"] > region["lon_max"]
        ):
            raise EventsFileError(f"{where}: bbox must lie inside the modelled region")

        suggested = event["suggested_depths_m"]
        if not isinstance(suggested, list) or not suggested:
            raise EventsFileError(f"{where}: suggested_depths_m must be a non-empty list")
        snapped: list[float] = []
        for depth in suggested:
            if not isinstance(depth, int | float):
                raise EventsFileError(f"{where}: suggested depths must be numbers")
            nearest = min(depths_m, key=lambda d: abs(d - float(depth)))
            if abs(nearest - depth) > max(DEPTH_SNAP_ABS_M, DEPTH_SNAP_REL * abs(depth)):
                raise EventsFileError(
                    f"{where}: suggested depth {depth} m has no model depth nearby"
                )
            if nearest not in snapped:
                snapped.append(float(nearest))

        copy = dict(event)
        copy["bbox"] = {k: float(bbox[k]) for k in sorted(_BBOX_FIELDS)}
        copy["suggested_depths_m"] = snapped
        cleaned.append(copy)
    return cleaned


def visible_events(
    events: list[dict[str, Any]], *, test_range: list[str], test_evaluated: bool
) -> list[dict[str, Any]]:
    """Drop events with any date in the sealed test period until the test has been run."""
    if test_evaluated:
        return list(events)
    lo, hi = date.fromisoformat(test_range[0]), date.fromisoformat(test_range[1])
    out = []
    for event in events:
        dates = [date.fromisoformat(str(event[k])) for k in ("before", "start", "end", "after")]
        if not any(lo <= d <= hi for d in dates):
            out.append(event)
    return out


def check_events_file(manifest: dict[str, Any], path: Path | None = None) -> int:
    """Startup check: validate the events file against the manifest. Returns the event count."""
    raw = load_events(path)
    validate_events(
        raw,
        region=manifest["region"],
        depths_m=[float(d) for d in manifest["depths_m"]],
        available=manifest["available_dates"],
    )
    logger.info("Curated events validated: %d event(s)", len(raw))
    return len(raw)


def _parse_date(value: Any, where: str, key: str) -> date:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise EventsFileError(f"{where}: {key} must be a YYYY-MM-DD date")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise EventsFileError(f"{where}: {key} is not a real calendar date") from None
