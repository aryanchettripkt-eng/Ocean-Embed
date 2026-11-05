"""Guide tools: data tools and UI tools (PRD_BACKEND §6.2).

Data tools return numbers and call the same `OutputStore` methods as the HTTP endpoints.
UI tools return no numbers: their arguments are validated here and, if valid, become
`ui_actions` for the browser. Invalid arguments come back to the model as a tool error and are
never forwarded. Tool errors are returned as `{"error": {"code", "message"}}`, never raised.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from app.data.events import load_events
from app.data.grid import snap_coordinate
from app.data.store import OutputStore, validate_date
from app.errors import APIError
from app.schemas.caveats import AI_CAVEAT, ARGO_CAVEAT, GLORYS_CAVEAT, LIVE_CAVEAT, SYNTHETIC_CAVEAT

DATA_TOOLS = {
    "get_temperature",
    "get_temperature_profile",
    "get_region_summary",
    "get_section",
    "get_validation_metrics",
    "get_dataset_info",
    "list_events",
}
UI_TOOLS = {"set_view", "select_point", "slice_at", "open_event"}

REGIONS = ("arabian_sea", "bay_of_bengal", "equatorial")
FIELDS = ("pred", "glorys", "error")
BBOX_KEYS = ("lat_min", "lat_max", "lon_min", "lon_max")
ISOTHERM_C = 20.0


class ToolArgError(Exception):
    """Bad tool arguments; returned to the model as a tool error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ToolOutcome:
    kind: Literal["data", "ui", "unknown"]
    result: dict[str, Any]
    ui_action: dict[str, Any] | None = None


def _fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_LAT = {"type": "number", "description": "Latitude in degrees north"}
_LON = {"type": "number", "description": "Longitude in degrees east"}
_DATE = {"type": "string", "description": "UTC day, YYYY-MM-DD"}
_DEPTH = {"type": "number", "description": "Depth in metres; snapped to the nearest model depth"}
_FIELD = {
    "type": "string",
    "enum": list(FIELDS),
    "description": "pred = model prediction, glorys = GLORYS reanalysis, error = pred - glorys",
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn(
        "get_temperature",
        "Temperature at one point, date and depth (model prediction and GLORYS).",
        {"lat": _LAT, "lon": _LON, "date": _DATE, "depth_m": _DEPTH},
        ["lat", "lon", "date", "depth_m"],
    ),
    _fn(
        "get_temperature_profile",
        "Temperature at all model depths for one point and date.",
        {"lat": _LAT, "lon": _LON, "date": _DATE},
        ["lat", "lon", "date"],
    ),
    _fn(
        "get_region_summary",
        "Area-weighted mean, min, max and std over a named region or a bounding box.",
        {
            "region": {"type": "string", "enum": list(REGIONS)},
            "lat_min": {"type": "number"},
            "lat_max": {"type": "number"},
            "lon_min": {"type": "number"},
            "lon_max": {"type": "number"},
            "date": _DATE,
            "depth_m": _DEPTH,
            "field": _FIELD,
        },
        ["date", "depth_m"],
    ),
    _fn(
        "get_section",
        "Vertical section along a latitude (east-west) or a longitude (north-south): min/max "
        "per depth and the model depth level where temperature first drops below 20 °C.",
        {"date": _DATE, "field": _FIELD, "lat": _LAT, "lon": _LON},
        ["date"],
    ),
    _fn(
        "get_validation_metrics",
        "Validation-split accuracy (RMSE, MAE, bias vs GLORYS) of the model and baselines.",
        {
            "depth_m": _DEPTH,
            "group_by": {"type": "string", "enum": ["depth", "season", "region"]},
        },
        [],
    ),
    _fn(
        "get_dataset_info",
        "Dates, depths, region, splits and caveats of the dataset.",
        {},
        [],
    ),
    _fn("list_events", "Curated ocean events (cyclones, etc.) with their dates.", {}, []),
    _fn(
        "set_view",
        "Move the explorer: date, depth, field, view mode and camera.",
        {
            "date": _DATE,
            "depth_m": _DEPTH,
            "field": _FIELD,
            "mode": {"type": "string", "enum": ["map", "3d", "layers"]},
            "center": {
                "type": "object",
                "properties": {"lat": _LAT, "lon": _LON},
                "required": ["lat", "lon"],
            },
            "zoom": {"type": "number", "minimum": 3, "maximum": 8},
            "bearing": {"type": "number", "minimum": -180, "maximum": 180},
        },
        [],
    ),
    _fn(
        "select_point",
        "Select a grid cell and open its details panel.",
        {"lat": _LAT, "lon": _LON},
        ["lat", "lon"],
    ),
    _fn(
        "slice_at",
        "Place the 3D slice along a latitude or a longitude (give exactly one).",
        {"lat": _LAT, "lon": _LON},
        [],
    ),
    _fn(
        "open_event",
        "Open the time machine on a curated event.",
        {
            "event_id": {"type": "string"},
            "phase": {"type": "string", "enum": ["before", "during", "after"]},
        },
        ["event_id"],
    ),
]


class ToolRunner:
    """Runs tool calls for one guide request."""

    def __init__(self, store: OutputStore, events: list[dict[str, Any]] | None = None) -> None:
        self.store = store
        manifest = store.loader.manifest
        assert manifest is not None
        self.manifest = manifest
        self.depths = [float(d) for d in manifest["depths_m"]]
        self.region = manifest["region"]
        self.available = manifest["available_dates"]
        self.test_range = manifest["splits"]["test"]
        self.test_evaluated = bool(manifest.get("test_evaluated", False))
        raw_events = load_events() if events is None else events
        self.events = [e for e in raw_events if self._event_visible(e)]

    # -- dispatch ---------------------------------------------------------------

    def run(self, name: str, raw_args: str) -> ToolOutcome:
        kind: Literal["data", "ui", "unknown"] = (
            "data" if name in DATA_TOOLS else "ui" if name in UI_TOOLS else "unknown"
        )
        if kind == "unknown":
            return ToolOutcome(kind, _err("unknown_tool", f"No tool named '{name}'"))
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return ToolOutcome(kind, _err("invalid_arguments", "Arguments must be a JSON object"))
        try:
            if kind == "data":
                return ToolOutcome(kind, getattr(self, f"_{name}")(args))
            action = getattr(self, f"_{name}")(args)
            return ToolOutcome(kind, {"ok": True, "applied": action}, ui_action=action)
        except ToolArgError as exc:
            return ToolOutcome(kind, _err(exc.code, exc.message))
        except APIError as exc:
            return ToolOutcome(kind, _err(exc.code, exc.message))

    # -- shared validation ------------------------------------------------------

    def _snap_depth(self, value: Any) -> float:
        """Snap to the nearest model depth if within 10 % (at least 1 m) of it."""
        depth = _number(value, "depth_m")
        nearest = min(self.depths, key=lambda d: abs(d - depth))
        if abs(nearest - depth) > max(1.0, 0.1 * nearest):
            raise ToolArgError(
                "bad_depth",
                f"depth_m {depth} is not a model depth; use one of {self.depths}",
            )
        return nearest

    def _in_sealed_test(self, date: str) -> bool:
        # Applies to data tools as well as UI tools: test-period values (and above all the
        # error against GLORYS) must not reach the user while the test split is sealed.
        return not self.test_evaluated and self.test_range[0] <= date <= self.test_range[1]

    def _check_date(self, value: Any) -> str:
        if not isinstance(value, str):
            raise ToolArgError("invalid_date_format", "date must be a YYYY-MM-DD string")
        validate_date(value)
        if self._in_sealed_test(value):
            raise ToolArgError(
                "sealed_test_period", "Dates in the sealed test period cannot be shown"
            )
        if not self.available["start"] <= value <= self.available["end"]:
            raise ToolArgError(
                "date_unavailable",
                f"date must be between {self.available['start']} and {self.available['end']}",
            )
        return value

    def _snap_point(self, lat_v: Any, lon_v: Any) -> tuple[float, float]:
        ds = self.store.get_predictions_dataset()
        lat = _number(lat_v, "lat")
        lon = _number(lon_v, "lon")
        reg = self.region
        _, s_lat = snap_coordinate(lat, ds.coords["lat"].values, reg["lat_min"], reg["lat_max"])
        _, s_lon = snap_coordinate(lon, ds.coords["lon"].values, reg["lon_min"], reg["lon_max"])
        return s_lat, s_lon

    def _event_visible(self, event: dict[str, Any]) -> bool:
        dates = [str(event.get(k, "")) for k in ("before", "start", "end", "after")]
        if not all(self.available["start"] <= d <= self.available["end"] for d in dates):
            return False
        return not any(self._in_sealed_test(d) for d in dates)

    def clean_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """The explorer context with every field validated like a tool argument. Invalid fields
        are dropped, so user-sent values never reach the prompt or count as number sources
        unless they are a real date, a model depth and a point in the region."""
        out: dict[str, Any] = {}
        checks: dict[str, Any] = {
            "date": self._check_date,
            "depth_m": self._snap_depth,
        }
        for key, check in checks.items():
            if key in context:
                try:
                    out[key] = check(context[key])
                except (ToolArgError, APIError):
                    pass
        if "lat" in context and "lon" in context:
            try:
                out["lat"], out["lon"] = self._snap_point(context["lat"], context["lon"])
            except (ToolArgError, APIError):
                pass
        return out

    # -- data tools ------------------------------------------------------------

    def _get_temperature(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"lat", "lon", "date", "depth_m"})
        depth = self._snap_depth(_req(args, "depth_m"))
        prof = self.store.get_profile(
            _number(_req(args, "lat"), "lat"),
            _number(_req(args, "lon"), "lon"),
            self._check_date(_req(args, "date")),
        )
        i = self.depths.index(depth)
        return {
            "date": prof.date,
            "lat": prof.lat,
            "lon": prof.lon,
            "depth_m": depth,
            "split": prof.split,
            "coverage": prof.coverage,
            "pred_c": prof.pred[i],
            "glorys_c": prof.glorys[i],
            "error_c": prof.error[i],
            "synthetic": prof.synthetic,
        }

    def _get_temperature_profile(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"lat", "lon", "date"})
        prof = self.store.get_profile(
            _number(_req(args, "lat"), "lat"),
            _number(_req(args, "lon"), "lon"),
            self._check_date(_req(args, "date")),
        )
        return prof.model_dump(mode="json", exclude={"run_id"})

    def _get_region_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(
            args, {"region", "lat_min", "lat_max", "lon_min", "lon_max", "date", "depth_m", "field"}
        )
        region = args.get("region")
        if region is not None and region not in REGIONS:
            raise ToolArgError("invalid_region", f"region must be one of {', '.join(REGIONS)}")
        bbox: list[float | None] = [None, None, None, None]
        if region is None:
            bbox = [_number(_req(args, k), k) for k in BBOX_KEYS]
        summary = self.store.get_region_summary(
            date=self._check_date(_req(args, "date")),
            depth_m=self._snap_depth(_req(args, "depth_m")),
            lat_min=bbox[0],
            lat_max=bbox[1],
            lon_min=bbox[2],
            lon_max=bbox[3],
            region=region,
            field=_field(args),
        )
        out = summary.model_dump(mode="json", exclude={"run_id"})
        out["region"] = region
        return out

    def _get_section(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"date", "field", "lat", "lon"})
        lat = args.get("lat")
        lon = args.get("lon")
        sec = self.store.get_section(
            date=self._check_date(_req(args, "date")),
            field=_field(args),
            lat=None if lat is None else _number(lat, "lat"),
            lon=None if lon is None else _number(lon, "lon"),
        )
        per_depth = []
        for depth, row in zip(sec.depths_m, sec.values):
            vals = [v for v in row if v is not None]
            per_depth.append(
                {
                    "depth_m": depth,
                    "min_c": min(vals) if vals else None,
                    "max_c": max(vals) if vals else None,
                }
            )
        out: dict[str, Any] = {
            "date": sec.date,
            "field": sec.field,
            "split": sec.split,
            "axis": sec.axis,
            "fixed": sec.fixed,
            "per_depth": per_depth,
            "synthetic": sec.synthetic,
        }
        if sec.field != "error":
            # The first model level below 20 °C under a level at or above it; no interpolation.
            iso: list[dict[str, Any]] = []
            for j, pos in enumerate(sec.positions):
                column = [row[j] for row in sec.values]
                level = None
                for k in range(1, len(column)):
                    above, below = column[k - 1], column[k]
                    if above is not None and below is not None:
                        if above >= ISOTHERM_C > below:
                            level = sec.depths_m[k]
                            break
                iso.append({sec.axis: pos, "depth_m": level})
            out["first_level_below_20c"] = iso
            out["isotherm_c"] = ISOTHERM_C
        return out

    def _get_validation_metrics(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"depth_m", "group_by"})
        group_by = args.get("group_by", "depth")
        if group_by not in ("depth", "season", "region"):
            raise ToolArgError("invalid_group_by", "group_by must be depth, season or region")
        metrics = self.store.get_metrics(split="val", group_by=group_by)
        rows = [r.model_dump(mode="json") for r in metrics.rows]
        if args.get("depth_m") is not None:
            depth = self._snap_depth(args["depth_m"])
            rows = [r for r in rows if isinstance(r["depth_m"], float) and r["depth_m"] == depth]
        active = str(self.manifest["model_name"])
        rows.sort(key=lambda r: r["model"] != active)
        return {
            "split": "val",
            "group_by": group_by,
            "model_name": active,
            "reference": "GLORYS reanalysis",
            "rows": rows,
            "synthetic": metrics.synthetic,
        }

    def _get_dataset_info(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, set())
        m = self.manifest
        return {
            "run_id": m["active_run_id"],
            "synthetic": m["synthetic"],
            "model_name": m["model_name"],
            "region": m["region"],
            "named_regions": list(REGIONS),
            "grid_resolution_deg": m["grid_resolution_deg"],
            "depths_m": self.depths,
            "available_dates": self.available,
            "splits": m["splits"],
            "test_evaluated": self.test_evaluated,
            "caveats": {
                "glorys": GLORYS_CAVEAT,
                "argo": ARGO_CAVEAT,
                "live": LIVE_CAVEAT,
                "synthetic": SYNTHETIC_CAVEAT,
                "ai": AI_CAVEAT,
            },
        }

    def _list_events(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, set())
        keys = ("id", "title", "kind", "start", "end", "before", "after", "bbox", "summary")
        return {"events": [{k: e.get(k) for k in keys} for e in self.events]}

    # -- UI tools (validated; return the normalised action) --------------------

    def _set_view(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"date", "depth_m", "field", "mode", "center", "zoom", "bearing"})
        if not args:
            raise ToolArgError("invalid_arguments", "set_view needs at least one argument")
        out: dict[str, Any] = {}
        if "date" in args:
            out["date"] = self._check_date(args["date"])
        if "depth_m" in args:
            out["depth_m"] = self._snap_depth(args["depth_m"])
        if "field" in args:
            out["field"] = _field(args)
            if out["field"] != "pred" and "date" in out:
                if self.store.get_date_split(out["date"]) == "live":
                    raise ToolArgError(
                        "reference_unavailable", "GLORYS is not available on live dates"
                    )
        if "mode" in args:
            if args["mode"] not in ("map", "3d", "layers"):
                raise ToolArgError("invalid_mode", "mode must be map, 3d or layers")
            out["mode"] = args["mode"]
        if "center" in args:
            center = args["center"]
            if not isinstance(center, dict) or set(center) != {"lat", "lon"}:
                raise ToolArgError("invalid_center", "center must be {lat, lon}")
            lat, lon = self._snap_point(center["lat"], center["lon"])
            out["center"] = {"lat": lat, "lon": lon}
        if "zoom" in args:
            out["zoom"] = _ranged(args["zoom"], "zoom", 3, 8)
        if "bearing" in args:
            out["bearing"] = _ranged(args["bearing"], "bearing", -180, 180)
        return {"type": "set_view", "args": out}

    def _select_point(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"lat", "lon"})
        lat, lon = self._snap_point(_req(args, "lat"), _req(args, "lon"))
        return {"type": "select_point", "args": {"lat": lat, "lon": lon}}

    def _slice_at(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"lat", "lon"})
        if ("lat" in args) == ("lon" in args):
            raise ToolArgError("invalid_section", "Give exactly one of lat or lon")
        ds = self.store.get_predictions_dataset()
        reg = self.region
        key = "lat" if "lat" in args else "lon"
        _, snapped = snap_coordinate(
            _number(args[key], key), ds.coords[key].values, reg[f"{key}_min"], reg[f"{key}_max"]
        )
        return {"type": "slice_at", "args": {key: snapped}}

    def _open_event(self, args: dict[str, Any]) -> dict[str, Any]:
        _only(args, {"event_id", "phase"})
        event_id = _req(args, "event_id")
        if not any(e["id"] == event_id for e in self.events):
            raise ToolArgError("unknown_event", f"No event with id '{event_id}'")
        out: dict[str, Any] = {"event_id": event_id}
        if "phase" in args:
            if args["phase"] not in ("before", "during", "after"):
                raise ToolArgError("invalid_phase", "phase must be before, during or after")
            out["phase"] = args["phase"]
        return {"type": "open_event", "args": out}


# -- helpers --------------------------------------------------------------------


def is_tool_error(result: dict[str, Any]) -> bool:
    """Tool errors are `{"error": {...}}`; a profile's `error` list is data, not an error."""
    return isinstance(result.get("error"), dict)


def _err(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _only(args: dict[str, Any], allowed: set[str]) -> None:
    extra = set(args) - allowed
    if extra:
        raise ToolArgError("invalid_arguments", f"Unknown arguments: {', '.join(sorted(extra))}")


def _req(args: dict[str, Any], key: str) -> Any:
    if args.get(key) is None:
        raise ToolArgError("invalid_arguments", f"'{key}' is required")
    return args[key]


def _number(value: Any, name: str) -> float:
    # json.loads accepts NaN and Infinity, so model-written arguments can carry them.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ToolArgError("invalid_arguments", f"'{name}' must be a finite number")
    return float(value)


def _ranged(value: Any, name: str, lo: float, hi: float) -> float:
    num = _number(value, name)
    if not lo <= num <= hi:
        raise ToolArgError("invalid_arguments", f"'{name}' must be between {lo} and {hi}")
    return num


def _field(args: dict[str, Any]) -> Literal["pred", "glorys", "error"]:
    value = args.get("field", "pred")
    if value == "pred":
        return "pred"
    if value == "glorys":
        return "glorys"
    if value == "error":
        return "error"
    raise ToolArgError("invalid_field", "field must be pred, glorys or error")
