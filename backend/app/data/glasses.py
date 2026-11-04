"""Glasses ocean guide: depth parsing, depth blending and the spoken answer (PRD_BACKEND §4.18).

Pure functions. Every number in the spoken answer comes from the prediction store (through the
caller) or from the user's own question; nothing here invents a value.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# "500 m", "500 metres", "500-meter", "500m" (the unit is required: a bare number is ambiguous).
_DEPTH_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*-?\s*(?:m|metres?|meters?|mtrs?)\b", re.I)
# Words right after a number that make it a distance, not a depth ("100 m from the shore").
_DISTANCE_AFTER_RE = re.compile(
    r"^\s*(?:from|away|off|out|offshore|ahead|in front|into the sea|to the)\b", re.I
)
_SURFACE_RE = re.compile(r"\b(?:surface|on top|at the top|skin)\b", re.I)

_POINTS = (
    "north",
    "north-north-east",
    "north-east",
    "east-north-east",
    "east",
    "east-south-east",
    "south-east",
    "south-south-east",
    "south",
    "south-south-west",
    "south-west",
    "west-south-west",
    "west",
    "west-north-west",
    "north-west",
    "north-north-west",
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
ROUGH_HEADING_DEG = 30.0


def parse_depth_m(question: str) -> float | None:
    """The depth the user asked about in metres, 0.0 for "surface", None if none was given.

    Numbers followed by "from", "away", "offshore" and similar are distances and are skipped.
    If several depths remain, the last one wins ("100 m out, at 500 m" -> 500).
    """
    depths: list[float] = []
    for m in _DEPTH_RE.finditer(question):
        if _DISTANCE_AFTER_RE.match(question[m.end() :]):
            continue
        depths.append(float(m.group(1).replace(",", ".")))
    if depths:
        return depths[-1]
    if _SURFACE_RE.search(question):
        return 0.0
    return None


def deep_enough(pred: np.ndarray, depths: list[float], depth_m: float) -> np.ndarray:
    """(lat, lon) mask of cells that have a predicted value at `depth_m`.

    `pred` is (depth, lat, lon). Values below the sea floor are NaN, so a cell is deep enough
    when the first model level at or below `depth_m` has a value.
    """
    idx = level_at_or_below(depths, depth_m)
    return np.asarray(~np.isnan(pred[idx]), dtype=bool)


def level_at_or_below(depths: list[float], depth_m: float) -> int:
    """Index of the first model level at or below `depth_m` (the last level if deeper)."""
    for i, d in enumerate(depths):
        if d >= depth_m - 1e-9:
            return i
    return len(depths) - 1


@dataclass(frozen=True)
class DepthValue:
    temperature_c: float
    depth_used_m: float
    levels_m: list[float]
    interpolated: bool


def value_at_depth(
    depths: list[float], values: list[float | None], depth_m: float
) -> DepthValue | None:
    """Temperature at `depth_m`: exact on a level, linear between the two levels around it,
    the top level above it. None if the cell has no value that deep."""
    valid = [(d, v) for d, v in zip(depths, values, strict=True) if v is not None]
    if not valid:
        return None
    for d, v in valid:
        if abs(d - depth_m) < 1e-6:
            return DepthValue(round(v, 1), d, [d], False)
    if depth_m <= valid[0][0]:
        d0, v0 = valid[0]
        return DepthValue(round(v0, 1), d0, [d0], False)
    for (d0, v0), (d1, v1) in zip(valid, valid[1:], strict=False):
        if d0 < depth_m < d1:
            w = (depth_m - d0) / (d1 - d0)
            return DepthValue(round(v0 + w * (v1 - v0), 1), depth_m, [d0, d1], True)
    return None


def deepest_value(depths: list[float], values: list[float | None]) -> DepthValue | None:
    """The deepest level with a value, or None."""
    for d, v in reversed(list(zip(depths, values, strict=True))):
        if v is not None:
            return DepthValue(round(v, 1), d, [d], False)
    return None


def compass_word(bearing_deg: float) -> str:
    """16-point compass name for a bearing, e.g. 75 -> "east-north-east"."""
    return _POINTS[int(math.floor((bearing_deg % 360.0) / 22.5 + 0.5)) % 16]


def date_words(date: str) -> str:
    """'2020-01-10' -> '10 January 2020'."""
    y, m, d = (int(p) for p in date.split("-"))
    return f"{d} {_MONTHS[m - 1]} {y}"


def _metres(depth_m: float) -> str:
    if depth_m < 1.0:
        return "the surface"
    return f"{depth_m:.0f} metres"


@dataclass(frozen=True)
class AnswerParts:
    distance_km: float
    bearing_deg: float
    date: str
    value: DepthValue
    depth_asked_m: float | None
    deepest_model_m: float
    too_shallow: bool
    max_km: float
    heading_accuracy_deg: float | None
    synthetic: bool


def spoken_answer(p: AnswerParts) -> str:
    """Two to four short sentences for text-to-speech. No markdown, units in words."""
    where = f"About {p.distance_km:.0f} km {compass_word(p.bearing_deg)} of you"
    temp = f"{p.value.temperature_c:.1f} degrees Celsius"
    sentences: list[str] = []
    if p.too_shallow and p.depth_asked_m is not None:
        sentences.append(
            f"The water within {p.max_km:.0f} km that way isn't "
            f"{_metres(min(p.depth_asked_m, p.deepest_model_m))} deep."
        )
        sentences.append(
            f"{where}, the deepest level with data is {_metres(p.value.depth_used_m)}, "
            f"where the model estimates {temp} for {date_words(p.date)}."
        )
    else:
        if p.depth_asked_m is not None and p.depth_asked_m > p.deepest_model_m + 1e-6:
            sentences.append(
                f"The model's deepest level is {_metres(p.deepest_model_m)}, so I used that."
            )
        sentences.append(
            f"{where}, at {_metres(p.value.depth_used_m)}, the model estimates {temp} "
            f"for {date_words(p.date)}."
        )
        if p.value.interpolated:
            a, b = p.value.levels_m
            sentences.append(f"That's blended from the {a:.0f} and {b:.0f} metre levels.")
        if p.depth_asked_m is None:
            sentences.append("Ask for a depth, like 200 metres, to hear about deeper water.")
    if p.heading_accuracy_deg is not None and p.heading_accuracy_deg > ROUGH_HEADING_DEG:
        sentences.append("Your compass reading is rough, so the direction may be off.")
    if p.synthetic:
        sentences.append("This is synthetic demo data, not model results.")
    return " ".join(sentences)
