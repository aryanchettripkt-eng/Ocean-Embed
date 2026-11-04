"""Number-grounding guard for guide answers (PRD_BACKEND §6.4), v2.

Every number in the answer that has a decimal point or a unit must appear, after rounding to
the precision shown, in a data tool result of the same request or in the validated request
context. UI tool arguments never count as sources. v2 (P7-R findings) tightens the match:

- Kind: a °C number must match a temperature field, an "m" number a depth field, a "%" number
  a percentage field, a "°N/°E" number a coordinate. Any field still grounds a bare decimal.
- Sign: an explicit "-" or "+" must match a source of that sign ("bias is -0.07 °C" fails when
  the bias is +0.07). A bare magnitude may still describe a negative value in words.
- Units: bare "°" and "degrees" count as °C. °F, kelvin and units the data does not have (PSU,
  ppt, feet, miles, knots) never ground: the guide gives temperatures in °C only.
- Dates: every YYYY-MM-DD in the answer must appear in a source or the context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

FALLBACK_PREFIX = "I couldn't verify that answer against the data; here are the raw results:"

Kind = Literal["temp", "depth", "pct", "coord", "other"]
UnitKind = Literal["temp", "depth", "pct", "coord", "dist", "forbidden", "none"]

_UNIT_KINDS: list[tuple[UnitKind, str]] = [
    ("coord", r"[°º]\s?[NSEW]\b|degrees?\s+(?:north|south|east|west)\b"),
    ("forbidden", r"[°º]\s?F\b|deg\s?F\b|degrees?\s+F(?:ahrenheit)?\b|fahrenheit\b|kelvin\b|K\b"),
    ("forbidden", r"psu\b|ppt\b|g/kg\b|ft\b|feet\b|foot\b|mi\b|miles?\b|knots?\b|kn\b|nm\b"),
    ("temp", r"[°º]\s?C\b|degC\b|deg\s?C\b|degrees?\s+C(?:elsius)?\b|celsius\b|[°º]|degrees?\b"),
    ("dist", r"km\b|kilomet(?:re|er)s?\b"),
    ("depth", r"m\b|metres?\b|meters?\b"),
    ("pct", r"%|percent\b"),
]
_UNIT = "|".join(f"(?P<u_{i}>{pat})" for i, (_, pat) in enumerate(_UNIT_KINDS))
_NUMBER_RE = re.compile(
    r"(?<![\w.,])(?P<sign>[-−+])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?:" + _UNIT + r"))?",
    re.I,
)
_PLAIN_NUMBER_RE = re.compile(r"[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−]?\d+(?:\.\d+)?")
_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")

_TEMP_KEYS = {"pred", "glorys", "error", "mean", "min", "max", "std", "values"}
_DEPTH_KEYS = {"depth_m", "depths_m", "suggested_depths_m"}
_COORD_KEYS = {"lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max"}


@dataclass(frozen=True)
class Found:
    value: float  # absolute value as written
    decimals: int
    sign: int  # -1, +1, or 0 when no sign was written
    unit: UnitKind


@dataclass(frozen=True)
class Source:
    value: float  # signed
    kind: Kind


def _key_kind(key: str | None) -> Kind:
    if key is None:
        return "other"
    k = key.lower()
    if k in _TEMP_KEYS or k.endswith("_c"):
        return "temp"
    if k in _DEPTH_KEYS:
        return "depth"
    if k in _COORD_KEYS:
        return "coord"
    if k.endswith("_pct"):
        return "pct"
    return "other"


def find_numbers(text: str) -> list[Found]:
    """Numbers the guard must verify. Dates are taken out first (they have their own check)."""
    text = _DATE_RE.sub(" ", text)
    found = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        unit: UnitKind = "none"
        for i, (kind, _) in enumerate(_UNIT_KINDS):
            if m.group(f"u_{i}"):
                unit = kind
                break
        if decimals == 0 and unit == "none":
            continue
        sign_txt = m.group("sign")
        sign = 0 if not sign_txt else (1 if sign_txt == "+" else -1)
        found.append(Found(float(raw), decimals, sign, unit))
    return found


def extract_checked_numbers(text: str) -> list[tuple[float, int]]:
    """Numbers the guard must verify, as (absolute value, decimals shown)."""
    return [(f.value, f.decimals) for f in find_numbers(text)]


def collect_typed_sources(values: Iterable[Any]) -> tuple[list[Source], set[str]]:
    """All numbers inside tool results / context, tagged by the field they came from, plus
    every date string found. Numbers written inside strings count as kind "other"."""
    out: list[Source] = []
    dates: set[str] = set()

    def walk(v: Any, kind: Kind) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            out.append(Source(float(v), kind))
        elif isinstance(v, str):
            dates.update(_DATE_RE.findall(v))
            for m in _PLAIN_NUMBER_RE.finditer(_DATE_RE.sub(" ", v)):
                num = m.group(0).replace(",", "").replace("−", "-")
                out.append(Source(float(num), "other"))
        elif isinstance(v, dict):
            for key, item in v.items():
                walk(item, _key_kind(str(key)))
        elif isinstance(v, (list, tuple)):
            for item in v:
                walk(item, kind)

    for value in values:
        walk(value, "other")
    return out, dates


def collect_sources(values: Iterable[Any]) -> list[float]:
    """All numbers inside tool results / context, untyped (kept for callers of guard v1)."""
    return [abs(s.value) for s in collect_typed_sources(values)[0]]


def _matches(f: Found, s: Source) -> bool:
    if abs(round(abs(s.value), f.decimals) - f.value) >= 1e-9:
        return False
    if f.sign and s.value != 0 and (s.value > 0) != (f.sign > 0):
        return False
    if f.unit in ("none", "dist"):
        return True
    return s.kind == f.unit


def ungrounded_numbers(answer: str, sources: list[float] | list[Source]) -> list[float]:
    """Numbers in `answer` that match no source. Plain floats are treated as untyped sources
    that match any unit (guard v1 behaviour); `Source` items get the v2 kind and sign rules."""
    typed = [s if isinstance(s, Source) else None for s in sources]
    missing = []
    for f in find_numbers(answer):
        if f.unit == "forbidden":
            missing.append(f.value)
            continue
        ok = False
        for raw, s in zip(sources, typed):
            if s is None:
                assert isinstance(raw, (int, float))
                ok = abs(round(abs(raw), f.decimals) - f.value) < 1e-9
            else:
                ok = _matches(f, s)
            if ok:
                break
        if not ok:
            missing.append(f.value)
    return missing


def ungrounded_dates(answer: str, dates: set[str]) -> list[str]:
    return [d for d in _DATE_RE.findall(answer) if d not in dates]


def leaks_prompt(answer: str, prompt: str) -> bool:
    """True if the answer repeats a long line of the system prompt (an obeyed injection)."""
    low = answer.lower()
    return any(
        len(line.strip()) > 50 and line.strip().lower()[:60] in low for line in prompt.splitlines()
    )


def fallback_answer(data_calls: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> str:
    """The fixed fallback: prefix plus a compact rendering of the data tool results."""
    lines = [FALLBACK_PREFIX]
    for name, args, result in data_calls:
        rendered = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        if len(rendered) > 600:
            rendered = rendered[:600] + "…"
        lines.append(f"- {name}({json.dumps(args, separators=(',', ':'))}): {rendered}")
    if len(lines) == 1:
        lines.append("- (no data tools were called)")
    return "\n".join(lines)
