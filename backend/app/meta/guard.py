"""Number-grounding guard for guide answers (PRD_BACKEND §6.4).

Every number in the answer that has a decimal point or a unit (°C, m, km, %) must appear,
after rounding to the precision shown, in a data tool result of the same request or in the
request context. UI tool arguments never count as sources.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

FALLBACK_PREFIX = "I couldn't verify that answer against the data; here are the raw results:"

_UNIT = r"(?:°\s?C|ºC|degC|degrees?\s+C(?:elsius)?|km\b|m\b|metres?\b|meters?\b|%)"
_NUMBER_RE = re.compile(
    r"(?<![\w.,])[-−]?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<unit>\s?" + _UNIT + r")?"
)


def extract_checked_numbers(text: str) -> list[tuple[float, int]]:
    """Numbers the guard must verify, as (absolute value, decimals shown)."""
    found = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if decimals == 0 and not m.group("unit"):
            continue
        found.append((float(raw), decimals))
    return found


def collect_sources(values: Iterable[Any]) -> list[float]:
    """All numbers inside tool results / context, including numbers written in strings."""
    out: list[float] = []

    def walk(v: Any) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            out.append(abs(float(v)))
        elif isinstance(v, str):
            for m in _NUMBER_RE.finditer(v):
                out.append(float(m.group("num").replace(",", "")))
        elif isinstance(v, dict):
            for item in v.values():
                walk(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                walk(item)

    for value in values:
        walk(value)
    return out


def ungrounded_numbers(answer: str, sources: list[float]) -> list[float]:
    """Numbers in `answer` that match no source at the shown precision."""
    missing = []
    for value, decimals in extract_checked_numbers(answer):
        if not any(abs(round(abs(s), decimals) - value) < 1e-9 for s in sources):
            missing.append(value)
    return missing


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
