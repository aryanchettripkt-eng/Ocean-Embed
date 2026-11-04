"""P7-R guide test harness: runs the question set through `run_guide` and checks the answers.

Two groups of checks per case:
- safety: must hold for every client (mock, worst-case script, real model). A safety failure
  means something wrong reached, or could reach, the user.
- quality: what a good answer looks like. Only meaningful for the real model, because the
  offline mock plans its tools from keywords and ignores most of the question.

Nothing here calls the network unless the caller passes an `HttpMetaModelClient` itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.data.loader import DataLoader
from app.data.store import OutputStore
from app.meta.client import MetaModelClient, MetaUpstreamError
from app.meta.guard import FALLBACK_PREFIX
from app.meta.guide import run_guide
from app.meta.prompts import SYSTEM_PROMPT
from app.meta.tools import is_tool_error
from app.schemas.models import ChatRequest

HERE = Path(__file__).resolve().parent
QUESTIONS_PATH = HERE / "questions.json"
FIXTURES_DIR = HERE.parent.parent.parent / "fixtures" / "sample_outputs"

# Distinctive lines of the system prompt; any of them in an answer counts as a leak.
_LEAK_MARKERS = [
    line.strip()[:60]
    for line in SYSTEM_PROMPT.splitlines()
    if len(line.strip()) > 50 and not line.strip().endswith(":")
] + ["EXPLORER_CONTEXT_JSON"]
_FAHRENHEIT = re.compile(r"\d(?:[.,]\d+)?\s?(?:°\s?F|ºF|degrees?\s+F(?:ahrenheit)?)\b", re.I)
_TEMPERATURE_LIKE = re.compile(r"\d\s?(?:°|º|degrees?\b)", re.I)


@dataclass
class Case:
    id: str
    kind: str
    question: str
    context: dict[str, Any] | None
    safety: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Check:
    group: str  # "safety" | "quality"
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CaseResult:
    case: Case
    answer: str
    grounded: bool
    fallback: bool
    tool_calls: list[dict[str, Any]]
    ui_actions: list[dict[str, Any]]
    upstream_error: bool
    checks: list[Check]

    @property
    def safety_ok(self) -> bool:
        return all(c.ok for c in self.checks if c.group == "safety")

    @property
    def quality_ok(self) -> bool:
        return all(c.ok for c in self.checks if c.group == "quality")

    def failed(self, group: str) -> list[Check]:
        return [c for c in self.checks if c.group == group and not c.ok]


def load_cases(path: Path = QUESTIONS_PATH) -> list[Case]:
    data = json.loads(path.read_text(encoding="utf-8"))
    default_ctx = data["default_context"]
    cases = []
    for raw in data["cases"]:
        ctx = default_ctx if raw.get("context") == "default" else raw.get("context")
        cases.append(
            Case(
                id=raw["id"],
                kind=raw["kind"],
                question=raw["question"],
                context=dict(ctx) if ctx else None,
                safety=raw.get("safety", {}),
                quality=raw.get("quality", {}),
            )
        )
    return cases


def make_store(
    data_dir: Path = FIXTURES_DIR, sealed_test: tuple[str, str] | None = None
) -> OutputStore:
    """A store over `data_dir`. `sealed_test` moves the manifest's test split onto published
    dates, to simulate the real deployment where test-period predictions are published but the
    test split is still sealed (PRD_AI_ML §6, publish_splits: [val, test])."""
    store = OutputStore(DataLoader(data_dir=data_dir))
    store._ensure_loaded()
    manifest = store.loader.manifest
    assert manifest is not None
    if sealed_test is not None:
        manifest["splits"] = {**manifest["splits"], "test": list(sealed_test)}
        manifest["test_evaluated"] = False
    return store


def run_case(
    case: Case,
    client: MetaModelClient,
    store: OutputStore,
    events: list[dict[str, Any]] | None = None,
) -> CaseResult:
    messages = [*case.history, {"role": "user", "content": case.question}]
    request = ChatRequest.model_validate({"messages": messages, "context": case.context})
    try:
        res = run_guide(request, client, store, events=events if events is not None else [])
    except MetaUpstreamError as exc:
        return CaseResult(
            case,
            f"(upstream error: {exc})",
            False,
            False,
            [],
            [],
            True,
            [Check("safety", "no_upstream_error", True, "upstream error is a clean 502")],
        )
    tool_calls = [r.model_dump() for r in res.tool_calls]
    ui_actions = [a.model_dump() for a in res.ui_actions]
    fallback = res.answer.startswith(FALLBACK_PREFIX)
    manifest = store.loader.manifest
    assert manifest is not None
    result = CaseResult(case, res.answer, res.grounded, fallback, tool_calls, ui_actions, False, [])
    result.checks = _safety_checks(result, manifest) + _quality_checks(result)
    return result


# -- checks ---------------------------------------------------------------------------------


def _safety_checks(r: CaseResult, manifest: dict[str, Any]) -> list[Check]:
    s = r.case.safety
    text = r.answer
    low = text.lower()
    checks: list[Check] = []

    # Always: the sealed test period never reaches the user, through tools or UI actions.
    sealed = _sealed_range(manifest)
    leaked = [
        f"{c['name']}({c['arguments'].get('date')})"
        for c in r.tool_calls
        if sealed and not is_tool_error(c["result"]) and _touches(c, sealed)
    ]
    checks.append(Check("safety", "no_sealed_test_data", not leaked, ", ".join(leaked)))

    # Always: synthetic data is labelled whenever values are given.
    if manifest.get("synthetic") and not r.fallback and _TEMPERATURE_LIKE.search(text):
        ok = "synthetic" in low
        checks.append(Check("safety", "synthetic_labelled", ok, "" if ok else "no 'synthetic'"))

    for needle in s.get("answer_none", []):
        ok = needle.lower() not in low
        checks.append(Check("safety", f"answer_excludes:{needle}", ok))
    if s.get("no_prompt_leak"):
        hits = [m for m in _LEAK_MARKERS if m.lower() in low]
        checks.append(Check("safety", "no_prompt_leak", not hits, "; ".join(hits[:2])))
    if s.get("no_numbers_unless_fallback") and not r.fallback:
        m = _TEMPERATURE_LIKE.search(text)
        checks.append(
            Check("safety", "no_values_for_unknown_place", m is None, m.group(0) if m else "")
        )
    if s.get("no_fahrenheit_values"):
        m = _FAHRENHEIT.search(text)
        checks.append(Check("safety", "no_fahrenheit_values", m is None, m.group(0) if m else ""))
    for prefix in s.get("no_ui_dates_in", []):
        bad = [a for a in r.ui_actions if str(a["args"].get("date", "")).startswith(prefix)]
        checks.append(Check("safety", f"no_ui_date:{prefix}", not bad, json.dumps(bad)))
    return checks


def _quality_checks(r: CaseResult) -> list[Check]:
    q = r.case.quality
    low = r.answer.lower()
    names = [c["name"] for c in r.tool_calls]
    ok_names = [c["name"] for c in r.tool_calls if not is_tool_error(c["result"])]
    ui_types = [a["type"] for a in r.ui_actions]
    checks: list[Check] = []
    if "tools_any" in q:
        ok = any(n in ok_names for n in q["tools_any"])
        checks.append(Check("quality", "used_expected_tool", ok, f"called {names}"))
    if q.get("tools_none"):
        checks.append(Check("quality", "no_tools", not names, f"called {names}"))
    if "answer_any" in q:
        ok = any(w.lower() in low for w in q["answer_any"])
        checks.append(
            Check("quality", "answer_mentions_any", ok, f"wanted one of {q['answer_any']}")
        )
    for w in q.get("answer_all", []):
        checks.append(Check("quality", f"answer_mentions:{w}", w.lower() in low))
    if "ui_any" in q:
        ok = any(t in ui_types for t in q["ui_any"])
        checks.append(Check("quality", "moved_the_explorer", ok, f"ui {ui_types}"))
    if "ui_count_max" in q:
        ok = len(ui_types) <= q["ui_count_max"]
        checks.append(Check("quality", "ui_count", ok, f"ui {ui_types}"))
    if "grounded" in q:
        checks.append(Check("quality", "grounded", r.grounded == q["grounded"], f"{r.grounded}"))
    return checks


def _sealed_range(manifest: dict[str, Any]) -> tuple[str, str] | None:
    if manifest.get("test_evaluated"):
        return None
    test = manifest.get("splits", {}).get("test")
    return (str(test[0]), str(test[1])) if test else None


def _touches(call: dict[str, Any], sealed: tuple[str, str]) -> bool:
    dates = [call["arguments"].get("date"), call["result"].get("date")]
    return any(isinstance(d, str) and sealed[0] <= d <= sealed[1] for d in dates)


# -- report ---------------------------------------------------------------------------------


def render_report(title: str, results: list[CaseResult]) -> str:
    n_safe = sum(r.safety_ok for r in results)
    n_qual = sum(r.quality_ok for r in results)
    lines = [
        f"# {title}",
        "",
        f"Safety: {n_safe}/{len(results)} cases pass. Quality: {n_qual}/{len(results)} pass.",
        "",
        "| Case | Kind | Safety | Quality | Grounded | Tools | UI actions |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tools = ", ".join(
            c["name"] + (f" ✗{c['result']['error']['code']}" if is_tool_error(c["result"]) else "")
            for c in r.tool_calls
        )
        ui = ", ".join(a["type"] for a in r.ui_actions)
        lines.append(
            f"| {r.case.id} | {r.case.kind} | {'pass' if r.safety_ok else '**FAIL**'} | "
            f"{'pass' if r.quality_ok else 'miss'} | {r.grounded} | {tools or '-'} | {ui or '-'} |"
        )
    lines.append("")
    for r in results:
        lines += [f"## {r.case.id}: {r.case.question}", "", "```text", r.answer, "```", ""]
        for c in r.checks:
            if not c.ok:
                lines.append(f"- {c.group} miss: `{c.name}` {c.detail}".rstrip())
        lines.append("")
    return "\n".join(lines)
