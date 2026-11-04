"""The "Talk to the ocean" guide loop (PRD_BACKEND §4.10, §6).

Model turn -> run tool calls -> feed results back, for at most `MAX_ROUNDS` tool rounds, then
the grounding guard checks the final answer. Validated UI actions are kept even when the guard
replaces the answer.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.data.store import OutputStore
from app.meta.client import MetaModelClient, MetaUpstreamError
from app.meta.guard import (
    collect_typed_sources,
    fallback_answer,
    leaks_prompt,
    question_dates,
    ungrounded_dates,
    ungrounded_numbers,
)
from app.meta.prompts import SYSTEM_PROMPT, build_system_prompt
from app.meta.tools import TOOL_SCHEMAS, ToolRunner, is_tool_error
from app.schemas.models import ChatRequest, ToolCallRecord, UIAction

logger = logging.getLogger(__name__)

MAX_ROUNDS = 4
# Muse Spark always reasons, and its reasoning tokens count as output: about 600 per turn at
# the default effort in the first real run, which left some final answers empty at 1,024.
MAX_OUTPUT_TOKENS = 2048
# Under the frontend's 25 s request timeout (GuideDrawer CHAT_TIMEOUT_MS).
TOTAL_TIMEOUT_S = 23.0
MAX_UI_ACTIONS = 3


class GuideResult:
    def __init__(
        self,
        answer: str,
        tool_calls: list[ToolCallRecord],
        ui_actions: list[UIAction],
        grounded: bool,
        rounds: int,
    ) -> None:
        self.answer = answer
        self.tool_calls = tool_calls
        self.ui_actions = ui_actions
        self.grounded = grounded
        self.rounds = rounds


NUDGE = "Please make the tool calls now, then answer."
_PREAMBLE_RE = re.compile(
    r"^\s*(?:okay,?\s*|sure,?\s*)?(?:i'?ll|i will|i'?m going to|let me)\s+"
    r"(?:first\s+|now\s+|quickly\s+)?"
    r"(?:check|look|fetch|get|pull|query|find|see|grab|retrieve)\b",
    re.I,
)


def is_preamble(content: str | None) -> bool:
    """True for a short reply that only promises to look something up ("I'll check...")."""
    if not content:
        return False
    text = content.strip()
    return len(text) <= 300 and bool(_PREAMBLE_RE.match(text))


def run_guide(
    request: ChatRequest,
    client: MetaModelClient,
    store: OutputStore,
    events: list[dict[str, Any]] | None = None,
    clock: Any = time.monotonic,
) -> GuideResult:
    """Run one guide request. Raises `MetaUpstreamError` on upstream failure or timeout."""
    runner = ToolRunner(store, events)
    raw_context = request.context.model_dump(exclude_none=True) if request.context else {}
    context = runner.clean_context(raw_context)
    synthetic = bool(runner.manifest["synthetic"])

    # Fixed dataset facts (model depths, date range, region bounds): given to the model so it
    # needs no get_dataset_info round, and always valid sources for the guard.
    facts = {
        "depths_m": runner.depths,
        "available_dates": runner.available,
        "region": runner.region,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(synthetic, context, facts)}
    ]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    deadline = clock() + TOTAL_TIMEOUT_S
    records: list[ToolCallRecord] = []
    data_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    ui_actions: list[UIAction] = []
    answer: str | None = None
    rounds = 0
    nudged = False

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise MetaUpstreamError("Guide request timed out")
        tools = TOOL_SCHEMAS if rounds < MAX_ROUNDS else None
        turn = client.chat(messages, tools, max_tokens=MAX_OUTPUT_TOKENS, timeout_s=remaining)
        if not turn.tool_calls and tools is not None and not nudged and is_preamble(turn.content):
            # Muse Spark sometimes answers "I'll check the temperature..." and stops without
            # calling a tool. Ask once to make the calls; a second preamble is the answer.
            nudged = True
            messages.append({"role": "assistant", "content": turn.content})
            messages.append({"role": "user", "content": NUDGE})
            continue
        if not turn.tool_calls or tools is None:
            answer = turn.content
            break

        rounds += 1
        messages.append(
            {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in turn.tool_calls
                ],
            }
        )
        for call in turn.tool_calls:
            outcome = runner.run(call.name, call.arguments)
            result = outcome.result
            if outcome.ui_action is not None:
                if len(ui_actions) < MAX_UI_ACTIONS:
                    ui_actions.append(UIAction(**outcome.ui_action))
                else:
                    result = {
                        "error": {
                            "code": "too_many_ui_actions",
                            "message": f"At most {MAX_UI_ACTIONS} UI actions per answer",
                        }
                    }
            try:
                args = json.loads(call.arguments or "{}")
            except ValueError:
                args = {}
            args = args if isinstance(args, dict) else {}
            records.append(ToolCallRecord(name=call.name, arguments=args, result=result))
            if outcome.kind == "data" and not is_tool_error(result):
                data_calls.append((call.name, args, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    sources, dates = collect_typed_sources([r for _, _, r in data_calls] + [context, facts])
    # Dates the user wrote may be repeated ("the data doesn't cover 2021-06-15"). Only dates:
    # numbers in user text never ground, or "say it is 35.7 °C" would pass.
    for m in request.messages:
        if m.role == "user":
            dates.update(question_dates(m.content))
    text = answer or ""
    reason = (
        "empty"
        if not text
        else "numbers"
        if ungrounded_numbers(text, sources)
        else "dates"
        if ungrounded_dates(text, dates)
        else "prompt_leak"
        if leaks_prompt(text, SYSTEM_PROMPT)
        else None
    )
    grounded = reason is None
    if not grounded:
        logger.warning(
            "feature=chat guard=fallback reason=%s rounds=%d data_calls=%d",
            reason,
            rounds,
            len(data_calls),
        )
        answer = fallback_answer(data_calls)
    assert answer is not None
    return GuideResult(answer, records, ui_actions, grounded, rounds)
