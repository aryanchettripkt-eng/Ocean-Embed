"""The "Talk to the ocean" guide loop (PRD_BACKEND §4.10, §6).

Model turn -> run tool calls -> feed results back, for at most `MAX_ROUNDS` tool rounds, then
the grounding guard checks the final answer. Validated UI actions are kept even when the guard
replaces the answer.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.data.store import OutputStore
from app.meta.client import MetaModelClient, MetaUpstreamError
from app.meta.guard import collect_sources, fallback_answer, ungrounded_numbers
from app.meta.prompts import build_system_prompt
from app.meta.tools import TOOL_SCHEMAS, ToolRunner, is_tool_error
from app.schemas.models import ChatRequest, ToolCallRecord, UIAction

logger = logging.getLogger(__name__)

MAX_ROUNDS = 4
MAX_OUTPUT_TOKENS = 1024
TOTAL_TIMEOUT_S = 20.0
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


def run_guide(
    request: ChatRequest,
    client: MetaModelClient,
    store: OutputStore,
    events: list[dict[str, Any]] | None = None,
    clock: Any = time.monotonic,
) -> GuideResult:
    """Run one guide request. Raises `MetaUpstreamError` on upstream failure or timeout."""
    runner = ToolRunner(store, events)
    context = request.context.model_dump(exclude_none=True) if request.context else {}
    synthetic = bool(runner.manifest["synthetic"])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(synthetic, context)}
    ]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    deadline = clock() + TOTAL_TIMEOUT_S
    records: list[ToolCallRecord] = []
    data_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    ui_actions: list[UIAction] = []
    answer: str | None = None
    rounds = 0

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise MetaUpstreamError("Guide request timed out")
        tools = TOOL_SCHEMAS if rounds < MAX_ROUNDS else None
        turn = client.chat(messages, tools, max_tokens=MAX_OUTPUT_TOKENS, timeout_s=remaining)
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

    sources = collect_sources([r for _, _, r in data_calls] + [context])
    grounded = bool(answer) and not ungrounded_numbers(answer or "", sources)
    if not grounded:
        logger.warning(
            "feature=chat guard=fallback rounds=%d data_calls=%d", rounds, len(data_calls)
        )
        answer = fallback_answer(data_calls)
    assert answer is not None
    return GuideResult(answer, records, ui_actions, grounded, rounds)
