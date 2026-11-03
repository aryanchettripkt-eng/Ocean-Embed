"""Tests for the "Talk to the ocean" guide (PRD_BACKEND Tests 7, 14, 15, 16).

No test calls the Meta API: the guide runs against a scripted client or the offline mock,
and the HTTP client runs against an in-process httpx.MockTransport.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api.v1.chat as chat_module
import app.data.store as store_module
from app.main import app
from app.meta import client as client_module
from app.meta.client import (
    ChatTurn,
    HttpMetaModelClient,
    MetaUpstreamError,
    MockMetaModelClient,
    ToolCall,
)
from app.meta.guard import FALLBACK_PREFIX, extract_checked_numbers, ungrounded_numbers
from app.meta.guide import MAX_ROUNDS, run_guide
from app.meta.tools import ToolRunner
from app.schemas.caveats import AI_CAVEAT
from app.schemas.models import ChatRequest
from app.settings import settings

SENTINEL_KEY = "sk-SENTINEL-meta-key-0123456789"
DATE = "2020-01-05"


class ScriptedClient:
    """Returns pre-written turns in order and records what it was sent."""

    def __init__(self, turns: list[ChatTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatTurn:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.turns.pop(0)


def call(name: str, i: int = 0, **args: Any) -> ToolCall:
    return ToolCall(id=f"c{i}", name=name, arguments=json.dumps(args))


def turn_calls(*calls: ToolCall) -> ChatTurn:
    return ChatTurn(content=None, tool_calls=list(calls))


def ask(
    text: str = "What is happening here?", context: dict[str, Any] | None = None
) -> ChatRequest:
    return ChatRequest.model_validate(
        {"messages": [{"role": "user", "content": text}], "context": context}
    )


@pytest.fixture(autouse=True)
def guide_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_ENABLED", True)
    monkeypatch.setattr(settings, "META_CLIENT", "mock")
    monkeypatch.setattr(settings, "META_API_BASE_URL", None)
    monkeypatch.setattr(settings, "META_API_KEY", None)
    monkeypatch.setattr(settings, "META_CHAT_MODEL", None)
    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT", "100/minute")
    monkeypatch.setattr(settings, "CHAT_DAILY_CAP", 1000)
    monkeypatch.setattr(chat_module, "_limiter", None)


def use_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    monkeypatch.setattr(client_module, "build_client", lambda: client)


def store() -> store_module.OutputStore:
    return store_module.store


def profile_value(lat: float, lon: float, depth_index: int) -> float:
    value = store().get_profile(lat, lon, DATE).pred[depth_index]
    assert value is not None
    return value


# ---------------------------------------------------------------------------
# Test 7: tool dispatch, grounding fallback, round limit, endpoint errors
# ---------------------------------------------------------------------------


def test_tool_dispatch_and_grounded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    value = profile_value(12.5, 65.5, 7)
    scripted = ScriptedClient(
        [
            turn_calls(call("get_temperature", lat=12.5, lon=65.5, date=DATE, depth_m=100)),
            ChatTurn(content=f"The model prediction at 100.0 m is {value} °C (synthetic data)."),
        ]
    )
    use_client(monkeypatch, scripted)
    res = TestClient(app).post(
        "/api/v1/chat", json={"messages": [{"role": "user", "content": "How warm at 100 m?"}]}
    )
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    body = res.json()
    assert body["grounded"] is True
    assert body["answer"].startswith("The model prediction")
    assert body["tool_calls"][0]["name"] == "get_temperature"
    assert body["tool_calls"][0]["result"]["pred_c"] == value
    assert body["synthetic"] is True and body["run_id"] == "synthetic_v1"
    # The tool result went back to the model as a tool message.
    tool_msgs = [m for m in scripted.calls[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["pred_c"] == value


def test_fabricated_number_triggers_fallback() -> None:
    value = profile_value(12.5, 65.5, 7)
    fake = round(value + 3.17, 2)
    scripted = ScriptedClient(
        [
            turn_calls(call("get_temperature", lat=12.5, lon=65.5, date=DATE, depth_m=100)),
            ChatTurn(content=f"It is {fake} °C at 100.0 m."),
        ]
    )
    result = run_guide(ask(), scripted, store())
    assert result.grounded is False
    assert result.answer.startswith(FALLBACK_PREFIX)
    assert "get_temperature" in result.answer
    assert str(fake) not in result.answer


def test_round_limit_enforced() -> None:
    looping = [turn_calls(call("get_dataset_info", i)) for i in range(MAX_ROUNDS)]
    scripted = ScriptedClient([*looping, ChatTurn(content="The data covers a date range.")])
    result = run_guide(ask(), scripted, store())
    assert result.rounds == MAX_ROUNDS
    assert len(scripted.calls) == MAX_ROUNDS + 1
    assert scripted.calls[-1]["tools"] is None  # the last call cannot request tools
    assert all(c["tools"] for c in scripted.calls[:-1])


def test_bad_tool_calls_return_errors_to_model() -> None:
    scripted = ScriptedClient(
        [
            ChatTurn(
                content=None,
                tool_calls=[
                    ToolCall(id="a", name="run_python", arguments="{}"),
                    ToolCall(id="b", name="get_temperature", arguments="not json"),
                    call("get_temperature_profile", 2, lat=12.5, lon=65.5, date="2030-01-01"),
                ],
            ),
            ChatTurn(content="The data doesn't cover that."),
        ]
    )
    result = run_guide(ask(), scripted, store())
    codes = [r.result["error"]["code"] for r in result.tool_calls]
    assert codes == ["unknown_tool", "invalid_arguments", "date_unavailable"]
    assert result.grounded is True


def test_chat_disabled_and_misconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(settings, "CHAT_ENABLED", False)
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 503 and res.json()["error"]["code"] == "chat_disabled"
    assert client.get("/api/v1/meta").json()["features"]["chat"] is False

    # Enabled, but the real client has no key/model configured: still disabled.
    monkeypatch.setattr(settings, "CHAT_ENABLED", True)
    monkeypatch.setattr(settings, "META_CLIENT", "http")
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 503 and res.json()["error"]["code"] == "chat_disabled"


def test_meta_features_and_ai_caveat() -> None:
    meta = TestClient(app).get("/api/v1/meta").json()
    assert meta["features"]["chat"] is True
    assert meta["features"]["provider_label"] == meta["chat"]["provider_label"]
    assert "mock" in meta["features"]["provider_label"].lower()
    for flag in ("voice", "postcards", "segment", "beach"):
        assert meta["features"][flag] is False
    assert meta["caveats"]["ai"] == AI_CAVEAT


def test_upstream_failure_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing:
        def chat(self, *a: Any, **k: Any) -> ChatTurn:
            raise MetaUpstreamError("Meta API returned HTTP 500")

    use_client(monkeypatch, Failing())
    res = TestClient(app).post(
        "/api/v1/chat", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "upstream_error"


def test_rate_limit_and_daily_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    payload = {"messages": [{"role": "user", "content": "What does the data cover?"}]}
    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT", "2/minute")
    codes = [client.post("/api/v1/chat", json=payload).status_code for _ in range(3)]
    assert codes == [200, 200, 429]

    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT", "100/minute")
    monkeypatch.setattr(settings, "CHAT_DAILY_CAP", 1)
    monkeypatch.setattr(chat_module, "_limiter", None)
    assert client.post("/api/v1/chat", json=payload).status_code == 200
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 429 and res.json()["error"]["code"] == "daily_cap_reached"


def test_request_limits_validated() -> None:
    client = TestClient(app)
    too_long = {"messages": [{"role": "user", "content": "x" * 2001}]}
    too_many = {"messages": [{"role": "user", "content": "x"}] * 11}
    empty: dict[str, Any] = {"messages": []}
    for payload in (too_long, too_many, empty):
        assert client.post("/api/v1/chat", json=payload).status_code == 422


def test_mock_client_answers_are_grounded() -> None:
    mock = MockMetaModelClient()
    ctx = {"date": DATE, "lat": 12.5, "lon": 65.5, "depth_m": 100.0}
    for text, context in [
        ("What is the temperature here?", ctx),
        ("How accurate is the model?", {"depth_m": 100.0}),
        ("What does the data cover?", None),
        ("Show me a cyclone", None),
    ]:
        result = run_guide(ask(text, context), mock, store())
        assert result.grounded, result.answer
        assert "mock" in result.answer.lower()
    here = run_guide(ask("What is the temperature here?", ctx), mock, store())
    assert [a.type for a in here.ui_actions] == ["select_point"]


# ---------------------------------------------------------------------------
# Test 14: Meta client mocked, key never leaks
# ---------------------------------------------------------------------------


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": message}]}


def test_http_client_parses_tool_calls_and_sends_key_only_in_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=_completion(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "get_dataset_info", "arguments": "{}"},
                        }
                    ],
                }
            ),
        )

    client = HttpMetaModelClient(
        "https://meta.invalid/v1/",
        SENTINEL_KEY,
        "chat-model",
        transport=httpx.MockTransport(handler),
    )
    turn = client.chat([{"role": "user", "content": "hi"}], [], max_tokens=10, timeout_s=5)
    assert turn.tool_calls[0].name == "get_dataset_info"
    req = seen[0]
    assert str(req.url) == "https://meta.invalid/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {SENTINEL_KEY}"
    body = json.loads(req.content)
    assert body["model"] == "chat-model" and "tools" not in body  # empty tools are not sent
    assert SENTINEL_KEY not in req.content.decode()


def test_http_client_fallback_model_on_429() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        models.append(model)
        if model == "spark":
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_completion({"content": "ok"}))

    client = HttpMetaModelClient(
        "https://meta.invalid", SENTINEL_KEY, "spark", "glimmer", httpx.MockTransport(handler)
    )
    assert client.chat([], None, max_tokens=10, timeout_s=5).content == "ok"
    assert models == ["spark", "glimmer"]


def test_key_and_upstream_body_never_leak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"internal: echo {request.headers['authorization']}")

    monkeypatch.setattr(settings, "META_CLIENT", "http")
    monkeypatch.setattr(settings, "META_API_BASE_URL", "https://meta.invalid")
    monkeypatch.setattr(settings, "META_API_KEY", SENTINEL_KEY)
    monkeypatch.setattr(settings, "META_CHAT_MODEL", "spark")
    real_client = HttpMetaModelClient(
        "https://meta.invalid", SENTINEL_KEY, "spark", transport=httpx.MockTransport(handler)
    )
    use_client(monkeypatch, real_client)

    with caplog.at_level(logging.DEBUG):
        client = TestClient(app)
        res = client.post("/api/v1/chat", json={"messages": [{"role": "user", "content": "hello"}]})
        meta = client.get("/api/v1/meta")
    assert res.status_code == 502
    assert meta.json()["features"]["chat"] is True
    for text in (res.text, meta.text, caplog.text):
        assert SENTINEL_KEY not in text
        assert "internal: echo" not in text

    with pytest.raises(MetaUpstreamError) as exc_info:
        real_client.chat([], None, max_tokens=10, timeout_s=5)
    assert SENTINEL_KEY not in str(exc_info.value)
    assert exc_info.value.__suppress_context__ is True  # no chained upstream detail


def test_no_user_text_in_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret_question = "my-private-question-XYZ about 17.5 m"
    with caplog.at_level(logging.DEBUG):
        TestClient(app).post(
            "/api/v1/chat", json={"messages": [{"role": "user", "content": secret_question}]}
        )
    assert "my-private-question-XYZ" not in caplog.text


# ---------------------------------------------------------------------------
# Test 15: UI tool validation
# ---------------------------------------------------------------------------

EVENTS = [
    {
        "id": "visible-event",
        "title": "Visible event",
        "start": "2020-01-03",
        "end": "2020-01-06",
        "before": "2020-01-01",
        "after": "2020-01-09",
    },
    {
        "id": "sealed-event",
        "title": "Sealed event",
        "start": "2021-05-01",
        "end": "2021-05-03",
        "before": "2021-04-28",
        "after": "2021-05-08",
    },
]


def test_valid_ui_actions_pass_through_normalised() -> None:
    scripted = ScriptedClient(
        [
            turn_calls(
                call(
                    "set_view",
                    0,
                    date=DATE,
                    depth_m=98,
                    mode="3d",
                    center={"lat": 15.2, "lon": 87.1},
                    zoom=5,
                ),
                call("slice_at", 1, lon=88.4),
                call("open_event", 2, event_id="visible-event", phase="after"),
            ),
            ChatTurn(content="Here is the view."),
        ]
    )
    result = run_guide(ask(), scripted, store(), events=EVENTS)
    actions = [a.model_dump() for a in result.ui_actions]
    assert actions[0]["type"] == "set_view"
    assert actions[0]["args"]["depth_m"] == 100.0  # snapped to the model depth
    assert actions[0]["args"]["mode"] == "3d"
    assert set(actions[0]["args"]["center"]) == {"lat", "lon"}
    assert actions[1] == {"type": "slice_at", "args": {"lon": pytest.approx(88.5)}}
    assert actions[2] == {
        "type": "open_event",
        "args": {"event_id": "visible-event", "phase": "after"},
    }


@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("set_view", {"date": "2019-06-01"}, "date_unavailable"),
        ("set_view", {"depth_m": 400}, "bad_depth"),
        ("select_point", {"lat": 45.0, "lon": 120.0}, "out_of_region"),
        ("set_view", {"date": "2021-03-01"}, "sealed_test_period"),
        ("open_event", {"event_id": "no-such-event"}, "unknown_event"),
        ("open_event", {"event_id": "sealed-event"}, "unknown_event"),
        ("set_view", {"zoom": 12}, "invalid_arguments"),
        ("slice_at", {"lat": 10, "lon": 80}, "invalid_section"),
        ("set_view", {"teleport": True}, "invalid_arguments"),
    ],
)
def test_invalid_ui_args_return_tool_errors(name: str, args: dict[str, Any], code: str) -> None:
    scripted = ScriptedClient([turn_calls(call(name, **args)), ChatTurn(content="Sorry.")])
    result = run_guide(ask(), scripted, store(), events=EVENTS)
    assert result.ui_actions == []
    assert result.tool_calls[0].result["error"]["code"] == code
    tool_msg = [m for m in scripted.calls[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["error"]["code"] == code


def test_more_than_three_ui_actions_truncated() -> None:
    calls = [call("select_point", i, lat=10 + i, lon=70) for i in range(5)]
    scripted = ScriptedClient([turn_calls(*calls), ChatTurn(content="Done.")])
    result = run_guide(ask(), scripted, store())
    assert len(result.ui_actions) == 3
    assert [a.args["lat"] for a in result.ui_actions] == [10.5, 11.5, 12.5]
    assert result.tool_calls[3].result["error"]["code"] == "too_many_ui_actions"


def test_list_events_hides_sealed_test_events() -> None:
    runner = ToolRunner(store(), EVENTS)
    out = runner.run("list_events", "{}")
    assert [e["id"] for e in out.result["events"]] == ["visible-event"]


# ---------------------------------------------------------------------------
# Test 16: guard with UI actions
# ---------------------------------------------------------------------------


def test_guard_keeps_ui_actions_and_ignores_ui_args_as_sources() -> None:
    scripted = ScriptedClient(
        [
            turn_calls(call("set_view", 0, depth_m=300, center={"lat": 22.5, "lon": 88.5})),
            ChatTurn(content="At 22.5°N and 300 m the water is 14.25 °C."),
        ]
    )
    result = run_guide(ask(), scripted, store())
    assert result.grounded is False
    assert result.answer.startswith(FALLBACK_PREFIX)
    assert [a.type for a in result.ui_actions] == ["set_view"]


def test_guard_accepts_context_numbers() -> None:
    scripted = ScriptedClient([ChatTurn(content="You are looking at 100.0 m.")])
    result = run_guide(ask(context={"depth_m": 100.0}), scripted, store())
    assert result.grounded is True


def test_guard_number_extraction() -> None:
    text = "On 2019-05-01 in 2019, 3 floats saw 28.1 °C at 50 m, 1,000 km away, 12% and 15°N."
    assert extract_checked_numbers(text) == [(28.1, 1), (50.0, 0), (1000.0, 0), (12.0, 0)]
    assert ungrounded_numbers("It is 28.10 °C.", [28.1]) == []
    assert ungrounded_numbers("It is 28.1 °C.", [28.14]) == []
    assert ungrounded_numbers("It is 28.2 °C.", [28.14]) == [28.2]
    assert ungrounded_numbers("Cooler by 0.3 °C.", [-0.3]) == []
