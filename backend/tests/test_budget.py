"""Credit budget for real Meta calls (app/meta/budget.py). No test calls the Meta API: the HTTP
client runs against httpx.MockTransport, and stores are local files or fakes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api.v1.chat as chat_module
from app.errors import APIError
from app.main import app
from app.meta import budget as budget_module
from app.meta.budget import (
    Budget,
    BudgetedClient,
    BudgetExhaustedError,
    FileLedgerStore,
    budget_configured,
    estimate_chat_usd,
)
from app.meta.client import ChatTurn, HttpMetaModelClient, Transcript, build_client
from app.meta.config import chat_available, postcard_available, voice_available
from app.meta.limits import FeatureLimiter
from app.settings import settings

KEY = "sk-test-not-a-real-key"


class MemoryStore:
    def __init__(self, remote: bool = True, fail: bool = False) -> None:
        self.remote = remote
        self.fail = fail
        self.saved: list[dict[str, Any]] = []
        self.initial: dict[str, Any] | None = None

    def load(self) -> dict[str, Any] | None:
        return self.initial

    def save(self, state: dict[str, Any]) -> None:
        if self.fail:
            raise OSError("disk full")
        self.saved.append(json.loads(json.dumps(state)))


class Day:
    def __init__(self, value: str = "2026-11-04") -> None:
        self.value = value

    def __call__(self) -> str:
        return self.value


def http_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "META_CLIENT", "http")
    monkeypatch.setattr(settings, "META_API_BASE_URL", "https://meta.invalid")
    monkeypatch.setattr(settings, "META_API_KEY", KEY)
    monkeypatch.setattr(settings, "META_CHAT_MODEL", "spark")
    monkeypatch.setattr(settings, "META_TRANSCRIBE_MODEL", "voice")
    monkeypatch.setattr(settings, "META_IMAGE_MODEL", "image")
    monkeypatch.setattr(settings, "CHAT_ENABLED", True)
    monkeypatch.setattr(settings, "VOICE_ENABLED", True)
    monkeypatch.setattr(settings, "POSTCARD_ENABLED", True)


# -- fail closed ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unset",
    [
        "META_BUDGET_USD",
        "META_DAILY_BUDGET_USD",
        "META_BUDGET_LEDGER_PATH",
        "META_PRICE_INPUT_PER_MTOK",
    ],
)
def test_real_chat_stays_off_until_the_budget_is_configured(
    monkeypatch: pytest.MonkeyPatch, unset: str
) -> None:
    http_settings(monkeypatch)
    assert chat_available() is True
    monkeypatch.setattr(settings, unset, 0.0 if unset.endswith("_USD") else None)
    assert chat_available() is False
    assert TestClient(app).get("/api/v1/meta").json()["features"]["chat"] is False


def test_each_feature_needs_its_own_price(monkeypatch: pytest.MonkeyPatch) -> None:
    http_settings(monkeypatch)
    monkeypatch.setattr(settings, "META_PRICE_PER_IMAGE", None)
    assert postcard_available() is False
    assert voice_available() is True
    assert budget_configured("image") is False


def test_mock_mode_needs_no_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "META_CLIENT", "mock")
    monkeypatch.setattr(settings, "CHAT_ENABLED", True)
    monkeypatch.setattr(settings, "META_BUDGET_USD", 0.0)
    assert chat_available() is True


def test_real_client_is_always_wrapped_in_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    http_settings(monkeypatch)
    assert isinstance(build_client(), BudgetedClient)


def test_unreadable_ledger_disables_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "ledger.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(settings, "META_BUDGET_LEDGER_PATH", str(bad))
    with pytest.raises(BudgetExhaustedError):
        budget_module.get_budget()


# -- caps -----------------------------------------------------------------------------------


def test_total_cap_counts_spend_and_open_reservations() -> None:
    b = Budget(MemoryStore(), total_usd=1.0, daily_usd=10.0)
    r1 = b.reserve("chat", 0.6)
    with pytest.raises(BudgetExhaustedError):
        b.reserve("chat", 0.5)  # 0.6 held + 0.5 > 1.0
    b.settle(r1, 0.2)
    b.reserve("chat", 0.5)  # 0.2 spent + 0.5 fits
    assert b.spent_usd == pytest.approx(0.2)


def test_daily_cap_resets_next_utc_day() -> None:
    day = Day()
    b = Budget(MemoryStore(), total_usd=10.0, daily_usd=1.0, today=day)
    b.settle(b.reserve("image", 0.9), None)
    with pytest.raises(BudgetExhaustedError):
        b.reserve("image", 0.2)
    day.value = "2026-11-05"
    b.reserve("image", 0.2)
    assert b.spent_usd == pytest.approx(0.9)


def test_unknown_cost_charges_the_full_reservation() -> None:
    b = Budget(MemoryStore(), total_usd=10.0, daily_usd=10.0)
    b.settle(b.reserve("chat", 0.03), None)
    assert b.spent_usd == pytest.approx(0.03)


def test_ledger_survives_a_restart(tmp_path: Path) -> None:
    store = FileLedgerStore(tmp_path / "ledger.json")
    b1 = Budget(store, total_usd=1.0, daily_usd=1.0)
    b1.settle(b1.reserve("chat", 0.7), None)
    b2 = Budget(store, total_usd=1.0, daily_usd=1.0)  # a new process
    assert b2.spent_usd == pytest.approx(0.7)
    with pytest.raises(BudgetExhaustedError):
        b2.reserve("chat", 0.5)


def test_save_failure_pauses_spending() -> None:
    b = Budget(MemoryStore(remote=False, fail=True), total_usd=10.0, daily_usd=10.0)
    b.settle(b.reserve("chat", 0.01), None)
    with pytest.raises(BudgetExhaustedError):
        b.reserve("chat", 0.01)


def test_remote_store_syncs_every_quarter_dollar_and_on_flush() -> None:
    store = MemoryStore(remote=True)
    b = Budget(store, total_usd=10.0, daily_usd=10.0)
    b.settle(b.reserve("chat", 0.1), None)
    assert store.saved == []
    b.settle(b.reserve("chat", 0.2), None)
    assert store.saved[-1]["total_usd"] == pytest.approx(0.3)
    b.settle(b.reserve("chat", 0.05), None)
    b.flush()
    assert store.saved[-1]["total_usd"] == pytest.approx(0.35)
    assert store.saved[-1]["calls"]["chat"] == 3


# -- the wrapper ------------------------------------------------------------------------------


def test_chat_estimate_over_counts_prompt_and_counts_all_output_tokens() -> None:
    msgs = [{"role": "user", "content": "x" * 1000}]
    # 1,000+ chars -> at least 400 input tokens at $1/M; 1,024 output tokens at $4/M.
    assert estimate_chat_usd(msgs, None, 1024) >= (400 * 1.0 + 1024 * 4.0) / 1_000_000


def test_chat_settles_with_reported_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )

    inner = HttpMetaModelClient(
        "https://meta.invalid", KEY, "spark", transport=httpx.MockTransport(handler)
    )
    b = Budget(MemoryStore(), total_usd=10.0, daily_usd=10.0)
    turn = BudgetedClient(inner, b).chat(
        [{"role": "user", "content": "hi"}], None, max_tokens=1024, timeout_s=5
    )
    assert turn.usage == (1000, 100)
    assert b.spent_usd == pytest.approx((1000 * 1.0 + 100 * 4.0) / 1_000_000)


def test_no_request_leaves_once_the_budget_is_spent() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    inner = HttpMetaModelClient(
        "https://meta.invalid", KEY, "spark", transport=httpx.MockTransport(handler)
    )
    b = Budget(MemoryStore(), total_usd=0.000001, daily_usd=10.0)
    with pytest.raises(BudgetExhaustedError):
        BudgetedClient(inner, b).chat(
            [{"role": "user", "content": "hi"}], None, max_tokens=1024, timeout_s=5
        )
    assert sent == []


def test_failed_call_is_still_charged() -> None:
    class Boom:
        def chat(self, *a: Any, **k: Any) -> ChatTurn:
            raise httpx.ReadTimeout("slow")

    b = Budget(MemoryStore(), total_usd=10.0, daily_usd=10.0)
    with pytest.raises(httpx.ReadTimeout):
        BudgetedClient(Boom(), b).chat([], None, max_tokens=100, timeout_s=1)  # type: ignore[arg-type]
    assert b.spent_usd > 0


def test_transcribe_charges_by_reported_duration() -> None:
    class Voice:
        def transcribe(self, audio: bytes, mime: str, lang: str, *, timeout_s: float) -> Transcript:
            return Transcript(text="hello", duration_s=12.0)

    b = Budget(MemoryStore(), total_usd=10.0, daily_usd=10.0)
    BudgetedClient(Voice(), b).transcribe(b"x", "audio/webm", "en", timeout_s=5)  # type: ignore[arg-type]
    assert b.spent_usd == pytest.approx(0.01 * 12 / 60)


def test_guide_endpoint_degrades_to_502_when_budget_is_spent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http_settings(monkeypatch)
    monkeypatch.setattr(chat_module, "_limiter", None)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"total_usd": 100.0, "days": {}, "calls": {}}), encoding="utf-8")
    monkeypatch.setattr(settings, "META_BUDGET_LEDGER_PATH", str(ledger))
    res = TestClient(app).post(
        "/api/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "upstream_error"


# -- per-IP daily cap -------------------------------------------------------------------------


def test_one_ip_cannot_use_the_whole_daily_cap() -> None:
    lim = FeatureLimiter("100/minute", daily_cap=10, ip_daily_cap=2)
    lim.check("1.1.1.1")
    lim.check("1.1.1.1")
    with pytest.raises(APIError) as exc:
        lim.check("1.1.1.1")
    assert exc.value.code == "daily_cap_reached"
    lim.check("2.2.2.2")  # another visitor still gets in
