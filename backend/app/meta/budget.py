"""Hard spending cap for every Meta Model API call (PRD_BACKEND §6.5).

The owner has a fixed pot of free credits and no more. Before each real call the backend
reserves the worst-case cost of that call; the call goes ahead only if the reservation fits
under both the lifetime budget and today's budget. After the call the reservation is settled
with the actual cost (from `usage` when the API returns it, otherwise the reservation stands).

Fail closed: with `META_CLIENT=http`, no Meta call is made unless the budget, the prices for
that kind of call and a durable ledger are all configured, and the ledger loaded. If the ledger
cannot be saved, calls stop until it can.

The ledger must outlive restarts, or every restart would hand out a fresh budget. Two stores:
- `META_BUDGET_LEDGER_PATH`: a local JSON file (local runs, the P7-R harness, a persistent disk).
- `META_BUDGET_HF_REPO`: one JSON file in a private Hugging Face dataset repo, for Render's free
  tier, whose disk is wiped on every deploy and restart. Token: `HF_BUDGET_TOKEN`, else `HF_TOKEN`.

Prices come only from env vars, read off the owner's Meta dashboard; none are guessed here.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from app.meta.client import ChatTurn, MetaModelClient, MetaUpstreamError, Transcript
from app.settings import settings

logger = logging.getLogger(__name__)

CallKind = Literal["chat", "transcribe", "image", "segment"]

LEDGER_FILENAME = "meta_budget_ledger.json"
# Characters per token used to over-estimate prompt size before a chat call. Real English and
# JSON run at about 3-4 characters per token, so 2.5 errs on the side of reserving too much.
CHARS_PER_TOKEN_ESTIMATE = 2.5
# Longest clip /transcribe accepts (PRD_BACKEND §6.5): 30 s.
MAX_AUDIO_MIN = 0.5
# Push the ledger to a remote store at least this often while it has unsaved spend.
SYNC_EVERY_USD = 0.25
SYNC_EVERY_S = 300.0


class BudgetExhaustedError(MetaUpstreamError):
    """No budget left for this call. A MetaUpstreamError, so every feature degrades as it does
    when the API is down (502 for the guide, threshold method for the lasso, and so on)."""


# -- ledger stores ------------------------------------------------------------------------------


class LedgerStore(Protocol):
    def load(self) -> dict[str, Any] | None: ...
    def save(self, state: dict[str, Any]) -> None: ...
    @property
    def remote(self) -> bool: ...


class FileLedgerStore:
    remote = False

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".ledger-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)


class HfLedgerStore:
    """The ledger as one file in a private Hugging Face dataset repo."""

    remote = True

    def __init__(self, repo_id: str, token: str | None) -> None:
        self.repo_id = repo_id
        self.token = token

    def load(self) -> dict[str, Any] | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(
                self.repo_id,
                LEDGER_FILENAME,
                repo_type="dataset",
                token=self.token,
                force_download=True,
            )
        except EntryNotFoundError:
            return None  # first run: the repo exists but has no ledger yet
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def save(self, state: dict[str, Any]) -> None:
        from huggingface_hub import HfApi

        body = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
        HfApi(token=self.token).upload_file(
            path_or_fileobj=io.BytesIO(body),
            path_in_repo=LEDGER_FILENAME,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"budget ledger: ${state.get('total_usd', 0):.2f} spent",
        )


def configured_store() -> LedgerStore | None:
    if settings.META_BUDGET_HF_REPO:
        return HfLedgerStore(
            settings.META_BUDGET_HF_REPO, settings.HF_BUDGET_TOKEN or settings.HF_TOKEN
        )
    if settings.META_BUDGET_LEDGER_PATH:
        return FileLedgerStore(Path(settings.META_BUDGET_LEDGER_PATH))
    return None


# -- the budget -----------------------------------------------------------------------------------


@dataclass
class Reservation:
    kind: CallKind
    usd: float


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Budget:
    """Lifetime and daily USD caps over a persisted ledger. Thread-safe."""

    def __init__(
        self,
        store: LedgerStore,
        total_usd: float,
        daily_usd: float,
        today: Callable[[], str] = _utc_day,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.total_usd = total_usd
        self.daily_usd = daily_usd
        self._today = today
        self._clock = clock
        self._lock = threading.Lock()
        self._reserved = 0.0
        self._broken = False
        self._unsynced_usd = 0.0
        self._last_sync = clock()
        loaded = store.load()
        self.state: dict[str, Any] = loaded or {"total_usd": 0.0, "days": {}, "calls": {}}
        self.state.setdefault("days", {})
        self.state.setdefault("calls", {})

    @property
    def spent_usd(self) -> float:
        return float(self.state["total_usd"])

    def spent_today_usd(self) -> float:
        return float(self.state["days"].get(self._today(), 0.0))

    def reserve(self, kind: CallKind, usd: float) -> Reservation:
        """Hold `usd` for one call, or raise BudgetExhaustedError."""
        with self._lock:
            if self._broken:
                raise BudgetExhaustedError("Budget ledger could not be saved; Meta calls paused")
            if self.spent_usd + self._reserved + usd > self.total_usd:
                logger.warning("feature=%s budget=exhausted scope=total", kind)
                raise BudgetExhaustedError("Meta credit budget used up")
            if self.spent_today_usd() + self._reserved + usd > self.daily_usd:
                logger.warning("feature=%s budget=exhausted scope=day", kind)
                raise BudgetExhaustedError("Today's Meta credit budget used up")
            self._reserved += usd
            return Reservation(kind, usd)

    def settle(self, res: Reservation, actual_usd: float | None) -> None:
        """Record what the call cost. `None` (unknown) charges the full reservation."""
        cost = res.usd if actual_usd is None else max(0.0, actual_usd)
        with self._lock:
            self._reserved = max(0.0, self._reserved - res.usd)
            day = self._today()
            self.state["total_usd"] = round(self.spent_usd + cost, 6)
            self.state["days"][day] = round(self.state["days"].get(day, 0.0) + cost, 6)
            self.state["calls"][res.kind] = int(self.state["calls"].get(res.kind, 0)) + 1
            self.state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._unsynced_usd += cost
            self._maybe_save()

    def release(self, res: Reservation) -> None:
        """The call never reached the API (it failed before sending); free the hold."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - res.usd)

    def _maybe_save(self) -> None:
        if self.store.remote:
            due = (
                self._unsynced_usd >= SYNC_EVERY_USD
                or self._clock() - self._last_sync >= SYNC_EVERY_S
            )
            if not due:
                return
        try:
            self.store.save(self.state)
        except Exception:  # noqa: BLE001 - any failure to persist must stop spending
            logger.error("budget=ledger_save_failed; Meta calls paused")
            self._broken = True
            return
        self._unsynced_usd = 0.0
        self._last_sync = self._clock()

    def flush(self) -> None:
        """Save any unsynced spend now (called at shutdown)."""
        with self._lock:
            if self._unsynced_usd > 0:
                self._unsynced_usd = SYNC_EVERY_USD  # force the save
                self._maybe_save()


# -- prices ---------------------------------------------------------------------------------------


def prices_configured(kind: CallKind) -> bool:
    s = settings
    if kind == "chat":
        return s.META_PRICE_INPUT_PER_MTOK is not None and s.META_PRICE_OUTPUT_PER_MTOK is not None
    if kind == "transcribe":
        return s.META_PRICE_PER_AUDIO_MIN is not None
    if kind == "image":
        return s.META_PRICE_PER_IMAGE is not None
    return s.META_PRICE_PER_SEGMENT is not None


def budget_configured(kind: CallKind) -> bool:
    """True when a real call of this kind may be made at all (fail closed)."""
    return (
        settings.META_BUDGET_USD > 0
        and settings.META_DAILY_BUDGET_USD > 0
        and configured_store() is not None
        and prices_configured(kind)
    )


def chat_cost_usd(input_tokens: float, output_tokens: float) -> float:
    pin = settings.META_PRICE_INPUT_PER_MTOK or 0.0
    pout = settings.META_PRICE_OUTPUT_PER_MTOK or 0.0
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def estimate_chat_usd(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int
) -> float:
    chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(tools or []))
    return chat_cost_usd(math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE), max_tokens)


# -- the budgeted client ------------------------------------------------------------------------


class BudgetedClient:
    """Wraps the real client: every call is reserved first and settled after."""

    def __init__(self, inner: MetaModelClient, budget: Budget) -> None:
        self.inner = inner
        self.budget = budget

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatTurn:
        res = self.budget.reserve("chat", estimate_chat_usd(messages, tools, max_tokens))
        try:
            turn = self.inner.chat(messages, tools, max_tokens=max_tokens, timeout_s=timeout_s)
        except BaseException:
            # The request may have been processed upstream before failing; charge the hold.
            self.budget.settle(res, None)
            raise
        actual = None
        if turn.usage is not None:
            actual = chat_cost_usd(turn.usage[0], turn.usage[1])
        self.budget.settle(res, actual)
        return turn

    def transcribe(self, audio: bytes, mime: str, lang: str, *, timeout_s: float) -> Transcript:
        price = settings.META_PRICE_PER_AUDIO_MIN or 0.0
        res = self.budget.reserve("transcribe", price * MAX_AUDIO_MIN)
        try:
            out = self.inner.transcribe(audio, mime, lang, timeout_s=timeout_s)
        except BaseException:
            self.budget.settle(res, None)
            raise
        actual = None
        if out.duration_s is not None:
            actual = price * min(MAX_AUDIO_MIN, out.duration_s / 60.0)
        self.budget.settle(res, actual)
        return out

    def image(self, prompt: str, *, timeout_s: float) -> bytes:
        res = self.budget.reserve("image", settings.META_PRICE_PER_IMAGE or 0.0)
        try:
            return self.inner.image(prompt, timeout_s=timeout_s)
        finally:
            self.budget.settle(res, None)

    def segment(self, png: bytes, point_xy: tuple[int, int], *, timeout_s: float) -> np.ndarray:
        res = self.budget.reserve("segment", settings.META_PRICE_PER_SEGMENT or 0.0)
        try:
            return self.inner.segment(png, point_xy, timeout_s=timeout_s)
        finally:
            self.budget.settle(res, None)


# -- process-wide budget ------------------------------------------------------------------------

_budget: Budget | None = None
_budget_lock = threading.Lock()


def get_budget() -> Budget:
    """The process-wide budget, loaded once. Raises BudgetExhaustedError if it cannot load."""
    global _budget
    with _budget_lock:
        if _budget is None:
            store = configured_store()
            if store is None:
                raise BudgetExhaustedError("No budget ledger configured; Meta calls disabled")
            try:
                _budget = Budget(store, settings.META_BUDGET_USD, settings.META_DAILY_BUDGET_USD)
            except Exception:  # noqa: BLE001 - an unreadable ledger must not mean "zero spent"
                logger.error("budget=ledger_load_failed; Meta calls disabled")
                raise BudgetExhaustedError("Budget ledger could not be loaded") from None
            logger.info(
                "budget=loaded spent_usd=%.2f total_usd=%.2f daily_usd=%.2f",
                _budget.spent_usd,
                _budget.total_usd,
                _budget.daily_usd,
            )
        return _budget


def flush_budget() -> None:
    """Save unsynced spend if a budget was ever loaded in this process."""
    if _budget is not None:
        _budget.flush()


def reset_budget_for_tests() -> None:
    global _budget
    with _budget_lock:
        _budget = None
