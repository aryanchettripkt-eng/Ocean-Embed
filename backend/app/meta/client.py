"""MetaModelClient: chat with tool calling over the Meta Model API (PRD_BACKEND §6.1).

Two implementations share one interface, so the real key drops in without code changes:
- `HttpMetaModelClient` calls `META_API_BASE_URL` with `META_API_KEY` (env vars only).
- `MockMetaModelClient` is an offline, deterministic stand-in (`META_CLIENT=mock`) for local
  development, the frontend drawer and demos without a key. It never calls the network.

The HTTP client assumes an OpenAI-compatible `/chat/completions` surface with `tools`. This is
[CONFIRM] against the Meta Model API docs (OPEN_QUESTIONS 2026-10-24, PHASES P7-0). Model ids
come only from env vars; none are guessed here. Transcription and image generation follow the
Meta docs (2026-11-04); `segment()` is still a guess, because SAM 3.1 runs through the Responses
API with its own streamed mask format (DECISIONS 2026-11-04).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np

from app.settings import settings

# Status codes after which the chat fallback model is tried (§6.1).
_FALLBACK_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class ToolCall:
    """One tool call requested by the model. `arguments` is the raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass
class ChatTurn:
    """One model reply: either text, tool calls, or both."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # (prompt_tokens, completion_tokens) when the API reports them; used by the budget.
    usage: tuple[int, int] | None = None


@dataclass
class Transcript:
    """Text heard in a clip. `duration_s` is None when the provider does not report it."""

    text: str
    duration_s: float | None = None


# Upper bound on a decoded postcard image; anything larger is treated as an upstream error.
MAX_IMAGE_BYTES = 5_000_000


def is_webp(data: bytes) -> bool:
    """RIFF....WEBP container check; the only image format the postcard cache serves."""
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


class MetaUpstreamError(Exception):
    """Upstream failure or timeout. Messages never carry upstream bodies, headers or keys."""


class MetaModelClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatTurn: ...

    def transcribe(self, audio: bytes, mime: str, lang: str, *, timeout_s: float) -> Transcript: ...

    def image(self, prompt: str, *, timeout_s: float) -> bytes: ...

    def segment(self, png: bytes, point_xy: tuple[int, int], *, timeout_s: float) -> np.ndarray: ...


class HttpMetaModelClient:
    """Meta Model API client over httpx (OpenAI-compatible chat completions, [CONFIRM])."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        transport: httpx.BaseTransport | None = None,
        *,
        transcribe_model: str | None = None,
        image_model: str | None = None,
        segment_model: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._url = self._base + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._fallback_model = fallback_model
        self._transcribe_model = transcribe_model
        self._image_model = image_model
        self._segment_model = segment_model
        self._transport = transport

    def _post_json(self, path: str, payload: dict[str, Any], timeout_s: float) -> Any:
        """POST JSON and return the parsed body. Errors never carry the upstream body or key."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            with httpx.Client(timeout=timeout_s, transport=self._transport) as client:
                resp = client.post(self._base + path, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise MetaUpstreamError("Meta API request timed out") from None
        except httpx.HTTPError as exc:
            raise MetaUpstreamError(f"Meta API request failed ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise MetaUpstreamError(f"Meta API returned HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError:
            raise MetaUpstreamError("Meta API returned an unreadable response") from None

    def image(self, prompt: str, *, timeout_s: float) -> bytes:
        """Generate one postcard image (WebP) from a template-built prompt."""
        if not self._image_model:
            raise MetaUpstreamError("Image model is not configured")
        # Route and fields per dev.meta.ai/docs/api-reference/images (CreateImageRequest).
        payload = {
            "model": self._image_model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
            "output_format": "webp",
            # Muse Image reasons before it renders (default "high"); "low" keeps a postcard
            # well inside the request timeout. Price is per image either way.
            "reasoning_strength": "low",
        }
        body = self._post_json("/images/generations", payload, timeout_s)
        try:
            encoded = body["data"][0]["b64_json"]
            if not isinstance(encoded, str) or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 4:
                raise TypeError
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, KeyError, IndexError, TypeError):
            raise MetaUpstreamError("Meta API returned an unreadable image") from None
        if not is_webp(data) or len(data) > MAX_IMAGE_BYTES:
            raise MetaUpstreamError("Meta API returned an image that is not WebP")
        return data

    def segment(self, png: bytes, point_xy: tuple[int, int], *, timeout_s: float) -> np.ndarray:
        """Ask SAM for the mask containing `point_xy` (pixel x, y) in our own rendered PNG.

        Returns a (height, width) bool array; an all-false array means SAM found nothing.
        """
        if not self._segment_model:
            raise MetaUpstreamError("Segment model is not configured")
        # [CONFIRM] route and fields; masks assumed as uncompressed COCO RLE (SAM's usual output).
        payload = {
            "model": self._segment_model,
            "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "points": [[int(point_xy[0]), int(point_xy[1])]],
            "point_labels": [1],
            "multimask_output": False,
        }
        body = self._post_json("/images/segmentations", payload, timeout_s)
        try:
            masks = body["masks"]
            if not masks:
                return np.zeros((0, 0), dtype=bool)
            return decode_rle(masks[0]["size"], masks[0]["counts"])
        except (ValueError, KeyError, IndexError, TypeError):
            raise MetaUpstreamError("Meta API returned an unreadable mask") from None

    def transcribe(self, audio: bytes, mime: str, lang: str, *, timeout_s: float) -> Transcript:
        """Send one WAV clip to Muse Voice Transcribe. The audio is held in memory only.

        Meta's file endpoint is `POST /asr/transcribe` with two multipart parts: `request`
        (JSON settings) and `audio` (mono 16-bit PCM WAV at 16 or 24 kHz; other containers are
        refused upstream). It answers `{sessionId, transcript, audioDurationMs, turns}`
        (dev.meta.ai/docs/speech-to-text).
        """
        if not self._transcribe_model:
            raise MetaUpstreamError("Voice model is not configured")
        url = self._base + "/asr/transcribe"
        request: dict[str, Any] = {
            "model": self._transcribe_model,
            "mode": "PUSH_TO_TALK",
            "audioEncoding": "WAV",
        }
        bias = language_bias(lang)
        if bias:
            request["languageBias"] = bias
        files: dict[str, tuple[str | None, bytes | str, str]] = {
            "request": (None, json.dumps(request), "application/json"),
            "audio": ("clip.wav", audio, "audio/wav"),
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        try:
            with httpx.Client(timeout=timeout_s, transport=self._transport) as client:
                resp = client.post(url, files=files, headers=headers)
        except httpx.TimeoutException:
            raise MetaUpstreamError("Meta API request timed out") from None
        except httpx.HTTPError as exc:
            raise MetaUpstreamError(f"Meta API request failed ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise MetaUpstreamError(f"Meta API returned HTTP {resp.status_code}")
        try:
            body = resp.json()
            text = body["transcript"]
            if not isinstance(text, str):
                raise TypeError
            duration_ms = body.get("audioDurationMs")
            duration_s = (
                float(duration_ms) / 1000.0
                if isinstance(duration_ms, int | float) and not isinstance(duration_ms, bool)
                else None
            )
            return Transcript(text=text.strip(), duration_s=duration_s)
        except (ValueError, KeyError, TypeError):
            raise MetaUpstreamError("Meta API returned an unreadable response") from None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatTurn:
        try:
            return self._chat_once(self._model, messages, tools, max_tokens, timeout_s)
        except _RetryableStatus as exc:
            if not self._fallback_model:
                raise MetaUpstreamError(f"Meta API returned HTTP {exc.status}") from None
        try:
            return self._chat_once(self._fallback_model, messages, tools, max_tokens, timeout_s)
        except _RetryableStatus as exc:
            raise MetaUpstreamError(f"Meta API fallback returned HTTP {exc.status}") from None

    def _chat_once(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatTurn:
        # max_completion_tokens: max_tokens is a deprecated alias on the Meta Model API.
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if settings.META_CHAT_REASONING_EFFORT:
            payload["reasoning_effort"] = settings.META_CHAT_REASONING_EFFORT
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            with httpx.Client(timeout=timeout_s, transport=self._transport) as client:
                resp = client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise MetaUpstreamError("Meta API request timed out") from None
        except httpx.HTTPError as exc:
            raise MetaUpstreamError(f"Meta API request failed ({type(exc).__name__})") from None

        if resp.status_code in _FALLBACK_STATUSES:
            raise _RetryableStatus(resp.status_code)
        if resp.status_code != 200:
            raise MetaUpstreamError(f"Meta API returned HTTP {resp.status_code}")
        try:
            body = resp.json()
            message = body["choices"][0]["message"]
            calls = [
                ToolCall(
                    id=str(c["id"]),
                    name=str(c["function"]["name"]),
                    arguments=str(c["function"].get("arguments") or "{}"),
                )
                for c in message.get("tool_calls") or []
            ]
            content = message.get("content")
            return ChatTurn(
                content=str(content) if content else None,
                tool_calls=calls,
                usage=_read_usage(body.get("usage")),
            )
        except (ValueError, KeyError, IndexError, TypeError):
            raise MetaUpstreamError("Meta API returned an unreadable response") from None


# BCP-47 primary subtag -> Muse Voice Transcribe `languageBias` name. Only languages named in
# Meta's speech-to-text guide; anything else is left to the model's own detection.
_LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "ar": "Arabic", "vi": "Vietnamese"}


def language_bias(lang: str) -> list[str] | None:
    """`languageBias` hint for a BCP-47 tag, or None to let the model detect the language."""
    name = _LANGUAGE_NAMES.get(lang.split("-")[0].lower())
    return [name] if name else None


def _read_usage(usage: Any) -> tuple[int, int] | None:
    """OpenAI-style `usage` -> (prompt_tokens, completion_tokens), or None if absent/odd."""
    if not isinstance(usage, dict):
        return None
    pin, pout = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(pin, int) and isinstance(pout, int) and pin >= 0 and pout >= 0:
        return pin, pout
    return None


class _RetryableStatus(Exception):
    def __init__(self, status: int) -> None:
        self.status = status


# ---------------------------------------------------------------------------
# Offline mock
# ---------------------------------------------------------------------------

CONTEXT_MARKER = "EXPLORER_CONTEXT_JSON:"
# 96 x 64 lossless WebP: a plain paper-coloured card with blue lines, drawn by code (not by a
# model). Lets the postcard flow run offline; the response is labelled as a mock.
MOCK_POSTCARD_WEBP = base64.b64decode(
    "UklGRjAAAABXRUJQVlA4TCQAAAAvX8APAA9wRfJjud8Yev7jAUTTNv3/6dq6MRfR/31gLrPbXwA="
)
MOCK_TRANSCRIPT = "What is the temperature profile here?"
_ACCURACY_WORDS = re.compile(r"accura|rmse|error|how good|validat|reliab|trust", re.I)
_EVENT_WORDS = re.compile(r"event|cyclone|storm|monsoon|time machine", re.I)


class MockMetaModelClient:
    """Deterministic offline guide. It plans tool calls from keywords and the explorer
    context, then writes an answer that quotes tool results verbatim, so it passes the
    grounding guard. It is clearly labelled as a mock (`MOCK_PROVIDER_LABEL`)."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatTurn:
        last = messages[-1]
        if last["role"] == "user" and tools:
            return ChatTurn(content=None, tool_calls=self._plan(messages))
        return ChatTurn(content=self._answer(messages))

    def transcribe(self, audio: bytes, mime: str, lang: str, *, timeout_s: float) -> Transcript:
        """Fixed offline transcript. The clip itself is not inspected."""
        return Transcript(text=MOCK_TRANSCRIPT, duration_s=None)

    def image(self, prompt: str, *, timeout_s: float) -> bytes:
        """Fixed placeholder WebP (a plain paper card). The prompt is not inspected."""
        return MOCK_POSTCARD_WEBP

    def segment(self, png: bytes, point_xy: tuple[int, int], *, timeout_s: float) -> np.ndarray:
        """Deterministic disc around the tapped pixel, sized from the image header."""
        width, height = png_size(png)
        radius = max(1, max(width, height) // 8)
        ys, xs = np.ogrid[:height, :width]
        x0, y0 = point_xy
        return np.asarray((xs - x0) ** 2 + (ys - y0) ** 2 <= radius**2, dtype=bool)

    def _plan(self, messages: list[dict[str, Any]]) -> list[ToolCall]:
        question = str(messages[-1].get("content") or "")
        ctx = _read_context(messages)
        calls: list[tuple[str, dict[str, Any]]] = []
        if _ACCURACY_WORDS.search(question):
            args: dict[str, Any] = {}
            if ctx.get("depth_m") is not None:
                args["depth_m"] = ctx["depth_m"]
            calls.append(("get_validation_metrics", args))
        elif _EVENT_WORDS.search(question):
            calls.append(("list_events", {}))
        elif ctx.get("lat") is not None and ctx.get("lon") is not None and ctx.get("date"):
            point = {"lat": ctx["lat"], "lon": ctx["lon"]}
            calls.append(("get_temperature_profile", {**point, "date": ctx["date"]}))
            calls.append(("select_point", point))
        else:
            calls.append(("get_dataset_info", {}))
        return [
            ToolCall(id=f"mock_{i}", name=n, arguments=json.dumps(a))
            for i, (n, a) in enumerate(calls)
        ]

    def _answer(self, messages: list[dict[str, Any]]) -> str:
        results: dict[str, dict[str, Any]] = {}
        names = {
            c["id"]: c["function"]["name"]
            for m in messages
            if m["role"] == "assistant"
            for c in m.get("tool_calls") or []
        }
        for m in messages:
            if m["role"] == "tool":
                try:
                    results[names.get(m["tool_call_id"], "")] = json.loads(m["content"])
                except (ValueError, KeyError):
                    continue

        parts: list[str] = []
        for name, res in results.items():
            if isinstance(res.get("error"), dict):
                parts.append("The data doesn't cover that.")
            elif name == "get_temperature_profile":
                parts.append(_describe_profile(res))
            elif name == "get_validation_metrics":
                parts.append(_describe_metrics(res))
            elif name == "list_events":
                titles = [str(e.get("title")) for e in res.get("events", [])]
                parts.append(
                    "Curated events: " + ", ".join(titles) + "."
                    if titles
                    else "No curated events are available yet."
                )
            elif name == "get_dataset_info":
                parts.append(_describe_dataset(res))
        if not parts:
            parts.append("I can answer questions about this dataset's ocean temperatures.")
        if any(r.get("synthetic") for r in results.values()):
            parts.append("These values are synthetic demo data, not model results.")
        parts.append("(Offline mock guide: no Meta model was called.)")
        return " ".join(parts)


def _read_context(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for m in messages:
        if m["role"] == "system":
            text = str(m.get("content") or "")
            idx = text.find(CONTEXT_MARKER)
            if idx >= 0:
                line = text[idx + len(CONTEXT_MARKER) :].splitlines()[0]
                try:
                    ctx = json.loads(line)
                    return ctx if isinstance(ctx, dict) else {}
                except ValueError:
                    return {}
    return {}


def _describe_profile(res: dict[str, Any]) -> str:
    depths = res.get("depths_m") or []
    pred = res.get("pred") or []
    pairs = [(d, v) for d, v in zip(depths, pred) if v is not None]
    if not pairs:
        return "The data doesn't cover that point on that date."
    top, bottom = pairs[0], pairs[-1]
    return (
        f"At {res.get('lat')}, {res.get('lon')} on {res.get('date')}, the model prediction is "
        f"{top[1]} °C at {top[0]} m and {bottom[1]} °C at {bottom[0]} m."
    )


def _describe_metrics(res: dict[str, Any]) -> str:
    rows = res.get("rows", [])
    if not rows:
        return "The data doesn't cover that."
    row = rows[0]
    return (
        f"On the validation split, the {row.get('model')} RMSE at {row.get('depth_m')} m "
        f"is {row.get('rmse_c')} °C against GLORYS."
    )


def _describe_dataset(res: dict[str, Any]) -> str:
    dates = res.get("available_dates") or {}
    depths = res.get("depths_m") or []
    if not depths:
        return "The dataset information is unavailable."
    return (
        f"The data covers {dates.get('start')} to {dates.get('end')} at {len(depths)} depths, "
        f"from {depths[0]} m to {depths[-1]} m."
    )


def decode_rle(size: Any, counts: Any) -> np.ndarray:
    """Uncompressed COCO RLE -> (height, width) bool mask. Runs alternate 0/1 starting with 0,
    in column-major order."""
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0 or height * width > 16_000_000:
        raise ValueError("bad mask size")
    runs = [int(c) for c in counts]
    if any(r < 0 for r in runs) or sum(runs) != height * width:
        raise ValueError("bad mask counts")
    flat = np.zeros(height * width, dtype=bool)
    pos = 0
    for i, run in enumerate(runs):
        if i % 2 == 1:
            flat[pos : pos + run] = True
        pos += run
    return flat.reshape((width, height)).T


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """Inverse of `decode_rle`; used by tests to fake SAM responses."""
    flat = np.asarray(mask, dtype=bool).T.reshape(-1)
    counts: list[int] = []
    current, run = False, 0
    for v in flat:
        if bool(v) == current:
            run += 1
        else:
            counts.append(run)
            current, run = bool(v), 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def png_size(png: bytes) -> tuple[int, int]:
    """(width, height) from a PNG IHDR header."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def build_client() -> MetaModelClient:
    """Create the configured client. Call only when `chat_available()` is true."""
    if settings.META_CLIENT == "mock":
        return MockMetaModelClient()
    assert settings.META_API_BASE_URL and settings.META_API_KEY
    from app.meta.budget import BudgetedClient, get_budget

    # Every real call goes through the credit budget (app/meta/budget.py); no bypass.
    return BudgetedClient(_http_client(), get_budget())


def _http_client() -> HttpMetaModelClient:
    assert settings.META_API_BASE_URL and settings.META_API_KEY
    return HttpMetaModelClient(
        base_url=settings.META_API_BASE_URL,
        api_key=settings.META_API_KEY,
        model=settings.META_CHAT_MODEL or "",
        fallback_model=settings.META_CHAT_FALLBACK_MODEL,
        transcribe_model=settings.META_TRANSCRIBE_MODEL,
        image_model=settings.META_IMAGE_MODEL,
        segment_model=settings.META_SEGMENT_MODEL,
    )
