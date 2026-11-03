"""Voice input: POST /transcribe (PRD_BACKEND §4.12, §4.13, §9 Test 17).

No test calls the real Meta API: the HTTP client runs over httpx.MockTransport.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api.v1.voice as voice_module
from app.main import app
from app.meta import client as client_module
from app.meta.client import MOCK_TRANSCRIPT, HttpMetaModelClient
from app.settings import settings

SENTINEL_KEY = "sk-SENTINEL-voice-key-0123456789"
WEBM = "audio/webm;codecs=opus"
CLIP = b"\x1a\x45\xdf\xa3" + b"\x00" * 900_000  # about 900 kB; the bytes are not decoded here


@pytest.fixture(autouse=True)
def voice_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "VOICE_ENABLED", True)
    monkeypatch.setattr(settings, "META_CLIENT", "mock")
    monkeypatch.setattr(settings, "META_API_BASE_URL", None)
    monkeypatch.setattr(settings, "META_API_KEY", None)
    monkeypatch.setattr(settings, "META_TRANSCRIBE_MODEL", None)
    monkeypatch.setattr(settings, "VOICE_RATE_LIMIT", "100/minute")
    monkeypatch.setattr(settings, "VOICE_DAILY_CAP", 1000)
    monkeypatch.setattr(voice_module, "_limiter", None)


def upload(
    client: TestClient, clip: bytes = CLIP, mime: str = WEBM, lang: str | None = None
) -> httpx.Response:
    data = {"lang": lang} if lang is not None else {}
    return client.post("/api/v1/transcribe", files={"audio": ("clip", clip, mime)}, data=data)


def use_http_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    monkeypatch.setattr(settings, "META_CLIENT", "http")
    monkeypatch.setattr(settings, "META_API_BASE_URL", "https://meta.invalid/v1")
    monkeypatch.setattr(settings, "META_API_KEY", SENTINEL_KEY)
    monkeypatch.setattr(settings, "META_TRANSCRIBE_MODEL", "voice-x")
    real = HttpMetaModelClient(
        "https://meta.invalid/v1",
        SENTINEL_KEY,
        "",
        transport=httpx.MockTransport(handler),
        transcribe_model="voice-x",
    )
    monkeypatch.setattr(client_module, "build_client", lambda: real)


def test_disabled_and_misconfigured_return_503(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(settings, "VOICE_ENABLED", False)
    res = upload(client)
    assert res.status_code == 503 and res.json()["error"]["code"] == "voice_disabled"
    assert client.get("/api/v1/meta").json()["features"]["voice"] is False

    # Enabled, but the HTTP client has no transcription model configured.
    monkeypatch.setattr(settings, "VOICE_ENABLED", True)
    monkeypatch.setattr(settings, "META_CLIENT", "http")
    monkeypatch.setattr(settings, "META_API_BASE_URL", "https://meta.invalid/v1")
    monkeypatch.setattr(settings, "META_API_KEY", SENTINEL_KEY)
    res = upload(client)
    assert res.status_code == 503 and res.json()["error"]["code"] == "voice_disabled"
    assert client.get("/api/v1/meta").json()["features"]["voice"] is False


def test_mock_transcript_and_meta_flag() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/meta").json()["features"]["voice"] is True
    res = upload(client, lang="en-IN")
    assert res.status_code == 200
    body = res.json()
    assert body["text"] == MOCK_TRANSCRIPT
    assert body["duration_s"] is None
    assert body["run_id"] == "synthetic_v1" and body["synthetic"] is True
    assert "Offline mock" in body["provider_label"]
    assert res.headers["Cache-Control"] == "no-store"


def test_size_type_and_language_limits() -> None:
    client = TestClient(app)
    too_big = upload(client, clip=b"x" * 1_000_001)
    assert too_big.status_code == 413 and too_big.json()["error"]["code"] == "audio_too_large"

    wrong_type = upload(client, mime="text/plain")
    assert wrong_type.status_code == 415
    assert wrong_type.json()["error"]["code"] == "unsupported_audio"

    bad_lang = upload(client, lang="not a tag!")
    assert bad_lang.status_code == 422 and bad_lang.json()["error"]["code"] == "invalid_lang"

    empty = upload(client, clip=b"")
    assert empty.status_code == 422 and empty.json()["error"]["code"] == "empty_audio"


def test_upload_without_content_length_is_refused() -> None:
    def body() -> Iterator[bytes]:
        yield b"--x\r\n"

    client = TestClient(app)
    res = client.post(
        "/api/v1/transcribe",
        content=body(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert res.status_code == 411 and res.json()["error"]["code"] == "length_required"


def test_clip_is_never_written_to_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test 17: audio stays in memory; the temp dir stays empty for accepted and refused clips."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = TestClient(app)
    assert upload(client).status_code == 200
    assert upload(client, clip=b"x" * 1_000_001).status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_http_client_sends_clip_and_keeps_key_in_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "  hello there  ", "duration": 4.2})

    use_http_client(monkeypatch, handler)
    client = TestClient(app)
    res = upload(client, lang="hi-IN")
    assert res.status_code == 200
    assert res.json()["text"] == "hello there"
    assert res.json()["duration_s"] == 4.2
    assert SENTINEL_KEY not in res.text

    req = seen[0]
    assert req.url.path.endswith("/audio/transcriptions")
    assert req.headers["Authorization"] == f"Bearer {SENTINEL_KEY}"
    body = req.read()
    assert b'name="model"' in body and b"voice-x" in body
    assert b'name="language"' in body and b"hi-IN" in body
    assert CLIP[:4] in body


def test_upstream_failure_is_502_without_leaks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"SECRET-BODY {SENTINEL_KEY}")

    use_http_client(monkeypatch, handler)
    with caplog.at_level(logging.DEBUG):
        res = upload(TestClient(app))
    assert res.status_code == 502 and res.json()["error"]["code"] == "upstream_error"
    for leaked in (SENTINEL_KEY, "SECRET-BODY"):
        assert leaked not in res.text
        assert leaked not in caplog.text


def test_rate_limit_per_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "VOICE_RATE_LIMIT", "2/minute")
    client = TestClient(app)
    codes = [upload(client, clip=b"small").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
