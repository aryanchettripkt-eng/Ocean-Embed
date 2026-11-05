"""Glasses ocean guide: POST /glasses/ask and its helpers (PRD_BACKEND §4.18, §9 Test 22).

No test calls the real Meta API: audio questions use a fake client.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.v1.glasses as glasses_module
import app.api.v1.voice as voice_module
import app.data.store as store_module
from app.data import glasses
from app.main import app
from app.meta import client as client_module
from app.meta.client import Transcript
from app.settings import settings
from tests.test_voice import make_wav

# Same coastal point as the beach tests: the fixture's Arabian Sea strip lies to the west.
COAST = {"lat": "20.137", "lon": "69.213", "bearing_deg": "270"}
DEPTHS = [0.5, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0]


@pytest.fixture(autouse=True)
def glasses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GLASSES_ENABLED", True)
    monkeypatch.setattr(settings, "BEACH_ENABLED", True)
    monkeypatch.setattr(settings, "VOICE_ENABLED", True)
    monkeypatch.setattr(settings, "META_CLIENT", "mock")
    monkeypatch.setattr(glasses_module, "_limiter", None)
    monkeypatch.setattr(voice_module, "_limiter", None)


def ask(client: TestClient, **form: str) -> tuple[int, dict[str, Any]]:
    res = client.post("/api/v1/glasses/ask", data={**COAST, **form})
    return res.status_code, res.json()


def profile_value(client: TestClient, body: dict[str, Any], depth: float) -> float:
    prof = client.get(
        "/api/v1/profile", params={"lat": body["lat"], "lon": body["lon"], "date": body["date"]}
    ).json()
    return float(prof["pred"][prof["depths_m"].index(depth)])


# --- helpers -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "depth"),
    [
        ("How warm is it out there at 500 m?", 500.0),
        ("temperature 100 m from the shore at a depth of 500 metres", 500.0),
        ("what about 200 meters down", 200.0),
        ("100 m offshore, 250m deep", 250.0),
        ("how warm is the surface", 0.0),
        ("how warm is the sea over there", None),
        ("at 1,5 m", 1.5),
    ],
)
def test_parse_depth(question: str, depth: float | None) -> None:
    assert glasses.parse_depth_m(question) == depth


def test_value_at_depth_exact_blend_top_and_too_deep() -> None:
    values: list[float | None] = [28.0 - i for i in range(len(DEPTHS))]
    exact = glasses.value_at_depth(DEPTHS, values, 500.0)
    assert exact is not None and exact.temperature_c == 16.0 and not exact.interpolated
    mid = glasses.value_at_depth(DEPTHS, values, 400.0)
    assert mid is not None and mid.interpolated and mid.levels_m == [300.0, 500.0]
    assert mid.temperature_c == 16.5  # halfway between 17.0 and 16.0
    top = glasses.value_at_depth(DEPTHS, values, 0.0)
    assert top is not None and top.depth_used_m == 0.5 and top.temperature_c == 28.0
    shelf: list[float | None] = [*values[:6], *([None] * (len(DEPTHS) - 6))]
    assert glasses.value_at_depth(DEPTHS, shelf, 500.0) is None
    deepest = glasses.deepest_value(DEPTHS, shelf)
    assert deepest is not None and deepest.depth_used_m == 50.0


def test_deep_enough_uses_the_level_at_or_below() -> None:
    pred = np.full((len(DEPTHS), 1, 2), 20.0)
    pred[12:, 0, 0] = np.nan  # cell 0 ends above 500 m
    assert glasses.deep_enough(pred, DEPTHS, 400.0).tolist() == [[False, True]]
    assert glasses.deep_enough(pred, DEPTHS, 300.0).tolist() == [[True, True]]


def test_compass_words() -> None:
    assert glasses.compass_word(0) == "north"
    assert glasses.compass_word(359) == "north"
    assert glasses.compass_word(75) == "east-north-east"
    assert glasses.compass_word(270) == "west"


# --- endpoint ------------------------------------------------------------------------------


def test_disabled_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GLASSES_ENABLED", False)
    code, body = ask(TestClient(app), text="how warm at 500 m")
    assert code == 503 and body["error"]["code"] == "glasses_disabled"


def test_typed_question_on_a_level_matches_the_profile() -> None:
    """Test 22: the spoken number is the store's value at that cell, depth and date."""
    client = TestClient(app)
    code, body = ask(client, text="How warm is it out there at 500 m?")
    assert code == 200, body
    assert body["lat"] == 20.5 and body["lon"] == 68.5 and body["distance_km"] >= 5.0
    assert body["depth_used_m"] == 500.0 and not body["interpolated"]
    assert body["temperature_c"] == round(profile_value(client, body, 500.0), 1)
    assert f"{body['temperature_c']:.1f} degrees Celsius" in body["answer_text"]
    assert "500 metres" in body["answer_text"] and body["direction"] in body["answer_text"]
    assert "synthetic demo data" in body["answer_text"] and body["synthetic"] is True


def test_between_levels_is_blended_and_said() -> None:
    client = TestClient(app)
    code, body = ask(client, text="and at 400 metres?")
    assert code == 200, body
    assert body["interpolated"] and body["levels_m"] == [300.0, 500.0]
    a, b = profile_value(client, body, 300.0), profile_value(client, body, 500.0)
    assert body["temperature_c"] == round(a + 0.5 * (b - a), 1)
    assert "blended from the 300 and 500 metre levels" in body["answer_text"]


def test_no_depth_gives_the_surface_and_a_hint() -> None:
    code, body = ask(TestClient(app), text="how warm is the sea over there")
    assert code == 200 and body["depth_asked_m"] is None and body["depth_used_m"] == 0.5
    assert "Ask for a depth" in body["answer_text"]


def test_deeper_than_the_model_uses_the_deepest_level() -> None:
    code, body = ask(TestClient(app), text="what about 3000 m down")
    assert code == 200 and body["depth_used_m"] == 1000.0 and not body["too_shallow"]
    assert "deepest level is 1000 metres" in body["answer_text"]


def test_no_deep_water_in_view_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    def shallow_everywhere(pred: np.ndarray, depths: list[float], depth_m: float) -> np.ndarray:
        return np.zeros(pred.shape[1:], dtype=bool)

    real_profile = store_module.store.get_profile

    def shelf_profile(lat: float, lon: float, date: str) -> Any:
        prof = real_profile(lat, lon, date)
        return prof.model_copy(update={"pred": [*prof.pred[:6], *([None] * 9)]})

    monkeypatch.setattr(glasses, "deep_enough", shallow_everywhere)
    monkeypatch.setattr(store_module.store, "get_profile", shelf_profile)
    code, body = ask(TestClient(app), text="how warm at 500 m")
    assert code == 200 and body["too_shallow"] is True and body["depth_used_m"] == 50.0
    assert "isn't 500 metres deep" in body["answer_text"]
    assert "deepest level with data is 50 metres" in body["answer_text"]


def test_rough_compass_is_mentioned() -> None:
    code, body = ask(TestClient(app), text="at 500 m", heading_accuracy_deg="45")
    assert code == 200 and "compass reading is rough" in body["answer_text"]


def test_spoken_question_is_transcribed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def transcribe(self, audio: bytes, mime: str, lang: str, *, timeout_s: float) -> Any:
            return Transcript(text="How warm is it at 200 metres?", duration_s=2.0)

    monkeypatch.setattr(client_module, "build_client", lambda: Fake())
    res = TestClient(app).post(
        "/api/v1/glasses/ask",
        data=COAST,
        files={"audio": ("q.wav", make_wav(2.0), "audio/wav")},
    )
    assert res.status_code == 200, res.json()
    body = res.json()
    assert body["question"] == "How warm is it at 200 metres?" and body["depth_used_m"] == 200.0


def test_audio_needs_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "VOICE_ENABLED", False)
    res = TestClient(app).post(
        "/api/v1/glasses/ask",
        data=COAST,
        files={"audio": ("q.wav", make_wav(1.0), "audio/wav")},
    )
    assert res.status_code == 503 and res.json()["error"]["code"] == "voice_disabled"


def test_bad_inputs() -> None:
    client = TestClient(app)
    assert ask(client)[1]["error"]["code"] == "empty_question"
    assert ask(client, text="at 500 m", lat="45.0", lon="75.0")[0] == 422
    assert ask(client, text="at 500 m", bearing_deg="360")[0] == 422
    code, body = ask(client, text="at 500 m", bearing_deg="90", max_km="100")
    assert code == 404 and body["error"]["code"] == "no_ocean_in_view"


def test_location_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        ask(TestClient(app), text="at 500 m")
    lines = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app."))
    assert "feature=glasses" in lines
    for fragment in ("20.137", "69.213", "20.1", "69.2"):
        assert fragment not in lines


def test_nearest_ocean_min_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    params = {"lat": 20.137, "lon": 69.213, "bearing_deg": 270}
    res = client.get("/api/v1/nearest-ocean", params={**params, "min_depth_m": 500})
    assert res.status_code == 200 and res.json()["lon"] == 68.5
    res = client.get("/api/v1/nearest-ocean", params={**params, "min_depth_m": 5000})
    assert res.status_code == 422 and res.json()["error"]["code"] == "depth_out_of_range"

    def shallow_everywhere(pred: np.ndarray, depths: list[float], depth_m: float) -> np.ndarray:
        return np.zeros(pred.shape[1:], dtype=bool)

    monkeypatch.setattr(glasses, "deep_enough", shallow_everywhere)
    res = client.get("/api/v1/nearest-ocean", params={**params, "min_depth_m": 500})
    assert res.status_code == 404
