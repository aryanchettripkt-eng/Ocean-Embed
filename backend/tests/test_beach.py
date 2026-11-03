"""Beach mode: GET /nearest-ocean (PRD_BACKEND §4.17, §9 Test 21)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.data import nearest
from app.main import app
from app.settings import settings

# On the fixture's land block just east of the Arabian Sea strip (ocean ends at lon 69.0).
COAST = {"lat": 20.137, "lon": 69.213}


@pytest.fixture(autouse=True)
def beach_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "BEACH_ENABLED", True)


def get(client: TestClient, **params: float) -> tuple[int, dict]:  # type: ignore[type-arg]
    res = client.get("/api/v1/nearest-ocean", params=params)
    return res.status_code, res.json()


def test_disabled_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "BEACH_ENABLED", False)
    client = TestClient(app)
    code, body = get(client, **COAST)
    assert code == 503 and body["error"]["code"] == "beach_disabled"
    assert client.get("/api/v1/meta").json()["features"]["beach"] is False


def test_seaward_bearing_finds_ocean_at_least_5_km_away() -> None:
    """Test 21: a coastal point with a seaward bearing returns an ocean cell >= 5 km away."""
    client = TestClient(app)
    assert client.get("/api/v1/meta").json()["features"]["beach"] is True
    code, body = get(client, **COAST, bearing_deg=270)
    assert code == 200, body
    assert body["lon"] == 68.5 and body["lat"] == 20.5
    assert body["distance_km"] >= 5.0
    assert 270 < body["bearing_deg"] < 360  # west-north-west to the cell centre
    assert body["date"] == "2020-01-10" and body["synthetic"] is True


def test_landward_bearing_is_404() -> None:
    """Test 21: a landward bearing with no ocean within max_km returns 404."""
    code, body = get(TestClient(app), **COAST, bearing_deg=90, max_km=100)
    assert code == 404 and body["error"]["code"] == "no_ocean_in_view"


def test_without_bearing_returns_nearest_cell() -> None:
    code, body = get(TestClient(app), **COAST)
    assert code == 200
    assert body["lon"] == 68.5 and 5.0 <= body["distance_km"] < 120


def test_far_outside_region_is_422_and_inputs_are_checked() -> None:
    client = TestClient(app)
    code, body = get(client, lat=45.0, lon=75.0)
    assert code == 422 and body["error"]["code"] == "out_of_region"
    assert get(client, **COAST, bearing_deg=360)[0] == 422
    assert get(client, **COAST, max_km=501)[0] == 422


def test_location_is_never_logged_finer_than_1_degree(caplog: pytest.LogCaptureFixture) -> None:
    """Test 21: no location in the logs at better than 1°."""
    client = TestClient(app)
    with caplog.at_level(logging.DEBUG):
        get(client, **COAST, bearing_deg=270)
        get(client, **COAST, bearing_deg=90, max_km=100)
        get(client, lat=45.123, lon=75.456)
    # The app's own log lines (the test client's request log is not part of the service).
    app_lines = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app."))
    assert "feature=beach" in app_lines
    for fragment in ("20.137", "69.213", "20.1", "69.2", "45.123", "75.456"):
        assert fragment not in app_lines


def test_access_log_drops_location_query() -> None:
    """uvicorn's access log would otherwise record the full query string."""
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", "/api/v1/nearest-ocean?lat=20.137&lon=69.213", "1.1", 200),
        None,
    )
    for f in logging.getLogger("uvicorn.access").filters:
        f.filter(record)
    line = record.getMessage()
    assert "20.137" not in line and "69.213" not in line
    assert "/api/v1/nearest-ocean?<redacted>" in line


def test_geometry_helpers() -> None:
    lat, lon = nearest.destination(10.0, 70.0, 90.0, 111.2)
    assert abs(lat - 10.0) < 0.05 and abs(lon - 71.015) < 0.05
    assert abs(nearest.initial_bearing_deg(10.0, 70.0, 10.0, 71.0) - 90.0) < 0.2
    assert (
        nearest.km_outside_bbox(
            15, 70, {"lat_min": 0, "lat_max": 30, "lon_min": 40, "lon_max": 100}
        )
        == 0
    )
