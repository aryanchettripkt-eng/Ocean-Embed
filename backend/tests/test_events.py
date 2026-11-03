"""Curated events: validation, visibility and GET /events (PRD_BACKEND §4.14, §9 Test 18)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.data.events as events_module
from app.data.events import (
    EventsFileError,
    check_events_file,
    load_events,
    validate_events,
    visible_events,
)
from app.main import app

REGION = {"lon_min": 40.0, "lon_max": 100.0, "lat_min": 0.0, "lat_max": 30.0}
# Real model depths (ml/configs/experiment.yaml, confirmed 2026-10-24).
REAL_DEPTHS = [
    0.494, 5.078, 9.573, 18.496, 29.445, 47.374, 65.807, 92.326, 130.666, 186.126,
    222.475, 266.040, 380.213, 643.567, 1062.440,
]  # fmt: skip
REAL_AVAILABLE = {"start": "1993-01-01", "end": "2021-12-31"}
TEST_RANGE = ["2021-01-01", "2021-12-31"]


def fani() -> dict[str, Any]:
    return {
        "id": "cyclone-fani-2019",
        "title": "Cyclone Fani",
        "kind": "cyclone",
        "start": "2019-04-26",
        "end": "2019-05-04",
        "before": "2019-04-20",
        "after": "2019-05-08",
        "bbox": {"lat_min": 8, "lat_max": 22, "lon_min": 80, "lon_max": 92},
        "suggested_depths_m": [0.5, 50, 100],
        "summary": "A severe cyclone crossed the Bay of Bengal and made landfall in India.",
        "source_url": "https://rsmcnewdelhi.imd.gov.in/uploads/archive/60/60_a53fa0_fani.pdf",
        "source_name": "India Meteorological Department",
    }


def run_validate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return validate_events(raw, region=REGION, depths_m=REAL_DEPTHS, available=REAL_AVAILABLE)


def test_valid_event_snaps_depths_to_model_levels() -> None:
    cleaned = run_validate([fani()])
    assert cleaned[0]["suggested_depths_m"] == [0.494, 47.374, 92.326]
    assert cleaned[0]["bbox"] == {"lat_min": 8.0, "lat_max": 22.0, "lon_min": 80.0, "lon_max": 92.0}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(summary="Landfall came on 4 May 2019."),  # numbers in summary
        lambda e: e.update(source_url="http://example.org/fani"),  # not https
        lambda e: e.update(start="2019-02-30"),  # not a real date
        lambda e: e.update(start="2019-05-10", end="2019-05-04"),  # end before start
        lambda e: e.update(after="2018-12-01"),  # ordering broken
        lambda e: e.update(before="1992-12-01", start="1993-01-02"),  # before available range
        lambda e: e.update(bbox={"lat_min": 8, "lat_max": 35, "lon_min": 80, "lon_max": 92}),
        lambda e: e.update(suggested_depths_m=[500]),  # no model depth nearby
        lambda e: e.update(suggested_depths_m=[]),
        lambda e: e.update(id="Cyclone Fani"),  # not a slug
        lambda e: e.update(extra_field="typo"),  # unknown field
        lambda e: e.pop("source_name"),  # missing field
    ],
)
def test_bad_events_fail_validation(mutate: Any) -> None:
    event = fani()
    mutate(event)
    with pytest.raises(EventsFileError):
        run_validate([event])


def test_duplicate_ids_fail() -> None:
    with pytest.raises(EventsFileError, match="duplicate"):
        run_validate([fani(), fani()])


def test_sealed_test_period_hidden_until_test_is_run() -> None:
    cleaned = run_validate([fani()])
    late = copy.deepcopy(fani())
    late.update(
        id="test-year-event",
        before="2020-12-25",
        start="2021-01-02",
        end="2021-01-10",
        after="2021-01-20",
    )
    cleaned.append(late)
    assert [
        e["id"] for e in visible_events(cleaned, test_range=TEST_RANGE, test_evaluated=False)
    ] == ["cyclone-fani-2019"]
    assert len(visible_events(cleaned, test_range=TEST_RANGE, test_evaluated=True)) == 2


def test_missing_file_means_no_events_and_bad_file_fails(tmp_path: Path) -> None:
    manifest = {
        "region": REGION,
        "depths_m": REAL_DEPTHS,
        "available_dates": REAL_AVAILABLE,
    }
    assert check_events_file(manifest, tmp_path / "absent.json") == 0

    bad = tmp_path / "events.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(EventsFileError):
        check_events_file(manifest, bad)

    bad.write_text(
        json.dumps({"events": [fani() | {"summary": "Cited 3 times."}]}), encoding="utf-8"
    )
    with pytest.raises(EventsFileError):
        check_events_file(manifest, bad)


def test_get_events_empty_until_the_list_is_approved() -> None:
    res = TestClient(app).get("/api/v1/events")
    assert res.status_code == 200
    body = res.json()
    assert body == {"events": [], "run_id": "synthetic_v1", "synthetic": True}
    assert res.headers["Cache-Control"] == "public, max-age=300"


def test_get_events_serves_validated_list_on_fixture_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sample outputs cover 2020-01-01 to 2020-01-10, so this test event sits inside that range.
    event = fani() | {
        "id": "sample-event-2020",
        "before": "2020-01-01",
        "start": "2020-01-02",
        "end": "2020-01-05",
        "after": "2020-01-08",
    }
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"events": [event]}), encoding="utf-8")
    monkeypatch.setattr(events_module, "EVENTS_PATH", path)

    res = TestClient(app).get("/api/v1/events")
    assert res.status_code == 200
    (served,) = res.json()["events"]
    assert served["id"] == "sample-event-2020"
    assert set(served) == set(event)
    assert served["suggested_depths_m"] == [0.5, 50.0, 100.0]
    assert load_events(path)[0]["id"] == "sample-event-2020"
