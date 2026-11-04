"""Lasso: POST /segment (PRD_BACKEND §4.16, §9 Test 20).

SAM is always mocked (a fake client or httpx.MockTransport); no test calls the Meta API.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from meta_sam_parser._mask_codec import _encode_raster

import app.api.v1.segment as segment_api
from app.main import app
from app.meta import client as client_module
from app.meta import segment as seg
from app.meta.client import HttpMetaModelClient, MetaUpstreamError, png_size, sam_text_to_mask
from app.schemas.caveats import SEGMENT_CAVEAT
from app.settings import settings
from tests.conftest import FIXTURES_DIR

SENTINEL_KEY = "sk-SENTINEL-segment-key-0123456789"
BODY = {"date": "2020-01-05", "depth_m": 100.0, "field": "pred", "lat": 5.5, "lon": 75.5}
ZARR = FIXTURES_DIR / "predictions" / "predictions_synthetic_v1.zarr"


def fixture_grid(date: str = "2020-01-05", depth_m: float = 100.0) -> tuple[Any, Any, Any]:
    ds = xr.open_zarr(ZARR)
    day = ds.sel(time=date)
    values = day["temp_pred"].sel(depth=depth_m).values.astype(np.float32)
    return values, day["coverage"].values.astype(int), ds


def cells_to_pixels(cells: np.ndarray, png: bytes) -> np.ndarray:
    width, _ = png_size(png)
    scale = width // cells.shape[1]
    return np.repeat(np.repeat(cells[::-1], scale, axis=0), scale, axis=1)


class FakeSam:
    """Returns a fixed cell rectangle as a pixel mask, as SAM would on our rendered PNG."""

    def __init__(self, cells: np.ndarray | None = None, fail: bool = False) -> None:
        self.cells = cells
        self.fail = fail
        self.calls: list[tuple[int, int]] = []

    def segment(self, png: bytes, point_xy: tuple[int, int], *, timeout_s: float) -> np.ndarray:
        self.calls.append(point_xy)
        if self.fail:
            raise MetaUpstreamError("timed out")
        assert self.cells is not None
        return cells_to_pixels(self.cells, png)


def rectangle(r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    cells = np.zeros((30, 60), dtype=bool)
    cells[r0 : r1 + 1, c0 : c1 + 1] = True
    return cells


@pytest.fixture(autouse=True)
def segment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEGMENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEGMENT_PROVIDER", "sam")
    monkeypatch.setattr(settings, "SEGMENT_THRESHOLD_C", 0.5)
    monkeypatch.setattr(settings, "META_CLIENT", "mock")
    monkeypatch.setattr(settings, "META_API_BASE_URL", None)
    monkeypatch.setattr(settings, "META_API_KEY", None)
    monkeypatch.setattr(settings, "META_SEGMENT_MODEL", None)
    monkeypatch.setattr(settings, "SEGMENT_RATE_LIMIT", "100/minute")
    monkeypatch.setattr(settings, "SEGMENT_DAILY_CAP", 1000)
    monkeypatch.setattr(segment_api, "_limiter", None)
    segment_api.clear_cache()


def use_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeSam) -> None:
    monkeypatch.setattr(client_module, "build_client", lambda: fake)


def test_disabled_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SEGMENT_ENABLED", False)
    client = TestClient(app)
    res = client.post("/api/v1/segment", json=BODY)
    assert res.status_code == 503 and res.json()["error"]["code"] == "segment_disabled"
    assert client.get("/api/v1/meta").json()["features"]["segment"] is False


def test_sam_stats_equal_region_summary_over_same_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 20: SAM mocked; stats equal /region-summary over the same cells. The rectangle
    crosses land, so this also checks the coverage intersection."""
    fake = FakeSam(rectangle(3, 14, 25, 40))  # lat 3.5-14.5, lon 65.5-80.5
    use_fake(monkeypatch, fake)
    client = TestClient(app)
    meta = client.get("/api/v1/meta").json()
    assert meta["features"]["segment_method"] == "sam"
    assert meta["caveats"]["segment"] == SEGMENT_CAVEAT

    res = client.post("/api/v1/segment", json=BODY)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["method"] == "sam"
    assert body["caveat"] == SEGMENT_CAVEAT
    assert body["synthetic"] is True and body["run_id"] == "synthetic_v1"
    assert len(fake.calls) == 1

    summary = client.get(
        "/api/v1/region-summary",
        params={
            "date": "2020-01-05",
            "depth_m": 100.0,
            "lat_min": 3.5,
            "lat_max": 14.5,
            "lon_min": 65.5,
            "lon_max": 80.5,
        },
    ).json()
    stats = body["stats"]
    for k in ("mean", "min", "max", "std", "n_valid"):
        assert stats[k] == summary[k], k
    assert stats["area_km2"] > 0

    # The mask never includes coverage != 1 cells.
    values, coverage, _ = fixture_grid()
    for r, c0, c1 in body["mask"]["lat_index_ranges"]:
        assert (coverage[r, c0 : c1 + 1] == 1).all()
        assert not np.isnan(values[r, c0 : c1 + 1]).any()
    n_cells = sum(c1 - c0 + 1 for _, c0, c1 in body["mask"]["lat_index_ranges"])
    assert n_cells == stats["n_valid"]
    assert (rectangle(3, 14, 25, 40) & (coverage != 1)).any()  # the shape did cross land

    outline = body["outline"]
    assert outline[0] == outline[-1] and len(outline) >= 5
    lons = [p[0] for p in outline]
    lats = [p[1] for p in outline]
    assert min(lons) >= 65.0 and max(lons) <= 81.0 and min(lats) >= 3.0 and max(lats) <= 15.0


@pytest.mark.parametrize("mode", ["fails", "empty", "provider_threshold"])
def test_threshold_fallback_is_used_and_reported(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Test 20: SAM failing, empty, or SEGMENT_PROVIDER=threshold -> threshold, reported."""
    if mode == "fails":
        use_fake(monkeypatch, FakeSam(fail=True))
    elif mode == "empty":
        use_fake(monkeypatch, FakeSam(np.zeros((30, 60), dtype=bool)))
    else:
        monkeypatch.setattr(settings, "SEGMENT_PROVIDER", "threshold")
        fake = FakeSam(fail=True)
        use_fake(monkeypatch, fake)

    client = TestClient(app)
    res = client.post("/api/v1/segment", json=BODY)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["method"] == "threshold"
    assert "Similar-temperature" in body["provider_label"]
    if mode == "provider_threshold":
        assert fake.calls == []
        assert client.get("/api/v1/meta").json()["features"]["segment_method"] == "threshold"

    values, coverage, _ = fixture_grid()
    seed = values[5, 35]
    for r, c0, c1 in body["mask"]["lat_index_ranges"]:
        row = values[r, c0 : c1 + 1]
        assert (coverage[r, c0 : c1 + 1] == 1).all()
        assert (np.abs(row - seed) <= 0.5 + 1e-6).all()
    assert body["stats"]["n_valid"] > 1


def test_threshold_cap_holds() -> None:
    """Test 20: the 5,000-cell cap holds on a large uniform field."""
    values = np.full((120, 120), 20.0, dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    mask, capped = seg.threshold_select(values, valid, (60, 60), 0.5)
    assert capped is True
    assert int(mask.sum()) == seg.MAX_THRESHOLD_CELLS == 5_000

    small, capped_small = seg.threshold_select(values[:10, :10], valid[:10, :10], (0, 0), 0.5)
    assert capped_small is False and int(small.sum()) == 100


def test_bad_points_and_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake(monkeypatch, FakeSam(fail=True))
    client = TestClient(app)
    land = client.post("/api/v1/segment", json={**BODY, "lat": 20.5, "lon": 75.5})
    assert land.status_code == 422 and land.json()["error"]["code"] == "point_not_ocean"
    outside = client.post("/api/v1/segment", json={**BODY, "lat": 45.0})
    assert outside.status_code == 422 and outside.json()["error"]["code"] == "out_of_region"
    free_text = client.post("/api/v1/segment", json={**BODY, "image": "data:..."})
    assert free_text.status_code == 422
    error_field = client.post("/api/v1/segment", json={**BODY, "field": "error"})
    assert error_field.status_code == 422


def test_identical_requests_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSam(rectangle(3, 8, 30, 40))
    use_fake(monkeypatch, fake)
    client = TestClient(app)
    first = client.post("/api/v1/segment", json=BODY).json()
    second = client.post("/api/v1/segment", json={**BODY, "lat": 5.6}).json()  # same cell
    assert first == second and len(fake.calls) == 1


def test_rendered_png_is_our_grid_and_point_maps_to_tapped_cell() -> None:
    values, coverage, _ = fixture_grid()
    rendered = seg.render_grid_png(values, coverage, 10.0, 30.0)
    width, height = png_size(rendered.png)
    assert (width, height) == (60 * rendered.scale, 30 * rendered.scale)
    cells = np.zeros((30, 60), dtype=bool)
    cells[5, 35] = True
    pixels = cells_to_pixels(cells, rendered.png)
    x, y = rendered.pixel_of(5, 35)
    assert pixels[y, x]
    assert (rendered.mask_to_cells(pixels) == cells).all()


def use_http_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    monkeypatch.setattr(settings, "META_CLIENT", "http")
    monkeypatch.setattr(settings, "META_API_BASE_URL", "https://meta.invalid/v1")
    monkeypatch.setattr(settings, "META_API_KEY", SENTINEL_KEY)
    monkeypatch.setattr(settings, "META_SEGMENT_MODEL", "sam-x")
    real = HttpMetaModelClient(
        "https://meta.invalid/v1",
        SENTINEL_KEY,
        "",
        transport=httpx.MockTransport(handler),
        segment_model="sam-x",
    )
    monkeypatch.setattr(client_module, "build_client", lambda: real)


def sam_record(mask: np.ndarray, object_id: int = 1, step: int = 1) -> str:
    """One SAM 3.1 output record for `mask`: a box around it and its box-local raster, encoded
    with Meta's own codec. `step` > 1 sends a coarser raster, as SAM may."""
    height, width = mask.shape
    rows, cols = np.nonzero(mask)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    local = mask[top : bottom + 1 : step, left : right + 1 : step]
    h, w = local.shape
    payload = _encode_raster([int(v) for v in local.reshape(-1)], w, h)
    return (
        f"{object_id}<|box;x1={left};y1={top};x2={right};y2={bottom};w={width};h={height}|>"
        f"<|mask;x=0;y=0;data={h},{w},{payload}|>"
    )


def sse(*events: dict[str, Any]) -> httpx.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def sam_stream(text: str, completed: bool = True) -> httpx.Response:
    mid = len(text) // 2
    events: list[dict[str, Any]] = [
        {"type": "response.output_text.delta", "delta": text[:mid]},
        {"type": "response.output_text.delta", "delta": text[mid:]},
        {"type": "response.content_part.done", "part": {"type": "output_text", "text": text}},
    ]
    if completed:
        events.append({"type": "response.completed", "response": {"status": "completed"}})
    return sse(*events)


def test_http_client_calls_sam_through_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        payload = json.loads(request.read())
        content = payload["input"][0]["content"]
        png = base64.b64decode(content[1]["image_url"].split(",", 1)[1])
        mask = cells_to_pixels(rectangle(3, 8, 30, 40), png)
        return sam_stream("<0f>" + sam_record(mask) + "\n")

    use_http_client(monkeypatch, handler)
    res = TestClient(app).post("/api/v1/segment", json=BODY)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["method"] == "sam" and body["provider_label"] == "SAM 3.1 via Meta Model API"
    assert SENTINEL_KEY not in res.text
    ranges = body["mask"]["lat_index_ranges"]
    assert ranges[0] == [3, 30, 40] and ranges[-1] == [8, 30, 38]  # cols 39-40 are land

    req = seen[0]
    assert req.url.path == "/v1/responses"
    assert req.headers["Authorization"] == f"Bearer {SENTINEL_KEY}"
    assert req.headers["Accept"] == "text/event-stream"
    payload = json.loads(req.read())
    assert payload["model"] == "sam-x" and payload["stream"] is True
    message = payload["input"][0]
    assert message["type"] == "message" and message["role"] == "user"
    text_part, image_part = message["content"]
    assert text_part["type"] == "input_text" and image_part["type"] == "input_image"
    png = base64.b64decode(image_part["image_url"].split(",", 1)[1])
    width, height = png_size(png)
    # The tapped cell's centre pixel, as one positive point for object 1 on frame 0.
    assert text_part["text"].startswith("<0f>1<|point;x=")
    assert text_part["text"].endswith(f";w={width};h={height}|>")


def test_coarse_and_multiple_sam_masks_are_placed_in_their_boxes() -> None:
    full = np.zeros((40, 60), dtype=bool)
    full[4:20, 6:30] = True
    other = np.zeros((40, 60), dtype=bool)
    other[30:36, 40:50] = True
    text = "<0f>" + sam_record(full, 1, step=2) + "," + sam_record(other, 2)
    out = sam_text_to_mask(text, 60, 40)
    assert out.shape == (40, 60)
    assert (out == (full | other)).all()


def test_sam_json_reply_and_nothing_found() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:9, 10:20] = True
    body = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "<0f>" + sam_record(mask)}],
            }
        ],
    }
    client = HttpMetaModelClient(
        "https://meta.invalid/v1",
        SENTINEL_KEY,
        "",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
        segment_model="sam-x",
    )
    png = seg.render_grid_png(np.zeros((20, 30), np.float32), np.ones((20, 30), int), 0, 1).png
    width, height = png_size(png)
    out = client.segment(png, (1, 1), timeout_s=5)
    assert out.shape == (height, width) and out.any()
    assert not sam_text_to_mask("<0f>\n", width, height).any()


@pytest.mark.parametrize(
    "reply",
    [
        sam_stream("<0f>", completed=False),  # stream ended without response.completed
        sse({"type": "response.failed", "response": {"status": "failed"}}),
        sse({"type": "error", "message": "bad prompt"}),
        sam_stream("<0f>1<|box;x1=0;y1=0;x2=3;y2=3;w=8;h=8|><|mask;x=0;y=0;data=4,4,!@@|>\n"),
        httpx.Response(200, json={"status": "incomplete", "output": []}),
    ],
)
def test_bad_sam_replies_fall_back_to_threshold(
    monkeypatch: pytest.MonkeyPatch, reply: httpx.Response
) -> None:
    use_http_client(monkeypatch, lambda r: reply)
    res = TestClient(app).post("/api/v1/segment", json=BODY)
    assert res.status_code == 200 and res.json()["method"] == "threshold"


def test_upstream_failure_falls_back_without_leaks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    use_http_client(monkeypatch, lambda r: httpx.Response(500, text=f"SECRET-BODY {SENTINEL_KEY}"))
    with caplog.at_level(logging.DEBUG):
        res = TestClient(app).post("/api/v1/segment", json=BODY)
    assert res.status_code == 200 and res.json()["method"] == "threshold"
    for leaked in (SENTINEL_KEY, "SECRET-BODY"):
        assert leaked not in res.text and leaked not in caplog.text
