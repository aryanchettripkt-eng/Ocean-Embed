"""Tests for backend data endpoints: /map, /profile, /timeseries, /region-summary.

Verifies PRD_BACKEND Sections 4.3-4.6, 9 (Tests 2, 3, 4, 5, 6, 10), and Milestone B2.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr
from fastapi.testclient import TestClient

from app.main import app


def test_map_happy_path_and_contract() -> None:
    """Verify /map satisfies Section 4.3 and Test 4 (shapes, nulls, coverage)."""
    client = TestClient(app)
    response = client.get("/api/v1/map?date=2020-01-01&depth_m=0.5&field=pred")
    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "public, max-age=86400"
    assert "ETag" in response.headers

    data = response.json()
    assert data["run_id"] == "synthetic_v1"
    assert data["synthetic"] is True
    assert data["date"] == "2020-01-01"
    assert data["depth_m"] == 0.5
    assert data["field"] == "pred"
    assert data["split"] == "val"
    assert data["units"] == "degC"

    # Shapes equal grid_shape: lat 30, lon 60
    assert len(data["lat"]) == 30
    assert len(data["lon"]) == 60
    assert len(data["values"]) == 30
    assert len(data["values"][0]) == 60
    assert len(data["coverage"]) == 30
    assert len(data["coverage"][0]) == 60

    # Test 4: nulls exactly where coverage != 1 or fixture has NaN
    vals = np.array(data["values"], dtype=float)
    cov = np.array(data["coverage"], dtype=int)
    assert np.all(np.isnan(vals[cov != 1])), "values must be null where coverage != 1"
    assert np.all(~np.isnan(vals[cov == 1])), "values must be present where coverage == 1"

    # Coverage codes pass through
    assert set(np.unique(cov)).issubset({0, 1, 2})

    # Stats validation
    assert data["stats"]["n_valid"] > 0
    assert data["stats"]["min"] is not None
    assert data["stats"]["max"] is not None
    assert data["stats"]["mean"] is not None
    assert data["stats"]["min"] <= data["stats"]["mean"] <= data["stats"]["max"]


def test_map_depth_index_and_glorys_and_error() -> None:
    """Verify depth_index alternative, field=glorys, and field=error."""
    client = TestClient(app)
    # Using depth_index=0
    res_glorys = client.get("/api/v1/map?date=2020-01-01&depth_index=0&field=glorys")
    assert res_glorys.status_code == 200
    g_data = res_glorys.json()
    assert g_data["field"] == "glorys"
    assert g_data["depth_m"] == 0.5

    # Using field=error (pred - glorys)
    res_err = client.get("/api/v1/map?date=2020-01-01&depth_index=0&field=error")
    assert res_err.status_code == 200
    err_data = res_err.json()
    assert err_data["field"] == "error"


def test_map_errors() -> None:
    """Verify /map error handling (bad depth, both depth, date unavailable)."""
    client = TestClient(app)

    # Both depth parameters -> 422 both_depth_params
    res_both = client.get("/api/v1/map?date=2020-01-01&depth_m=0.5&depth_index=0&field=pred")
    assert res_both.status_code == 422
    assert res_both.json()["error"]["code"] == "both_depth_params"

    # Bad depth_m -> 422 bad_depth
    res_bad_depth = client.get("/api/v1/map?date=2020-01-01&depth_m=999.0&field=pred")
    assert res_bad_depth.status_code == 422
    assert res_bad_depth.json()["error"]["code"] == "bad_depth"

    # Bad depth_index -> 422 bad_depth
    res_bad_idx = client.get("/api/v1/map?date=2020-01-01&depth_index=25&field=pred")
    assert res_bad_idx.status_code == 422
    assert res_bad_idx.json()["error"]["code"] == "bad_depth"

    # Date unavailable -> 404 date_unavailable
    res_bad_date = client.get("/api/v1/map?date=2015-01-01&depth_index=0&field=pred")
    assert res_bad_date.status_code == 404
    assert res_bad_date.json()["error"]["code"] == "date_unavailable"


def test_map_live_date_handling() -> None:
    """Verify Test 10: live dates return split: live and 404 for glorys/error."""
    client = TestClient(app)

    # field=pred on live date works
    res_live_pred = client.get("/api/v1/map?date=2020-01-09&depth_index=0&field=pred")
    assert res_live_pred.status_code == 200
    assert res_live_pred.json()["split"] == "live"

    # field=glorys on live date -> 404 reference_unavailable
    res_live_g = client.get("/api/v1/map?date=2020-01-09&depth_index=0&field=glorys")
    assert res_live_g.status_code == 404
    assert res_live_g.json()["error"]["code"] == "reference_unavailable"

    # field=error on live date -> 404 reference_unavailable
    res_live_err = client.get("/api/v1/map?date=2020-01-09&depth_index=0&field=error")
    assert res_live_err.status_code == 404
    assert res_live_err.json()["error"]["code"] == "reference_unavailable"


def test_profile_and_raw_zarr_match() -> None:
    """Verify Section 4.4 and Test 6: /profile values equal raw zarr values rounded to 2 dp."""
    fixtures_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"
    zarr_path = fixtures_dir / "predictions" / "predictions_synthetic_v1.zarr"
    ds = xr.open_zarr(str(zarr_path))

    client = TestClient(app)
    # Cell at Arabian Sea ocean point: lat=12.5, lon=65.5
    res = client.get("/api/v1/profile?lat=12.5&lon=65.5&date=2020-01-01")
    assert res.status_code == 200
    assert res.headers.get("Cache-Control") == "public, max-age=86400"

    data = res.json()
    assert data["date"] == "2020-01-01"
    assert data["lat"] == 12.5
    assert data["lon"] == 65.5
    assert data["split"] == "val"
    assert data["coverage"] == 1
    assert len(data["depths_m"]) == 15
    assert len(data["pred"]) == 15
    assert len(data["glorys"]) == 15
    assert len(data["error"]) == 15

    # Test 6: Compare against raw zarr values cast to float32, rounded to 2 dp
    raw_cell = ds.sel(time="2020-01-01").sel(lat=12.5, lon=65.5, method="nearest")
    raw_pred = [round(float(v), 2) for v in raw_cell["temp_pred"].values.astype(np.float32)]
    raw_glorys = [round(float(v), 2) for v in raw_cell["temp_glorys"].values.astype(np.float32)]

    assert data["pred"] == raw_pred
    assert data["glorys"] == raw_glorys

    # Check error == pred - glorys
    for p, g, e in zip(data["pred"], data["glorys"], data["error"]):
        assert p is not None and g is not None and e is not None
        assert abs(round(p - g, 2) - e) < 0.02


def test_profile_live_date() -> None:
    """Verify Test 10: /profile on live date has split: live and glorys/error: null."""
    client = TestClient(app)
    res = client.get("/api/v1/profile?lat=12.5&lon=65.5&date=2020-01-09")
    assert res.status_code == 200
    data = res.json()
    assert data["split"] == "live"
    assert all(v is None for v in data["glorys"])
    assert all(v is None for v in data["error"])
    assert any(v is not None for v in data["pred"])


def test_snapping_round_half_up() -> None:
    """Verify Test 3: deterministic snapping with round-half-up on index."""
    client = TestClient(app)

    # In fixture, centres are 0.5, 1.5, 2.5...
    # Exact midpoint between 0.5 and 1.5 is lat=1.0. With round-half-up on index,
    # frac_idx = (1.0 - 0.5)/1.0 = 0.5 -> floor(0.5 + 0.5) = 1 -> snaps to 1.5.
    res_mid = client.get("/api/v1/profile?lat=1.0&lon=50.0&date=2020-01-01")
    assert res_mid.status_code == 200
    assert res_mid.json()["lat"] == 1.5
    assert res_mid.json()["lon"] == 50.5  # 50.0 midpoint between 49.5 and 50.5 snaps to 50.5

    # Just below midpoint: lat=0.99 -> frac_idx=0.49 -> snaps to 0.5
    res_below = client.get("/api/v1/profile?lat=0.99&lon=50.0&date=2020-01-01")
    assert res_below.status_code == 200
    assert res_below.json()["lat"] == 0.5


def test_profile_out_of_region() -> None:
    """Verify Test 2: coordinates outside region return 422 out_of_region."""
    client = TestClient(app)
    # lat outside [0, 30]
    res_lat = client.get("/api/v1/profile?lat=-2.0&lon=50.0&date=2020-01-01")
    assert res_lat.status_code == 422
    assert res_lat.json()["error"]["code"] == "out_of_region"

    # lon outside [40, 100]
    res_lon = client.get("/api/v1/profile?lat=10.0&lon=110.0&date=2020-01-01")
    assert res_lon.status_code == 422
    assert res_lon.json()["error"]["code"] == "out_of_region"


def test_timeseries_happy_path_and_validation() -> None:
    """Verify Section 4.5: /timeseries returns valid daily series and enforces limits."""
    client = TestClient(app)
    res = client.get(
        "/api/v1/timeseries?lat=12.5&lon=65.5&depth_index=0&start=2020-01-01&end=2020-01-10"
    )
    assert res.status_code == 200
    assert res.headers.get("Cache-Control") == "public, max-age=86400"

    data = res.json()
    assert data["depth_m"] == 0.5
    assert len(data["dates"]) == 10
    assert len(data["pred"]) == 10
    assert len(data["glorys"]) == 10
    assert len(data["split"]) == 10
    assert data["dates"][0] == "2020-01-01"
    assert data["dates"][-1] == "2020-01-10"
    assert data["split"][0] == "val"
    assert data["split"][-1] == "live"

    # Timeseries exceeding 366 days -> 422 range_too_large
    res_large = client.get(
        "/api/v1/timeseries?lat=12.5&lon=65.5&depth_index=0&start=2020-01-01&end=2021-01-15"
    )
    assert res_large.status_code == 422
    assert res_large.json()["error"]["code"] == "range_too_large"


def test_region_summary_hand_computed_area_weighting() -> None:
    """Verify Test 5: /region-summary area weighting over 0-30°N domain."""
    client = TestClient(app)
    # Bbox spanning full 0-30°N domain where cos(lat) area weighting shifts mean from 27.05 to 27.11
    res = client.get(
        "/api/v1/region-summary?date=2020-01-01&depth_index=0&lat_min=0.0&lat_max=30.0&lon_min=40.0&lon_max=100.0&field=pred"
    )
    assert res.status_code == 200
    assert res.headers.get("Cache-Control") == "public, max-age=86400"

    data = res.json()
    assert data["date"] == "2020-01-01"
    assert data["depth_m"] == 0.5
    assert data["field"] == "pred"
    assert data["n_valid"] > 0
    # Area-weighted mean is 27.11 °C; unweighted mean would be 27.05 °C
    assert data["mean"] == 27.11
    assert data["mean"] != 27.05


def test_invalid_date_format_422() -> None:
    """Verify date params validate YYYY-MM-DD format and return 422 invalid_date_format."""
    client = TestClient(app)
    res_map = client.get("/api/v1/map?date=2020-1-5&depth_index=0&field=pred")
    assert res_map.status_code == 422
    assert res_map.json()["error"]["code"] == "invalid_date_format"

    res_ts = client.get(
        "/api/v1/timeseries?lat=12.5&lon=65.5&depth_index=0&start=2020-1-1&end=2020-01-10"
    )
    assert res_ts.status_code == 422
    assert res_ts.json()["error"]["code"] == "invalid_date_format"


def test_region_summary_predefined_regions() -> None:
    """Verify /region-summary accepts region=arabian_sea|bay_of_bengal|equatorial."""
    client = TestClient(app)
    for reg in ["arabian_sea", "bay_of_bengal", "equatorial"]:
        res = client.get(
            f"/api/v1/region-summary?date=2020-01-01&depth_index=0&region={reg}&field=pred"
        )
        assert res.status_code == 200
        data = res.json()
        assert data["n_valid"] > 0
        assert data["mean"] is not None


def test_map_performance_warm_response() -> None:
    """Verify Milestone B2 acceptance: /map warm p95 < 300 ms locally."""
    client = TestClient(app)
    # Warm up
    client.get("/api/v1/map?date=2020-01-01&depth_index=0&field=pred")

    timings: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = client.get("/api/v1/map?date=2020-01-01&depth_index=0&field=pred")
        t1 = time.perf_counter()
        assert res.status_code == 200
        timings.append((t1 - t0) * 1000.0)

    p95 = float(np.percentile(timings, 95))
    print(f"\nWarm /map p95 response time: {p95:.2f} ms")
    assert p95 < 300.0, f"Expected warm p95 < 300 ms, got {p95:.2f} ms"


def test_section_endpoint() -> None:
    """Verify /section satisfies PRD_BACKEND §4.11 and Test 13.

    Shapes are 15 × grid_shape.lon (for lat) and 15 × grid_shape.lat (for lon);
    values equal raw zarr values and match /map at the same depth along that row/col;
    nulls and coverage pass through; 422 for both/neither lat/lon; 404 reference_unavailable
    on live dates for glorys/error; warm p95 < 300 ms.
    """
    client = TestClient(app)

    # 1. East-West section along lat
    res_lat = client.get("/api/v1/section?date=2020-01-01&field=pred&lat=15.0")
    assert res_lat.status_code == 200
    assert res_lat.headers.get("Cache-Control") == "public, max-age=86400"
    assert "ETag" in res_lat.headers

    data_lat = res_lat.json()
    assert data_lat["run_id"] == "synthetic_v1"
    assert data_lat["synthetic"] is True
    assert data_lat["date"] == "2020-01-01"
    assert data_lat["field"] == "pred"
    assert data_lat["split"] == "val"
    assert data_lat["units"] == "degC"
    assert data_lat["axis"] == "lon"
    assert "lat" in data_lat["fixed"]
    snapped_lat = data_lat["fixed"]["lat"]

    # Shape: 15 depths x 60 lons
    assert len(data_lat["depths_m"]) == 15
    assert len(data_lat["positions"]) == 60
    assert len(data_lat["values"]) == 15
    assert len(data_lat["values"][0]) == 60
    assert len(data_lat["coverage"]) == 60

    # Consistency with /map at depth_index=0
    map_res = client.get("/api/v1/map?date=2020-01-01&depth_index=0&field=pred").json()
    lat_row_idx = map_res["lat"].index(snapped_lat)
    assert data_lat["values"][0] == map_res["values"][lat_row_idx]
    assert data_lat["coverage"] == map_res["coverage"][lat_row_idx]

    # 2. North-South section along lon
    res_lon = client.get("/api/v1/section?date=2020-01-01&field=pred&lon=65.0")
    assert res_lon.status_code == 200
    data_lon = res_lon.json()
    assert data_lon["axis"] == "lat"
    assert "lon" in data_lon["fixed"]
    snapped_lon = data_lon["fixed"]["lon"]

    # Shape: 15 depths x 30 lats
    assert len(data_lon["depths_m"]) == 15
    assert len(data_lon["positions"]) == 30
    assert len(data_lon["values"]) == 15
    assert len(data_lon["values"][0]) == 30
    assert len(data_lon["coverage"]) == 30

    # Consistency with /map at depth_index=0 along column
    lon_col_idx = map_res["lon"].index(snapped_lon)
    map_col_vals = [map_res["values"][r][lon_col_idx] for r in range(30)]
    map_col_cov = [map_res["coverage"][r][lon_col_idx] for r in range(30)]
    assert data_lon["values"][0] == map_col_vals
    assert data_lon["coverage"] == map_col_cov

    # 3. Validation: both lat and lon -> 422 invalid_section
    res_both = client.get("/api/v1/section?date=2020-01-01&lat=15.0&lon=65.0")
    assert res_both.status_code == 422
    assert res_both.json()["error"]["code"] == "invalid_section"

    # 4. Validation: neither lat nor lon -> 422 invalid_section
    res_neither = client.get("/api/v1/section?date=2020-01-01")
    assert res_neither.status_code == 422
    assert res_neither.json()["error"]["code"] == "invalid_section"

    # 5. Validation: out of region -> 422 out_of_region
    res_out_lat = client.get("/api/v1/section?date=2020-01-01&lat=35.0")
    assert res_out_lat.status_code == 422
    assert res_out_lat.json()["error"]["code"] == "out_of_region"

    res_out_lon = client.get("/api/v1/section?date=2020-01-01&lon=120.0")
    assert res_out_lon.status_code == 422
    assert res_out_lon.json()["error"]["code"] == "out_of_region"

    # 6. Live date reference unavailable: 2020-01-09 is live
    res_live_glorys = client.get("/api/v1/section?date=2020-01-09&field=glorys&lat=15.0")
    assert res_live_glorys.status_code == 404
    assert res_live_glorys.json()["error"]["code"] == "reference_unavailable"

    res_live_error = client.get("/api/v1/section?date=2020-01-09&field=error&lat=15.0")
    assert res_live_error.status_code == 404
    assert res_live_error.json()["error"]["code"] == "reference_unavailable"

    # Live date pred succeeds
    res_live_pred = client.get("/api/v1/section?date=2020-01-09&field=pred&lat=15.0")
    assert res_live_pred.status_code == 200
    assert res_live_pred.json()["split"] == "live"

    # 7. Date unavailable -> 404
    res_bad_date = client.get("/api/v1/section?date=1990-01-01&lat=15.0")
    assert res_bad_date.status_code == 404
    assert res_bad_date.json()["error"]["code"] == "date_unavailable"

    # 8. Benchmark warm p95 latency < 300 ms
    timings: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        r = client.get("/api/v1/section?date=2020-01-01&field=pred&lat=15.0")
        t1 = time.perf_counter()
        assert r.status_code == 200
        timings.append((t1 - t0) * 1000.0)

    p95 = float(np.percentile(timings, 95))
    print(f"\nWarm /section p95 response time: {p95:.2f} ms")
    assert p95 < 300.0, f"Expected warm p95 < 300 ms, got {p95:.2f} ms"
