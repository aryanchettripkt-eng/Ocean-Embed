"""Export or verify fixture-derived API mock JSON files for frontend testing.
(PRD_FRONTEND §2, F1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app

MOCK_ENDPOINTS = {
    "health.json": "/api/v1/health",
    "meta.json": "/api/v1/meta",
    "map.json": "/api/v1/map?date=2020-01-05&depth_m=100.0&field=pred",
    "profile.json": "/api/v1/profile?lat=12.5&lon=65.5&date=2020-01-05",
    "timeseries.json": (
        "/api/v1/timeseries?lat=12.5&lon=65.5&depth_m=100.0&start=2020-01-01&end=2020-01-10"
    ),
    "region_summary.json": (
        "/api/v1/region-summary?date=2020-01-05&depth_m=100.0&lat_min=10.0&lat_max=12.0"
        "&lon_min=60.0&lon_max=62.0&field=pred"
    ),
    "metrics.json": "/api/v1/metrics?split=val&group_by=depth",
    "argo.json": "/api/v1/argo",
    "argo_profiles.json": "/api/v1/argo/profiles",
    "headline.json": "/api/v1/headline",
    "section.json": "/api/v1/section?date=2020-01-05&field=pred&lat=12.5",
}


FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"


def dump_mocks(client: TestClient) -> dict[str, Any]:
    """Execute API requests on fixtures and return dumped json payload dict."""
    from app.data.loader import loader
    from app.data.store import store

    if not loader.is_loaded:
        loader.load(data_dir=FIXTURES_DIR)
        store.clear_caches()

    mocks: dict[str, Any] = {}
    for filename, endpoint in MOCK_ENDPOINTS.items():
        res = client.get(endpoint)
        if res.status_code != 200:
            raise RuntimeError(
                f"Failed fetching {endpoint} for mock export: status {res.status_code}: {res.text}"
            )
        mocks[filename] = res.json()
    return mocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump backend fixture responses to frontend mock JSON files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that frontend mock JSON files are up to date.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../frontend/src/mocks"),
        help="Target directory for mock JSON files.",
    )
    args = parser.parse_args()

    client = TestClient(app)
    mocks = dump_mocks(client)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        has_drift = False
        for filename, data in mocks.items():
            target_path = out_dir / filename
            expected_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
            if not target_path.exists():
                print(f"Error: {target_path} does not exist. Run without --check to generate.")
                has_drift = True
                continue
            current_json = target_path.read_text(encoding="utf-8")
            if current_json != expected_json:
                print(f"Error: {target_path} is out of date. Run without --check to update.")
                has_drift = True
        if has_drift:
            sys.exit(1)
        print(f"All mock JSON files in {out_dir} are up to date.")
    else:
        for filename, data in mocks.items():
            target_path = out_dir / filename
            target_path.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"Dumped {target_path}")
        print("Mock export complete.")


if __name__ == "__main__":
    main()
