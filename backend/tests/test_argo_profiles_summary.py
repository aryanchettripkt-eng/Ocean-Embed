"""/argo/profiles keeps only a per-profile summary of the matches file (Render memory limit)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.loader import DataLoader
from app.data.store import OutputStore

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "sample_outputs"
MATCHES = "metrics/argo_matches_synthetic_v1.parquet"


Row = tuple[str, int, str, float, float, int]


def _old_profiles(df: pd.DataFrame, date: str | None) -> list[Row]:
    """The previous implementation: filter every level, then group."""
    filtered = df
    if date is not None:
        filtered = filtered[filtered["time"].astype(str).str.startswith(date)]
    grouped = (
        filtered.groupby(["platform_number", "cycle", "time", "lat", "lon"], as_index=False)
        .size()
        .rename(columns={"size": "n_obs"})
    ).iloc[:500]
    return [
        (
            str(r["platform_number"]),
            int(r["cycle"]),
            str(r["time"]),
            round(float(r["lat"]), 4),
            round(float(r["lon"]), 4),
            int(r["n_obs"]),
        )
        for _, r in grouped.iterrows()
    ]


def _matches(n_profiles: int, levels: int) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    times = pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 40, n_profiles), "D")
    prof = pd.DataFrame(
        {
            "platform_number": [str(2902000 + i % 37) for i in range(n_profiles)],
            "cycle": rng.integers(1, 200, n_profiles),
            "time": times,
            "lat": rng.uniform(0, 30, n_profiles),
            "lon": rng.uniform(40, 100, n_profiles),
        }
    )
    rows = prof.loc[prof.index.repeat(rng.integers(1, levels, n_profiles))].reset_index(drop=True)
    rows = rows.sample(frac=1.0, random_state=1).reset_index(drop=True)  # unsorted on disk
    rows["depth_m"] = rng.uniform(0, 1000, len(rows))
    for col in ("temp_obs_c", "temp_pred_c", "temp_glorys_c", "temp_grep_c", "temp_armor3d_c"):
        rows[col] = pd.array(rng.normal(20, 5, len(rows)), dtype="Float64")
    rows.loc[3, "lat"] = np.nan  # a level with no position is not a marker
    return rows


def test_profiles_match_the_previous_grouping(tmp_path: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(FIXTURE, out)
    df = _matches(n_profiles=900, levels=60)
    df.to_parquet(out / MATCHES, index=False)
    store = OutputStore(loader=DataLoader(data_dir=out))

    for date in (None, "2018-01-05", "2018-01-20", "2019-01-01"):
        new = [
            (p.platform_number, p.cycle, p.time, p.lat, p.lon, p.n_obs)
            for p in store.get_argo_profiles(date=date).profiles
        ]
        assert new == _old_profiles(df, date), date

    boxed = store.get_argo_profiles(lat_min=10.0, lat_max=15.0, lon_min=80.0).profiles
    assert boxed
    assert all(10.0 <= p.lat <= 15.0 and p.lon >= 80.0 for p in boxed)


def test_only_the_profile_summary_is_cached(tmp_path: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(FIXTURE, out)
    df = _matches(n_profiles=300, levels=40)
    df.to_parquet(out / MATCHES, index=False)
    store = OutputStore(loader=DataLoader(data_dir=out))

    store.get_argo_profiles()
    cached = store._cached_argo_matches
    assert cached is not None
    assert (
        len(cached)
        == df.dropna(subset=["lat"])
        .drop_duplicates(["platform_number", "cycle", "time", "lat", "lon"])
        .shape[0]
    )
    assert "temp_obs_c" not in cached.columns
    assert "depth_m" not in cached.columns
