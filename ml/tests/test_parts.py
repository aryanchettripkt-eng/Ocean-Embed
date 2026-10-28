"""Tests for smoke builds and multi-part stores (PRD_AI_ML §4.2).

The full build runs as several part stores on Kaggle, after a short smoke build that
writes two weeks of each risky year. These tests cover the whole smoke path offline:
build parts with --smoke-days, fit stats across parts, run the check, run baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from oceanscope_ml.data.build_store import build_store
from oceanscope_ml.data.parts import build_stats, check_parts, open_parts, print_check
from oceanscope_ml.models.baselines import run_baselines
from tests.stream_mocks import install_stream_mocks
from tests.test_build_store import MockConfig


def _build_part(
    tmp_path: Path, cfg: MockConfig, name: str, years: list[int], smoke_days: int
) -> Path:
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir(exist_ok=True)
    return build_store(
        config=cfg,
        years=years,
        raw_dir=empty_raw,
        out_dir=tmp_path / name,
        scratch_dir=tmp_path / f"scratch_{name}",
        stream=True,
        max_days=smoke_days,
        part=True,
    )


@pytest.fixture
def smoke_cfg(tmp_path: Path) -> MockConfig:
    cfg = MockConfig(tmp_path / "raw", tmp_path / "processed", tmp_path / "outputs")
    cfg.splits.train = ["2018-01-01", "2018-12-31"]
    cfg.splits.val = ["2019-01-01", "2019-12-31"]
    cfg.train = {"seed": 42}  # type: ignore[attr-defined]
    return cfg


def test_smoke_days_limits_every_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_cfg: MockConfig
) -> None:
    """--smoke-days asks every remote product for the first N days only."""
    calls = install_stream_mocks(
        monkeypatch, smoke_cfg.depths_m, pd.date_range("2019-01-01", "2019-03-31")
    )
    store = _build_part(tmp_path, smoke_cfg, "a", [2019], smoke_days=14)

    assert {(start, end) for _, start, end in calls["copernicus"]} == {("2019-01-01", "2019-01-14")}
    assert len(calls["copernicus"]) == 6  # GLORYS, SLA, SST, SSS, GREP, ARMOR3D
    assert calls["era5_last_month"] == [1]
    assert calls["oscar_end"] == ["2019-01-14"]

    root = zarr.open_group(str(store), mode="r")
    assert root.attrs["built_ranges"] == {"2019": ["2019-01-01", "2019-01-14"]}
    assert root.attrs["smoke_days"] == 14
    ds = open_parts([store])
    assert len(ds.time) == 14
    assert bool(np.isfinite(ds["thetao"].isel(depth=0)).all())


def test_parts_stats_check_and_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_cfg: MockConfig
) -> None:
    """Two smoke parts -> stats across parts -> check is clean -> baselines run."""
    install_stream_mocks(monkeypatch, smoke_cfg.depths_m, pd.date_range("2018-01-01", "2019-01-31"))
    part_a = _build_part(tmp_path, smoke_cfg, "a", [2018], smoke_days=5)
    part_b = _build_part(tmp_path, smoke_cfg, "b", [2019], smoke_days=5)
    for part in (part_a, part_b):
        assert "clim" not in xr.open_zarr(part).data_vars  # parts are not finalized

    stats = build_stats(smoke_cfg, [part_a, part_b], tmp_path / "stats.zarr")
    ds_stats = xr.open_zarr(stats)
    assert ds_stats.attrs["train_years"] == [2018]
    assert ds_stats.attrs["n_train_dates"] == 5
    assert ds_stats.attrs["clim_harmonics"] == 0  # 5 days cannot fit an annual cycle
    assert ds_stats.attrs["built_years"] == [2018, 2019]
    assert "woa_clim_profile" in ds_stats.attrs["norm_stats"]["channels"]
    np.testing.assert_allclose(ds_stats["clim"].isel(dayofyear=200, depth=0).values, 24.0)

    ds = open_parts([part_a, part_b], stats)
    assert len(ds.time) == 10
    assert "clim" in ds and "norm_stats" in ds.attrs

    rows = check_parts([part_a, part_b], stats, expected_years=[2018, 2019])
    assert print_check(rows) == 0, rows
    products = {r["item"] for r in rows}
    assert {"GLORYS", "SLA", "SST", "SSS", "OSCAR", "ERA5", "GREP", "ARMOR3D"} <= products
    assert {"dates", "climatology", "norm stats"} <= products

    df_clim, df_ridge = run_baselines(
        config=smoke_cfg,  # type: ignore[arg-type]
        store_path=[part_a, part_b],
        stats_path=stats,
        out_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
        max_train_samples=2000,
    )
    assert int(df_clim["n"].iloc[0]) > 0
    assert (df_clim["n"] == df_ridge["n"]).all()


def test_check_flags_missing_product_and_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_cfg: MockConfig
) -> None:
    """An empty product or an unbuilt year must show up as FAIL, not pass silently."""
    install_stream_mocks(monkeypatch, smoke_cfg.depths_m, pd.date_range("2019-01-01", "2019-01-31"))
    part = _build_part(tmp_path, smoke_cfg, "a", [2019], smoke_days=5)
    root = zarr.open_group(str(part), mode="r+")
    arr: Any = root["grep_mean"]
    arr[:] = np.nan

    rows = check_parts([part], expected_years=[2019, 2020])
    fails = {(r["item"], r["year"]) for r in rows if r["status"] == "FAIL"}
    assert fails == {("GREP", 2019), ("year", 2020)}


def test_open_parts_rejects_overlapping_years(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_cfg: MockConfig
) -> None:
    install_stream_mocks(monkeypatch, smoke_cfg.depths_m, pd.date_range("2019-01-01", "2019-01-31"))
    part_a = _build_part(tmp_path, smoke_cfg, "a", [2019], smoke_days=3)
    part_b = _build_part(tmp_path, smoke_cfg, "b", [2019], smoke_days=3)
    with pytest.raises(ValueError, match="Year 2019 is built in both"):
        open_parts([part_a, part_b])


def test_max_hours_stops_cleanly_and_resume_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_cfg: MockConfig
) -> None:
    """A run that hits --max-hours keeps what it built; --resume-from finishes the rest."""
    install_stream_mocks(monkeypatch, smoke_cfg.depths_m, pd.date_range("2018-01-01", "2019-01-31"))
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    first = build_store(
        config=smoke_cfg,
        years=[2018, 2019],
        raw_dir=empty_raw,
        out_dir=tmp_path / "run1",
        scratch_dir=tmp_path / "s1",
        stream=True,
        max_days=3,
        part=True,
        max_hours=0.0,  # builds the first year, then stops before the second
    )
    assert zarr.open_group(str(first), mode="r").attrs["built_years"] == [2018]

    first.chmod(0o555)  # Kaggle inputs are read-only
    second = build_store(
        config=smoke_cfg,
        years=[2018, 2019],
        raw_dir=empty_raw,
        out_dir=tmp_path / "run2",
        scratch_dir=tmp_path / "s2",
        stream=True,
        max_days=3,
        part=True,
        resume_from=first,
    )
    first.chmod(0o755)
    assert zarr.open_group(str(second), mode="r").attrs["built_years"] == [2018, 2019]
    assert len(open_parts([second]).time) == 6
