"""Dataset over the synthetic store: test-date refusal, shapes, NaN handling, stats source."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import zarr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.transform import TestLeakageError
from tests.synthetic_store import build_synthetic_store


def _set_root_attr(store: Path, key: str, value: object) -> None:
    zarr.open_group(str(store), mode="r+").attrs[key] = value


def _root_attr(store: Path, key: str) -> object:
    return zarr.open_group(str(store), mode="r").attrs[key]


def test_refuses_the_test_split(store: Path, config: ExperimentConfig) -> None:
    with pytest.raises(TestLeakageError, match="final_test"):
        OceanScopeDataset(store, config, split="test")


def test_refuses_a_validation_window_that_reaches_the_test_period(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    config.splits.val = ["2000-08-01", "2000-12-15"]  # 30-day gap leaves days in the test period
    leaky = build_synthetic_store(tmp_path / "leaky.zarr", config)
    with pytest.raises(TestLeakageError):
        OceanScopeDataset(leaky, config, split="val")


def test_train_split_size_after_history_and_gap(store: Path, config: ExperimentConfig) -> None:
    # 2000-03-01 to 2000-06-30, minus the last 30 days, leaves 2000-03-01 to 2000-05-31.
    ds = OceanScopeDataset(store, config, split="train")
    assert len(ds) == 92


def test_sample_is_finite_and_marks_land_as_invalid(store: Path, config: ExperimentConfig) -> None:
    ds = OceanScopeDataset(store, config, split="train")
    sample = ds[0]
    assert sample["x"].dtype == torch.float32
    assert torch.isfinite(sample["x"]).all()
    assert sample["mask"].dtype == torch.bool
    # The synthetic land corner is rows 0-1, columns 0-2.
    assert not sample["mask"][:, :2, :3].any()
    assert (sample["y"][:, :2, :3] == 0).all()


def test_grep_missing_days_have_has_grep_false(store: Path, config: ExperimentConfig) -> None:
    ds = OceanScopeDataset(store, config, split="train")
    assert not bool(ds[0]["has_grep"])  # day index 60 is a no-GREP day in the synthetic store
    assert bool(ds[1]["has_grep"])
    assert not ds[0]["grep_mask"].any()


def test_sigma_and_stats_come_from_store_attributes(store: Path, config: ExperimentConfig) -> None:
    stats = _root_attr(store, "norm_stats")
    ds = OceanScopeDataset(store, config, split="train")
    assert ds.sigma_d.tolist() == pytest.approx(stats["sigma_d"], rel=1e-6)


def test_refuses_sample_only_stores(store: Path, config: ExperimentConfig) -> None:
    _set_root_attr(store, "sample_only", True)
    with pytest.raises(ValueError, match="sample-only"):
        OceanScopeDataset(store, config, split="train")


def test_missing_channel_stats_are_refused(store: Path, config: ExperimentConfig) -> None:
    stats = dict(_root_attr(store, "norm_stats"))
    channels = {k: v for k, v in stats["channels"].items() if k != "sla"}
    _set_root_attr(store, "norm_stats", {**stats, "channels": channels})
    with pytest.raises(KeyError, match="sla"):
        OceanScopeDataset(store, config, split="train")


def test_reads_a_part_store_with_a_separate_stats_store(
    tmp_path: Path, store: Path, config: ExperimentConfig
) -> None:
    """A multi-part build keeps clim and norm_stats in a stats store; items must not change."""
    import pandas as pd
    import xarray as xr

    ds = xr.open_zarr(store)
    times = pd.to_datetime(ds["time"].values)
    stats = ds[["clim"]].copy()
    stats.attrs = {k: ds.attrs[k] for k in ("norm_stats", "split_dates")}
    stats.to_zarr(tmp_path / "stats.zarr", mode="w")

    part = ds.drop_vars("clim")
    part.attrs = {k: v for k, v in ds.attrs.items() if k not in ("norm_stats", "split_dates")}
    years = sorted(set(times.year))
    part.attrs["built_years"] = years
    part.attrs["built_ranges"] = {
        str(y): [str(times[times.year == y][0].date()), str(times[times.year == y][-1].date())]
        for y in years
    }
    part.to_zarr(tmp_path / "part.zarr", mode="w")

    single = OceanScopeDataset(store, config, split="train")
    parted = OceanScopeDataset(
        [tmp_path / "part.zarr"], config, split="train", stats_path=tmp_path / "stats.zarr"
    )
    assert len(parted) == len(single)
    for key, value in single[0].items():
        assert torch.equal(parted[0][key], value), key


def test_drops_days_whose_history_crosses_a_gap_between_parts(
    tmp_path: Path, store: Path, config: ExperimentConfig
) -> None:
    """Parts built for separate years leave gaps; no sample may read history across one."""
    import pandas as pd
    import xarray as xr

    ds = xr.open_zarr(store)
    times = pd.to_datetime(ds["time"].values)
    hole = pd.date_range("2000-04-01", "2000-04-10")
    gappy = ds.sel(time=times[~times.isin(hole)])
    gappy.to_zarr(tmp_path / "gappy.zarr", mode="w")

    full = OceanScopeDataset(store, config, split="train")
    parted = OceanScopeDataset(tmp_path / "gappy.zarr", config, split="train")
    lookback = pd.Timedelta(days=full._lookback)
    full_dates = times[full._positions]
    want = [d for d in full_dates if not ((hole >= d - lookback) & (hole <= d)).any()]
    got = pd.to_datetime(gappy["time"].values)[parted._positions]
    assert list(got) == want
    assert 0 < len(parted) < len(full)
