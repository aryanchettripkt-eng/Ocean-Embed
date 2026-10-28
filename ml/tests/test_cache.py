"""Training cache: same samples as reading the store, no test dates, rebuilt when stale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.cache import open_cache
from oceanscope_ml.data.dataset import OceanScopeDataset


@pytest.mark.parametrize("split", ["train", "val"])
def test_cached_samples_match_the_store(
    tmp_path: Path, store: Path, config: ExperimentConfig, split: str
) -> None:
    direct = OceanScopeDataset(store, config, split=split)
    cached = OceanScopeDataset(store, config, split=split, cache_dir=tmp_path / "cache")
    assert len(cached) == len(direct)
    for i in (0, len(direct) // 2, len(direct) - 1):
        a, b = direct[i], cached[i]
        for key in ("mask", "grep_mask", "has_grep"):
            assert torch.equal(a[key], b[key]), key
        for key in ("x", "y", "grep_y"):
            # float16 storage: about 3 significant digits
            torch.testing.assert_close(b[key], a[key], atol=2e-2, rtol=2e-3, msg=key)


def test_cache_holds_no_test_dates(tmp_path: Path, store: Path, config: ExperimentConfig) -> None:
    OceanScopeDataset(store, config, split="train", cache_dir=tmp_path / "cache")
    meta, arrays = open_cache(tmp_path / "cache")
    last = pd.Timestamp(meta["last"])
    assert last <= max(pd.Timestamp(config.splits.train[1]), pd.Timestamp(config.splits.val[1]))
    assert last < pd.Timestamp(config.splits.test[0])
    assert arrays["y"].shape[0] == meta["n_days"]


def test_cache_is_reused_then_rebuilt_when_stats_change(
    tmp_path: Path, store: Path, config: ExperimentConfig
) -> None:
    cache = tmp_path / "cache"
    OceanScopeDataset(store, config, split="train", cache_dir=cache)
    stamp = (cache / "y.npy").stat().st_mtime_ns
    OceanScopeDataset(store, config, split="val", cache_dir=cache)
    assert (cache / "y.npy").stat().st_mtime_ns == stamp  # reused

    root = zarr.open_group(str(store), mode="r+")
    stats = dict(root.attrs["norm_stats"])
    stats["sigma_d"] = [2.0 * s for s in stats["sigma_d"]]
    root.attrs["norm_stats"] = stats
    OceanScopeDataset(store, config, split="train", cache_dir=cache)
    meta = json.loads((cache / "meta.json").read_text())
    assert (cache / "y.npy").stat().st_mtime_ns != stamp
    assert meta["n_days"] == np.load(cache / "y.npy", mmap_mode="r").shape[0]
