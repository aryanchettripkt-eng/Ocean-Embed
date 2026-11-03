"""Live mode (PRD_AI_ML §12, PHASES P9-1, P9-3) on the synthetic store, with fake inputs.

The real fetchers need network and credentials, so ``reduce_product`` is replaced by a fake
that returns ocean fields with chosen days missing. Everything else runs for real: the kit,
the model, the skeleton download, the zarr append and trim, and both commits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.evaluation import live_check
from oceanscope_ml.evaluation.evaluate import evaluate_predictions, write_manifest
from oceanscope_ml.inference.predict import predict_store
from oceanscope_ml.live import inputs as live_inputs
from oceanscope_ml.live.kit import Kit, build_kit, open_kit
from oceanscope_ml.live.run import LiveRunError, LiveSettings, load_live_config, run_live
from oceanscope_ml.live.store import LocalRepo
from oceanscope_ml.train import train_run

STORE_REL = "predictions/predictions_run1.zarr"


@pytest.fixture
def live_config(config: ExperimentConfig) -> ExperimentConfig:
    # The synthetic store's 9 x 11 grid (tests/synthetic_store.py).
    config.region.lat_min, config.region.lat_max = 5.0, 7.25
    config.region.lon_min, config.region.lon_max = 60.0, 62.75
    return config


@pytest.fixture
def published(tmp_path: Path, live_config: ExperimentConfig, store: Path) -> tuple[Kit, Path]:
    """A trained run, its kit, and a published outputs folder acting as the dataset repo."""
    train_run(live_config, store, "run1", device_choice="cpu", epochs=1)
    ckpt = Path(live_config.paths.checkpoints) / "run1" / "best.pt"
    pred = predict_store(live_config, store, ckpt, device_choice="cpu")
    write_manifest(live_config, pred, evaluate_predictions(live_config, pred, store))
    kit = build_kit(live_config, ckpt, store, store, tmp_path / "kit")
    return open_kit(kit.root, live_config), Path(live_config.paths.outputs)


class RecordingRepo(LocalRepo):
    def __init__(self, root: Path, fail_manifest: bool = False) -> None:
        super().__init__(root)
        self.commits: list[tuple[list[str], list[str]]] = []
        self.fail_manifest = fail_manifest

    def commit(self, local_dir: Path, add: list[str], delete: list[str], message: str) -> None:
        if self.fail_manifest and add == ["manifest.json"]:
            raise RuntimeError("network down")
        self.commits.append((list(add), list(delete)))
        super().commit(local_dir, add, delete, message)


def fake_inputs(
    monkeypatch: pytest.MonkeyPatch,
    missing: dict[str, list[str]] | None = None,
    only_nrt: bool = False,
) -> None:
    """Inputs present every day except ``missing[var]`` days (NRT sources only if asked)."""

    def reduce(product: str, raw: Any, grid: Any, days: pd.DatetimeIndex, ocean: Any) -> Any:
        is_nrt = "nrt" in raw.attrs["dataset_id"].lower()
        gaps = (missing or {}) if (is_nrt or not only_nrt) else {}
        rng = np.random.default_rng(len(product))
        out = {}
        for var in live_inputs.PRODUCT_VARS[product]:
            v = rng.normal(0.5, 1.0, (len(days), len(grid.lats), len(grid.lons)))
            v = np.where(ocean[None], v, np.nan).astype(np.float32)
            for d in gaps.get(var, []):
                if pd.Timestamp(d) in days:
                    v[days.get_loc(pd.Timestamp(d))] = np.nan
            out[var] = v
        return out

    monkeypatch.setattr(live_inputs, "reduce_product", reduce)


def fetch(product: str, spec: dict[str, Any], start: Any, end: Any) -> xr.Dataset:
    return xr.Dataset(attrs={"dataset_id": spec["dataset_id"]})


LIVE_CFG = load_live_config(Path(__file__).resolve().parents[1] / "configs" / "live.yaml")


def settings(config: ExperimentConfig, **kw: Any) -> LiveSettings:
    return LiveSettings(sources=LIVE_CFG["live"]["sources"], **kw)


def _run(config: ExperimentConfig, repo: Any, kit: Kit, tmp: Path, today: str, **kw: Any) -> Any:
    s = kw.pop("settings", None) or settings(config)
    return run_live(config, repo, kit, tmp / f"work-{today}", pd.Timestamp(today), fetch, s, **kw)


def _open(root: Path) -> xr.Dataset:
    return xr.open_zarr(root / STORE_REL, consolidated=True)


def test_run_appends_one_live_day_then_manifest(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    fake_inputs(monkeypatch)
    val_before = _open(root)["temp_pred"].values.copy()
    repo = RecordingRepo(root)

    log = _run(live_config, repo, kit, tmp_path, "2001-03-15")

    assert log.status == "published" and log.target_day == "2001-03-14"
    # Data first, manifest last, each in its own commit.
    assert len(repo.commits) == 2 and repo.commits[1] == (["manifest.json"], [])
    assert "manifest.json" not in repo.commits[0][0]
    assert all(p.startswith(STORE_REL) for p in repo.commits[0][0])
    ds = _open(root)
    assert pd.Timestamp(ds["time"].values[-1]) == pd.Timestamp("2001-03-14")
    np.testing.assert_array_equal(ds["temp_pred"].values[:-1], val_before)  # val days untouched
    assert np.isnan(ds["temp_glorys"].values[-1].astype(np.float32)).all()
    cov = ds["coverage"].values[-1]
    temp = ds["temp_pred"].values[-1].astype(np.float32)
    assert (cov == 1).any() and np.isfinite(temp[:, cov == 1]).all()
    assert json.loads(ds.attrs["split_of_each_date"])["2001-03-14/2001-03-14"] == "live"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["available_dates"]["end"] == "2001-03-14"
    assert set(log.input_latency_days.values()) == {1}

    again = _run(live_config, repo, kit, tmp_path / "again", "2001-03-15")
    assert again.status == "up_to_date" and len(repo.commits) == 2


def test_missing_newest_input_skips_to_the_last_complete_day(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    fake_inputs(monkeypatch, {"sst": ["2001-03-14"], "sla": ["2001-03-13"]})
    log = _run(live_config, RecordingRepo(root), kit, tmp_path, "2001-03-15")
    # 03-14 lacks sst on the day (and sla in its history); 03-13 lacks sla; 03-12 is complete.
    assert log.status == "published" and log.target_day == "2001-03-12"
    assert log.skipped_days["2001-03-14"] == [
        "sla missing 1 history day(s)",
        "sst missing on the day",
    ]
    assert log.skipped_days["2001-03-13"] == ["sla missing on the day"]
    assert log.input_last_day["sst"] == "2001-03-13"


def test_no_complete_day_fails_without_uploading(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    days = pd.date_range("2000-12-01", "2001-03-14").strftime("%Y-%m-%d").tolist()
    fake_inputs(monkeypatch, {"u_wind": days})
    repo = RecordingRepo(root)
    with pytest.raises(LiveRunError, match="has every input"):
        _run(live_config, repo, kit, tmp_path, "2001-03-15")
    assert repo.commits == []
    assert json.loads((root / "manifest.json").read_text())["available_dates"]["end"] < "2001"


def test_stale_kit_is_refused(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    fake_inputs(monkeypatch)
    with pytest.raises(LiveRunError, match="live kit is for"):
        _run(live_config, RecordingRepo(root), Kit(kit.root, "other"), tmp_path, "2001-03-15")


def test_lost_manifest_commit_is_repaired_next_run(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    fake_inputs(monkeypatch)
    with pytest.raises(RuntimeError, match="network down"):
        _run(live_config, RecordingRepo(root, fail_manifest=True), kit, tmp_path, "2001-03-15")
    assert json.loads((root / "manifest.json").read_text())["available_dates"]["end"] < "2001"
    log = _run(live_config, RecordingRepo(root), kit, tmp_path / "next", "2001-03-15")
    assert log.status == "up_to_date" and any("repaired" in n for n in log.notes)
    assert json.loads((root / "manifest.json").read_text())["available_dates"]["end"] == (
        "2001-03-14"
    )


def test_rolling_window_trims_oldest_live_days(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    fake_inputs(monkeypatch)
    n_val = int(_open(root).sizes["time"])
    val_before = _open(root)["temp_pred"].values.copy()
    s = settings(live_config, window_days=2, trim_slack_days=1)
    for today in ("2001-03-15", "2001-03-16", "2001-03-17", "2001-03-18"):
        log = _run(live_config, RecordingRepo(root), kit, tmp_path, today, settings=s)
    assert log.trimmed_days == 2
    ds = _open(root)
    times = pd.DatetimeIndex(ds["time"].values)
    assert list(times[n_val:].strftime("%Y-%m-%d")) == ["2001-03-16", "2001-03-17"]
    np.testing.assert_array_equal(ds["temp_pred"].values[:n_val], val_before)
    assert np.isfinite(ds["temp_pred"].values[n_val:].astype(np.float32)).any(axis=(1, 2, 3)).all()
    stale = [
        p for p in (root / STORE_REL / "temp_pred").iterdir() if p.name.startswith(f"{n_val + 2}.")
    ]
    assert stale == []  # chunks past the new end were deleted in the repo
    splits = json.loads(ds.attrs["split_of_each_date"])
    assert {k for k, v in splits.items() if v == "live"} == {"2001-03-16/2001-03-17"}
    assert json.loads((root / "manifest.json").read_text())["available_dates"]["end"] == (
        "2001-03-17"
    )


def test_dry_run_uploads_nothing(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, root = published
    fake_inputs(monkeypatch)
    repo = RecordingRepo(root)
    log = _run(live_config, repo, kit, tmp_path, "2001-03-15", dry_run=True)
    assert log.status == "dry_run" and log.uploaded_files > 0 and repo.commits == []


def test_live_split_refuses_test_dates(live_config: ExperimentConfig, store: Path) -> None:
    with pytest.raises(TestLeakageError, match="sealed test"):
        OceanScopeDataset(store, live_config, split="live", live_dates=[pd.Timestamp("2000-11-15")])
    with pytest.raises(ValueError, match="needs live_dates"):
        OceanScopeDataset(store, live_config, split="live")


def test_check_inputs_requires_the_whole_history() -> None:
    days = pd.date_range("2001-01-01", "2001-01-10")
    cov = pd.DataFrame({"sst": 1.0, "sla": 1.0}, index=days)
    cov.loc["2001-01-05", "sla"] = 0.2
    check = live_inputs.check_inputs(cov, {"sst": 0, "sla": 3}, days[-1], days[0])
    assert check.target == pd.Timestamp("2001-01-10")
    check = live_inputs.check_inputs(cov, {"sst": 0, "sla": 6}, days[-1], days[0])
    # 01-10 needs sla 01-04..10, which holds the bad 01-05; earlier days lack history.
    assert check.target is None and "2001-01-10" in check.skipped
    assert check.last_day == {"sst": "2001-01-10", "sla": "2001-01-10"}


def test_cds_requests_split_by_month() -> None:
    reqs = live_inputs.cds_requests(
        pd.Timestamp("2001-01-30"), pd.Timestamp("2001-02-02"), (60.0, 62.0, 5.0, 7.0)
    )
    assert [(r["month"], r["day"]) for r in reqs] == [("01", ["30", "31"]), ("02", ["01", "02"])]
    assert reqs[0]["daily_statistic"] == "daily_mean" and reqs[0]["area"] == [8.0, 59.0, 4.0, 63.0]


def test_nrt_check_refuses_test_month(live_config: ExperimentConfig) -> None:
    with pytest.raises(TestLeakageError):
        live_check.month_days("2000-11", live_config)


def test_nrt_check_writes_report(
    tmp_path: Path, live_config: ExperimentConfig, published: tuple[Kit, Path], monkeypatch: Any
) -> None:
    kit, _ = published
    fake_inputs(monkeypatch, {"sst": ["2001-01-20"]}, only_nrt=True)
    depths = len(live_config.depths_m)

    def glorys(days: pd.DatetimeIndex) -> np.ndarray:
        return np.full((len(days), depths, 9, 11), 12.0, dtype=np.float32)

    paths = live_check.run_check(
        live_config,
        kit,
        "2001-01",
        fetch,
        glorys,
        tmp_path / "out",
        tmp_path / "w",
        "fake",
        LIVE_CFG,
    )
    table = pd.read_csv(paths["csv"])
    assert len(table) == depths and (table["n"] > 0).all()
    assert np.allclose(table["rmse_diff_c"], table["rmse_nrt_c"] - table["rmse_reprocessed_c"])
    text = paths["report"].read_text()
    # sst is read at t-0..t-6, so a gap on 01-20 removes 01-20..01-26 from the NRT set.
    assert "NRT 24/31" in text and "reprocessed 31/31" in text
    assert "Decision (owner)" in text


def test_live_settings_live_outside_the_hashed_experiment_config() -> None:
    # train.py refuses to resume a run whose experiment.yaml hash changed, so live settings
    # must not be in it.
    exp = (Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml").read_text()
    assert "\nlive:" not in exp and "live_check:" not in exp
    s = LiveSettings.from_dict(LIVE_CFG)
    assert set(s.sources) == set(live_inputs.PRODUCT_VARS)
    assert set(LIVE_CFG["live_check"]["reprocessed"]) == set(live_inputs.PRODUCT_VARS)


def test_kit_for_a_run_published_under_another_name(
    tmp_path: Path, live_config: ExperimentConfig, store: Path, monkeypatch: Any
) -> None:
    # A fine-tuned run: best_ft.pt holds "main2_ft_l0.3" but it is published as "main2_ft".
    train_run(live_config, store, "main2_ft_l0.3", device_choice="cpu", epochs=1)
    ckpt = Path(live_config.paths.checkpoints) / "main2_ft_l0.3" / "best.pt"
    pred = predict_store(live_config, store, ckpt, device_choice="cpu", run_id="main2_ft")
    write_manifest(live_config, pred, evaluate_predictions(live_config, pred, store))
    root = Path(live_config.paths.outputs)
    fake_inputs(monkeypatch)

    wrong = open_kit(build_kit(live_config, ckpt, store, store, tmp_path / "k1").root, live_config)
    with pytest.raises(LiveRunError, match="live kit is for main2_ft_l0.3"):
        _run(live_config, RecordingRepo(root), wrong, tmp_path / "a", "2001-03-15")

    built = build_kit(live_config, ckpt, store, store, tmp_path / "k2", run_id="main2_ft")
    kit = open_kit(built.root, live_config)
    assert (kit.run_id, kit.checkpoint_run_id) == ("main2_ft", "main2_ft_l0.3")
    log = _run(live_config, RecordingRepo(root), kit, tmp_path / "b", "2001-03-15")
    assert log.status == "published" and log.run_id == "main2_ft"
