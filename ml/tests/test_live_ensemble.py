"""Live mode with an ensemble kit and ARGO input members (PRD_AI_ML §5.1, §7, §12).

The ARGO member is the trained run widened to the ARGO channel list (``expand_input_weights``):
with zero weights on the new channels it predicts exactly what the base run predicts, so an
ensemble of the two must match the single run. ERDDAP is replaced by a fake opener.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data import argo
from oceanscope_ml.data.argo_inputs import enable_argo_input, expand_input_weights
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.live import argo as live_argo
from oceanscope_ml.live import inputs as live_inputs
from oceanscope_ml.live.kit import Kit, build_kit, open_kit, predict_days
from oceanscope_ml.live.run import LiveRunError, run_live
from tests.test_live import (  # noqa: F401  (fixtures)
    RecordingRepo,
    fake_inputs,
    fetch,
    live_config,
    published,
    settings,
)

HELD_OUT = "9999999"


def _argo_checkpoint(config: ExperimentConfig, base: Path, out: Path, scale: float) -> Path:
    """``base`` widened to the ARGO channels; ``scale`` sets the new channels' weights."""
    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(config)
    enable_argo_input(cfg)
    names = channel_names(cfg)
    state = expand_input_weights(ckpt["model"], ckpt["channel_names"], names)
    if scale:
        w = state["encoders.0.0.weight"]
        for i, n in enumerate(names):
            if n.startswith("argo_"):
                w[:, i] = scale
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ckpt, "model": state, "channel_names": names, "run_id": out.stem}, out)
    return out


def _holdout(tmp: Path) -> Path:
    path = tmp / "argo_eval_floats.json"
    path.write_text(json.dumps({"held_out_floats": [HELD_OUT]}), encoding="utf-8")
    return path


def _profiles(config: ExperimentConfig, days: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for k, day in enumerate(days):
        rows.append(
            {
                "platform_number": str(3900000 + k % 3),
                "cycle": k,
                "day": day,
                "lat": 5.5 + 0.25 * (k % 5),
                "lon": 60.5 + 0.25 * (k % 7),
                **{f"t_{d}": 30.0 - 0.02 * dep for d, dep in enumerate(config.depths_m)},
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def kits(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    store: Path,
    published: tuple[Kit, Path],  # noqa: F811
) -> dict[str, Any]:
    single, root = published
    base = Path(live_config.paths.checkpoints) / "run1" / "best.pt"
    zero = _argo_checkpoint(live_config, base, tmp_path / "ck" / "run1_ai0.pt", 0.0)
    loud = _argo_checkpoint(live_config, base, tmp_path / "ck" / "run1_ai.pt", 0.5)
    holdout = _holdout(tmp_path)

    def build(members: dict[str, Path], name: str) -> Kit:
        built = build_kit(
            live_config, members, store, store, tmp_path / name, run_id="run1", holdout=holdout
        )
        return open_kit(built.root, live_config)

    return {
        "single": single,
        "root": root,
        "same": build({"a": base, "b": zero}, "k_same"),
        "argo": build({"a": base, "b": loud}, "k_argo"),
    }


def _window(config: ExperimentConfig, kit: Kit, tmp: Path, monkeypatch: Any) -> tuple[Path, Any]:
    fake_inputs(monkeypatch)
    static = kit.static()
    ocean = static["ocean_mask"].values.astype(bool)
    max_lb = max(live_inputs.var_lookbacks(config).values())
    target = pd.Timestamp("2001-03-14")
    days = pd.date_range(target - pd.Timedelta(days=max_lb), target, freq="D")
    srcs = settings(config).sources
    inputs = live_inputs.load_inputs(config, srcs, days, ocean, fetch)
    window = live_inputs.write_window_store(tmp / "w.zarr", inputs, static, days[0], target)
    return window, target


def test_ensemble_kit_layout_and_mean(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    kits: dict[str, Any],
    monkeypatch: Any,
) -> None:
    same: Kit = kits["same"]
    meta = json.loads((same.root / "kit.json").read_text(encoding="utf-8"))
    assert [m["name"] for m in meta["members"]] == ["a", "b"]
    assert (same.root / "members" / "b.pt").exists() and not (same.root / "best.pt").exists()
    assert same.needs_argo and same.holdout() == {HELD_OUT}
    assert not kits["single"].needs_argo

    window, target = _window(live_config, same, tmp_path, monkeypatch)
    profiles = _profiles(live_config, pd.date_range(target - pd.Timedelta(days=9), target))
    one = predict_days(live_config, kits["single"], window, [target])
    two = predict_days(live_config, same, window, [target], argo_profiles=profiles)
    np.testing.assert_allclose(two["temp_pred"].values, one["temp_pred"].values, atol=1e-4)
    assert (two["coverage"].values == one["coverage"].values).all()
    assert channel_names(live_config)[-1] != "argo_age"  # the caller's config is untouched

    loud: Kit = kits["argo"]
    with_argo = predict_days(live_config, loud, window, [target], argo_profiles=profiles)
    empty = predict_days(live_config, loud, window, [target], argo_profiles=profiles.iloc[:0])
    assert not np.allclose(
        np.nan_to_num(with_argo["temp_pred"].values), np.nan_to_num(empty["temp_pred"].values)
    )
    with pytest.raises(ValueError, match="ARGO input channels"):
        predict_days(live_config, loud, window, [target])
    held = profiles.assign(platform_number=HELD_OUT)
    with pytest.raises(AssertionError, match="held-out"):
        predict_days(live_config, loud, window, [target], argo_profiles=held)


def test_kit_refusals(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    store: Path,
    kits: dict[str, Any],
) -> None:
    base = Path(live_config.paths.checkpoints) / "run1" / "best.pt"
    ai = tmp_path / "ck" / "run1_ai.pt"
    with pytest.raises(ValueError, match="held-out float list"):
        build_kit(live_config, {"a": base, "b": ai}, store, store, tmp_path / "x", run_id="e")
    with pytest.raises(ValueError, match="published name"):
        build_kit(live_config, {"a": base, "b": base}, store, store, tmp_path / "y")
    (kits["argo"].root / "argo_eval_floats.json").unlink()
    with pytest.raises(FileNotFoundError, match="argo_eval_floats.json"):
        open_kit(kits["argo"].root, live_config)
    # A kit.json from before ensembles still opens as one member.
    path = kits["single"].root / "kit.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({k: v for k, v in meta.items() if k != "members"}), "utf-8")
    legacy = open_kit(kits["single"].root, live_config)
    assert [m.file for m in legacy.members] == ["best.pt"] and not legacy.needs_argo


def test_run_fetches_the_argo_window_for_argo_members(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    kits: dict[str, Any],
    monkeypatch: Any,
) -> None:
    fake_inputs(monkeypatch)
    kit: Kit = kits["argo"]
    root = kits["root"]
    asked: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def argo_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        asked.append((start, end))
        return _profiles(live_config, pd.date_range(start, end))

    def go(name: str, **kw: Any) -> Any:
        return run_live(
            live_config,
            RecordingRepo(root),
            kit,
            tmp_path / name,
            pd.Timestamp("2001-03-15"),
            fetch,
            settings(live_config),
            dry_run=True,
            **kw,
        )

    with pytest.raises(LiveRunError, match="no ARGO fetcher"):
        go("a")
    with pytest.raises(LiveRunError, match="No ARGO profiles"):
        go("b", argo_fetch=lambda s, e: _profiles(live_config, pd.DatetimeIndex([])).iloc[:0])
    log = go("c", argo_fetch=argo_fetch)
    assert log.status == "dry_run" and log.members == ["a", "b"]
    assert asked == [(pd.Timestamp("2001-03-05"), pd.Timestamp("2001-03-14"))]
    assert log.argo_profiles == 10 and log.argo_floats == 3


def _erddap_csv(rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(rows, columns=list(argo.VARIABLES))
    header, body = df.to_csv(index=False).split("\n", 1)
    return header + "\n" + ",".join("" for _ in argo.VARIABLES) + "\n" + body


def test_live_argo_profiles_from_erddap(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
) -> None:
    rows = []
    for k, (platform, day) in enumerate(
        [("3900001", "2001-03-10"), (HELD_OUT, "2001-03-11"), ("3900002", "2001-03-12")]
    ):
        for pres in (5.0, 100.0, 1000.0):
            rows.append(
                {
                    "platform_number": platform,
                    "cycle_number": k,
                    "time": f"{day}T06:00:00Z",
                    "latitude": 6.0,
                    "longitude": 61.0,
                    "pres": pres,
                    "temp": 28.0 - pres / 100.0,
                    "temp_qc": 1,
                    "pres_qc": 1,
                    "position_qc": 1,
                }
            )
    urls: list[str] = []

    def opener(url: str) -> str:
        urls.append(url)
        return _erddap_csv(rows)

    start, end = pd.Timestamp("2001-03-05"), pd.Timestamp("2001-03-14")
    prof = live_argo.live_profiles(start, end, live_config, tmp_path, {HELD_OUT}, opener)
    assert sorted(prof["platform_number"]) == ["3900001", "3900002"]
    assert "time%3E=2001-03-05T00%3A00%3A00Z" in urls[0]
    # The day's file is kept: a second call reads it without asking ERDDAP again.
    live_argo.live_profiles(start, end, live_config, tmp_path, {HELD_OUT}, opener)
    assert len(urls) == 1
    test_start = pd.Timestamp(live_config.splits.test[0])
    with pytest.raises(ValueError, match="sealed test period"):
        live_argo.fetch_range(test_start - pd.Timedelta(days=3), test_start, live_config, tmp_path)
    with pytest.raises(ValueError, match="held-out float list"):
        live_argo.live_profiles(start, end, live_config, tmp_path, set(), opener)


def test_nrt_check_runs_the_ensemble_with_argo(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    kits: dict[str, Any],
    monkeypatch: Any,
) -> None:
    from oceanscope_ml.evaluation import live_check
    from tests.test_live import LIVE_CFG

    fake_inputs(monkeypatch)
    asked: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def argo_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        asked.append((start, end))
        return _profiles(live_config, pd.date_range(start, end))

    def glorys(days: pd.DatetimeIndex) -> np.ndarray:
        return np.full((len(days), len(live_config.depths_m), 9, 11), 12.0, dtype=np.float32)

    args = (live_config, kits["argo"], "2001-01", fetch, glorys, tmp_path / "out", tmp_path / "w")
    with pytest.raises(ValueError, match="no ARGO fetcher"):
        live_check.run_check(*args, "fake", LIVE_CFG)
    paths = live_check.run_check(*args, "fake", LIVE_CFG, argo_fetch=argo_fetch)
    assert asked == [(pd.Timestamp("2000-12-23"), pd.Timestamp("2001-01-31"))]
    text = paths["report"].read_text()
    assert "(mean of a, b)" in text and "ARGO input profiles (same in both sets): 40" in text


def test_a_wider_argo_window_member_widens_the_live_fetch(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    store: Path,
    kits: dict[str, Any],
    monkeypatch: Any,
) -> None:
    # A member trained with a 30-day ARGO window (main3_aix30a) keeps it in the kit, and the
    # live run fetches 30 days instead of the config's 10.
    loud = tmp_path / "ck" / "run1_ai.pt"
    wide = tmp_path / "ck" / "run1_ai30.pt"
    ckpt = torch.load(loud, map_location="cpu", weights_only=False)
    torch.save({**ckpt, "argo_window_days": 30, "run_id": "run1_ai30"}, wide)
    base = Path(live_config.paths.checkpoints) / "run1" / "best.pt"
    built = build_kit(live_config, {"a": base, "b": wide}, store, store, tmp_path / "k30",
                      run_id="run1", holdout=_holdout(tmp_path))  # fmt: skip
    kit = open_kit(built.root, live_config)
    assert [m.argo_window_days for m in kit.members] == [None, 30]
    assert kit.argo_fetch_days(live_config) == 30
    assert kits["argo"].argo_fetch_days(live_config) == 10

    fake_inputs(monkeypatch)
    asked: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def argo_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        asked.append((start, end))
        return _profiles(live_config, pd.date_range(start, end))

    log = run_live(live_config, RecordingRepo(kits["root"]), kit, tmp_path / "run",
                   pd.Timestamp("2001-03-15"), fetch, settings(live_config), dry_run=True,
                   argo_fetch=argo_fetch)  # fmt: skip
    assert log.status == "dry_run"
    assert asked == [(pd.Timestamp("2001-02-13"), pd.Timestamp("2001-03-14"))]


def _armor_checkpoint(config: ExperimentConfig, base: Path, out: Path) -> Path:
    """``base`` widened to ARGO + ARMOR3D channels, the ARMOR3D ones with weight 0.5."""
    from oceanscope_ml.data.armor_inputs import enable_armor_input, is_armor_channel

    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(config)
    enable_argo_input(cfg)
    enable_armor_input(cfg)
    names = channel_names(cfg)
    state = expand_input_weights(ckpt["model"], ckpt["channel_names"], names)
    w = state["encoders.0.0.weight"]
    for i, n in enumerate(names):
        if is_armor_channel(n):
            w[:, i] = 0.5
    torch.save({**ckpt, "model": state, "channel_names": names, "run_id": out.stem}, out)
    return out


def _armor_fetch(missing: set[str]) -> Any:
    """Raw ARMOR3D (1/8 degree, ``to``) on every asked day except ``missing``."""
    import xarray as xr

    def fetch_all(product: str, spec: dict[str, Any], start: Any, end: Any) -> Any:
        if product != "armor3d":
            return fetch(product, spec, start, end)
        days = pd.date_range(start, end, freq="D")
        days = days[~days.strftime("%Y-%m-%d").isin(list(missing))]
        lat = 5.0 + 0.0625 + 0.125 * np.arange(20)
        lon = 60.0 + 0.0625 + 0.125 * np.arange(24)
        depth = np.array([0.0, 50.0, 100.0, 300.0, 600.0, 1100.0])
        prof = 29.0 - 0.02 * depth
        to = np.broadcast_to(prof[None, :, None, None], (len(days), len(depth), 20, 24))
        return xr.Dataset(
            {"to": (("time", "depth", "latitude", "longitude"), to.astype(np.float32))},
            coords={"time": days, "depth": depth, "latitude": lat, "longitude": lon},
        )

    return fetch_all


def test_armor_member_runs_live_on_the_newest_day_with_armor3d(
    tmp_path: Path,
    live_config: ExperimentConfig,  # noqa: F811
    store: Path,
    kits: dict[str, Any],
    monkeypatch: Any,
) -> None:
    base = Path(live_config.paths.checkpoints) / "run1" / "best.pt"
    armor_ck = _armor_checkpoint(live_config, base, tmp_path / "ck" / "run1_aia.pt")
    built = build_kit(live_config, {"a": base, "b": armor_ck}, store, store, tmp_path / "ka",
                      run_id="run1", holdout=_holdout(tmp_path))  # fmt: skip
    kit = open_kit(built.root, live_config)
    assert kit.needs_armor and [m.uses_armor for m in kit.members] == [False, True]

    fake_inputs(monkeypatch)
    srcs = settings(live_config).sources
    assert live_inputs.armor_spec(srcs)["dataset_id"].startswith("cmems_obs-mob_glo_phy_nrt")

    def argo_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return _profiles(live_config, pd.date_range(start, end))

    # ARMOR3D NRT lags: the two newest days have none, so the run goes back to 2001-03-12.
    log = run_live(live_config, RecordingRepo(kits["root"]), kit, tmp_path / "run",
                   pd.Timestamp("2001-03-15"), _armor_fetch({"2001-03-13", "2001-03-14"}),
                   settings(live_config), dry_run=True, argo_fetch=argo_fetch)  # fmt: skip
    assert log.status == "dry_run" and log.target_day == "2001-03-12"
    assert "armor3d missing on the day" in log.skipped_days["2001-03-14"]

    # The ARMOR3D channels reach the model: the prediction moves with the ARMOR3D field.
    static = kit.static()
    ocean = static["ocean_mask"].values.astype(bool)
    max_lb = max(live_inputs.var_lookbacks(live_config).values())
    target = pd.Timestamp("2001-03-12")
    days = pd.date_range(target - pd.Timedelta(days=max_lb), target, freq="D")
    inputs = live_inputs.load_inputs(live_config, srcs, days, ocean, fetch)
    armor = live_inputs.load_armor3d(
        live_config, live_inputs.armor_spec(srcs), pd.DatetimeIndex([target]), _armor_fetch(set())
    )
    assert np.isfinite(armor.values[0][:, ocean]).all()
    profiles = _profiles(live_config, pd.date_range(target - pd.Timedelta(days=9), target))
    preds = []
    for k, a in enumerate((armor, armor + 2.0)):
        window = live_inputs.write_window_store(
            tmp_path / f"w{k}.zarr", inputs, static, days[0], target, a
        )
        preds.append(predict_days(live_config, kit, window, [target], argo_profiles=profiles))
    diff = preds[1]["temp_pred"].values - preds[0]["temp_pred"].values
    assert np.nanmax(np.abs(diff)) > 0
