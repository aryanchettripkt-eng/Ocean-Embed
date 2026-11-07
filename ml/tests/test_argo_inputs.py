"""ARGO input channels (PRD_AI_ML §5.1, §5.9, §7, P4-6), on synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.argo_inputs import ArgoInputs, enable_argo_input, expand_input_weights
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.finetune import finetune_run, load_holdout
from oceanscope_ml.inference.predict import load_checkpoint, predict_store
from oceanscope_ml.train import build_model, train_run
from tests.test_finetune import HELD_OUT, setup  # noqa: F401  (fixture)

DEPTHS = 3
LAT = np.arange(0.125, 3.0, 0.25)
LON = np.arange(60.125, 63.0, 0.25)
DAY = pd.Timestamp("2000-04-10")


def _grid() -> xr.Dataset:
    dims = ("dayofyear", "depth", "lat", "lon")
    return xr.Dataset(
        {
            "clim": (dims, np.zeros((366, DEPTHS, LAT.size, LON.size))),
            "ocean_mask": (("lat", "lon"), np.ones((LAT.size, LON.size), dtype="int8")),
        },
        coords={"lat": LAT, "lon": LON, "depth": np.arange(DEPTHS)},
    )


def _profiles(n_floats: int = 20) -> pd.DataFrame:
    """One profile per float on DAY, each in its own cell, value = float index."""
    rows = []
    for f in range(n_floats):
        rows.append(
            {
                "platform_number": str(4900000 + f),
                "cycle": 1,
                "day": DAY - pd.Timedelta(days=f % 3),
                "lat": float(LAT[f % LAT.size]),
                "lon": float(LON[f // LAT.size]),
                **{f"t_{d}": float(f) for d in range(DEPTHS)},
            }
        )
    return pd.DataFrame(rows)


def test_held_out_floats_are_refused() -> None:
    prof = _profiles()
    with pytest.raises(AssertionError, match="held-out"):
        ArgoInputs(prof, _grid(), np.ones(DEPTHS), holdout={prof["platform_number"][0]})


def test_modes_never_show_a_target_profile_as_input() -> None:
    argo = ArgoInputs(_profiles(), _grid(), np.ones(DEPTHS), holdout=set(), window_days=5)
    full = argo.sample(DAY, "predict")
    assert full.planes[DEPTHS].sum() == 20 and not full.target_mask.any()  # all inputs

    val = argo.sample(DAY, "val")
    seen = val.planes[DEPTHS] > 0
    assert seen.sum() == 20 - len(argo.score_floats & set(_profiles()["platform_number"]))
    assert not (val.target_mask.any(axis=0) & seen).any()

    for epoch in range(3):
        argo.epoch = epoch
        train = argo.sample(DAY, "train", index=4)
        seen = train.planes[DEPTHS] > 0
        assert not (train.target_mask.any(axis=0) & seen).any()
        assert 0 < seen.sum() < 20


def test_age_channel_and_window() -> None:
    argo = ArgoInputs(_profiles(), _grid(), np.ones(DEPTHS), holdout=set(), window_days=2)
    s = argo.sample(DAY, "predict")
    ages = np.unique(s.planes[DEPTHS + 1][s.planes[DEPTHS] > 0])
    assert np.allclose(ages, [0.5, 1.0])  # day t and t-1 only; t-2 is outside the window
    assert not argo.has_profiles(DAY - pd.Timedelta(days=10))


def test_expanded_model_matches_the_base_model(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
) -> None:
    config, _, _, _, _ = setup
    base_names = channel_names(config)
    base = build_model(config, len(base_names)).eval()
    enable_argo_input(config)
    new_names = channel_names(config)
    assert len(new_names) == len(base_names) + len(config.depths_m) + 2
    wide = build_model(config, len(new_names)).eval()
    wide.load_state_dict(expand_input_weights(base.state_dict(), base_names, new_names))
    x = torch.randn(1, len(new_names), 16, 16)
    keep = [new_names.index(n) for n in base_names]
    with torch.no_grad():
        assert torch.allclose(wide(x), base(x[:, keep]), atol=1e-5)


def test_finetune_with_argo_inputs_and_predict(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store, base, files, holdout_path = setup
    config.inputs.setdefault("argo_input", {})["min_year"] = 2000  # synthetic years
    seen: list[set[str]] = []
    original = ArgoInputs.__init__

    def spy(self: ArgoInputs, profiles: pd.DataFrame, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        seen.append(set(profiles["platform_number"].astype(str)))
        original(self, profiles, *args, **kwargs)

    monkeypatch.setattr(ArgoInputs, "__init__", spy)
    holdout = load_holdout(holdout_path)
    result = finetune_run(
        config, [store], base, files, holdout, "base_ai", lam=1.0, device_choice="cpu",
        epochs=1, argo_input=True,
    )  # fmt: skip
    assert seen and all(HELD_OUT not in s for s in seen)
    history = pd.read_csv(Path(config.paths.outputs) / "metrics" / "finetune_base_ai.csv")
    assert history["epoch"].tolist() == [0, 1]
    assert history["val_argo_n_obs"].gt(0).all()

    model, run_id = load_checkpoint(config, result.best_path, torch.device("cpu"))
    assert run_id == "base_ai"
    with pytest.raises(ValueError, match="--argo"):
        predict_store(config, store, result.best_path, out_dir=tmp_path / "o1", device_choice="cpu")
    dest = predict_store(
        config, store, result.best_path, out_dir=tmp_path / "o2", device_choice="cpu",
        argo_paths=files, holdout=holdout,
    )  # fmt: skip
    pred = xr.open_zarr(dest)
    assert np.isfinite(pred["temp_pred"].values).any()
    assert all(HELD_OUT not in s for s in seen)


def test_train_from_scratch_with_argo_recency_and_year(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
    tmp_path: Path,
) -> None:
    config, store, _, files, holdout_path = setup
    config.inputs.setdefault("argo_input", {})["min_year"] = 2000  # synthetic years
    holdout = load_holdout(holdout_path)
    result = train_run(
        config, store, "scratch_ai", device_choice="cpu", epochs=1, argo_paths=files,
        holdout=holdout, argo_lambda=1.0, recency=2.0, year_input=True,
    )  # fmt: skip
    assert result.epochs_completed == 1
    names = channel_names(config)
    assert names[-1] == "year" and "argo_mask" in names
    best = Path(config.paths.checkpoints) / "scratch_ai" / "best.pt"
    dest = predict_store(
        config, store, best, out_dir=tmp_path / "o", device_choice="cpu",
        argo_paths=files, holdout=holdout,
    )  # fmt: skip
    assert np.isfinite(xr.open_zarr(dest)["temp_pred"].values).any()


def test_finetune_with_warm_day_weighting_runs_and_is_recorded(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
) -> None:
    config, store, base, files, holdout_path = setup
    config.inputs.setdefault("argo_input", {})["min_year"] = 2000  # synthetic years
    holdout = load_holdout(holdout_path)
    with pytest.raises(ValueError, match="argo_input"):
        finetune_run(config, [store], base, files, holdout, "base_w0", lam=1.0,
                     device_choice="cpu", epochs=1, warm_weight=3.0)  # fmt: skip
    result = finetune_run(
        config, [store], base, files, holdout, "base_aw", lam=1.0, device_choice="cpu",
        epochs=1, argo_input=True, warm_weight=3.0,
    )  # fmt: skip
    best = torch.load(result.best_path, weights_only=False)
    assert best["loss"]["warm_weight"] == 3.0


def test_finetune_with_a_longer_argo_window_records_it_for_predict(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
) -> None:
    import copy

    from oceanscope_ml.data.argo_inputs import argo_window

    config, store, base, files, holdout_path = setup
    config.inputs.setdefault("argo_input", {})["min_year"] = 2000  # synthetic years
    fresh = copy.deepcopy(config)
    holdout = load_holdout(holdout_path)
    result = finetune_run(
        config, [store], base, files, holdout, "base_a20", lam=3.0, device_choice="cpu",
        epochs=1, argo_input=True, argo_window=20,
    )  # fmt: skip
    best = torch.load(result.best_path, weights_only=False)
    assert best["argo_window_days"] == 20 and best["loss"]["argo_window"] == 20
    assert argo_window(fresh) == 10
    load_checkpoint(fresh, result.best_path, torch.device("cpu"))
    assert argo_window(fresh) == 20
    with pytest.raises(ValueError, match="other settings"):
        finetune_run(config, [store], base, files, holdout, "base_a20", lam=3.0,
                     device_choice="cpu", epochs=2, argo_input=True, argo_window=30)  # fmt: skip


def test_argo_inputs_reach_the_test_year_only_when_allowed(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oceanscope_ml.data import argo_inputs as ai

    config = setup[0]
    config.inputs.setdefault("argo_input", {})["min_year"] = 1990  # synthetic years
    test_end = pd.Timestamp(config.splits.test[1])
    with pytest.raises(ValueError, match="sealed test"):
        ai.argo_inputs_for(config, [], set(), None, None, [], end=test_end)  # type: ignore[arg-type]
    reads: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    monkeypatch.setattr(
        ai, "read_levels", lambda paths, a, b: reads.append((a, b)) or pd.DataFrame()
    )
    monkeypatch.setattr(ai, "profile_levels", lambda levels, depths, holdout: pd.DataFrame())
    monkeypatch.setattr(ai, "ArgoInputs", lambda *a, **k: "ok")
    start = pd.Timestamp(config.splits.test[0]) - pd.Timedelta(days=10)
    out = ai.argo_inputs_for(config, [], set(), None, None, [], start=start, end=test_end,  # type: ignore[arg-type]
                             allow_test=True)  # fmt: skip
    assert out == "ok"
    assert any(a == pd.Timestamp(config.splits.test[0]) for a, _ in reads)  # cut at the test start
