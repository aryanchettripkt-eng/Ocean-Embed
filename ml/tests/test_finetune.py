"""ARGO fine-tuning on synthetic data only (PRD_AI_ML §5.9, §7). No real float data is used."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data import argo
from oceanscope_ml.finetune import finetune_run, load_holdout
from oceanscope_ml.inference.predict import load_checkpoint
from oceanscope_ml.train import train_run
from tests.synthetic_store import build_synthetic_store, synthetic_config

PRES = [5.0, 30.0, 80.0, 150.0, 300.0, 700.0, 1000.0]
HELD_OUT = "3900001"
WARM_BIAS_C = 1.5  # synthetic floats read this much warmer than the store's thetao


def _argo_rows(store: Path, start: str, end: str, seed: int) -> pd.DataFrame:
    """Two profiles a day from five floats, at ocean cells, WARM_BIAS_C above thetao."""
    ds = xr.open_zarr(store)
    ocean = np.argwhere(np.asarray(ds["ocean_mask"].values).astype(bool))
    depths = np.asarray(ds["depth"].values, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for k, day in enumerate(pd.date_range(start, end, freq="D")):
        t = int(np.flatnonzero(pd.DatetimeIndex(ds["time"].values) == day)[0])
        for j in range(2):
            i_lat, i_lon = ocean[rng.integers(len(ocean))]
            profile = np.asarray(ds["thetao"].isel(time=t, lat=i_lat, lon=i_lon).values, float)
            temps = np.interp(PRES, depths, profile) + WARM_BIAS_C
            platform = HELD_OUT if j == 0 and k % 5 == 0 else f"390000{2 + (k + j) % 4}"
            for p, temp in zip(PRES, temps, strict=True):
                rows.append(
                    {
                        "platform_number": platform,
                        "cycle_number": k * 2 + j,
                        "time": day + pd.Timedelta(hours=6),
                        "latitude": float(ds["lat"].values[i_lat]),
                        "longitude": float(ds["lon"].values[i_lon]),
                        "pres": p,
                        "temp": temp,
                        "temp_qc": 1,
                        "pres_qc": 1,
                        "position_qc": 1,
                    }
                )
    return pd.DataFrame(rows)


def _write_months(rows: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    rows = rows.assign(month=pd.DatetimeIndex(rows["time"]).to_period("M"))
    for month, group in rows.groupby("month"):
        csv = group[list(argo.VARIABLES)].to_csv(index=False)
        units = ",".join("" for _ in argo.VARIABLES) + "\n"
        header, body = csv.split("\n", 1)
        ds = argo.csv_to_dataset(header + "\n" + units + body)
        path = out_dir / f"argo_{month}.nc"
        ds.to_netcdf(path)
        paths.append(path)
    return paths


@pytest.fixture
def setup(tmp_path: Path) -> tuple[ExperimentConfig, Path, Path, list[Path], Path]:
    config = synthetic_config(tmp_path)
    config.finetune_argo["lr"] = 1.0e-3
    store = build_synthetic_store(tmp_path / "store.zarr", config)
    train_run(config, store, "base", device_choice="cpu", epochs=1)
    base = Path(config.paths.checkpoints) / "base" / "best.pt"
    out = tmp_path / "argo"
    out.mkdir()
    rows = pd.concat(
        [
            _argo_rows(store, "2000-03-15", "2000-05-31", seed=1),
            _argo_rows(store, "2000-08-05", "2000-09-15", seed=2),
        ]
    )
    holdout = tmp_path / "argo_eval_floats.json"
    holdout.write_text(json.dumps({"held_out_floats": [HELD_OUT]}), encoding="utf-8")
    return config, store, base, _write_months(rows, out), holdout


def test_profile_levels_drop_held_out_floats_and_never_extrapolate() -> None:
    levels = pd.DataFrame(
        {
            "platform_number": ["1", "1", "1", "2", "2"],
            "cycle_number": [1, 1, 1, 1, 1],
            "time": pd.to_datetime(["2000-03-02"] * 5),
            "latitude": [6.0] * 5,
            "longitude": [61.0] * 5,
            "pres": [10.0, 50.0, 100.0, 10.0, 100.0],
            "temp": [28.0, 24.0, 20.0, 28.0, 20.0],
        }
    )
    out = argo.profile_levels(levels, [5.0, 30.0, 100.0, 200.0], exclude_floats={"2"})
    assert out["platform_number"].tolist() == ["1"]
    row = out.iloc[0]
    assert np.isnan(row["t_0"]) and np.isnan(row["t_3"])  # above 10 dbar, below 100 dbar
    assert row["t_1"] == pytest.approx(26.0)
    assert row["t_2"] == pytest.approx(20.0)


def test_read_levels_keeps_qc_1_and_2_only(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "platform_number": ["1", "1", "1"],
            "cycle_number": [1, 1, 1],
            "time": ["2000-03-02T06:00:00Z"] * 3,
            "latitude": [6.0] * 3,
            "longitude": [61.0] * 3,
            "pres": [10.0, 20.0, 30.0],
            "temp": [28.0, 27.0, 26.0],
            "temp_qc": [1, 2, 4],
            "pres_qc": [1, 1, 1],
            "position_qc": [1, 1, 1],
        }
    )
    csv = rows.to_csv(index=False)
    header, body = csv.split("\n", 1)
    path = tmp_path / "argo_2000-03.nc"
    argo.csv_to_dataset(header + "\n,,UTC,,,,,,,\n" + body).to_netcdf(path)
    levels = argo.read_levels([path], pd.Timestamp("2000-03-01"), pd.Timestamp("2000-03-31"))
    assert levels["pres"].tolist() == [10.0, 20.0]


def test_fetch_refuses_months_in_the_test_period(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)  # test split starts 2000-11-01
    with pytest.raises(ValueError, match="sealed test period"):
        argo.fetch_month(pd.Period("2000-11", freq="M"), config, tmp_path)


def test_finetune_moves_towards_argo_and_writes_a_predict_checkpoint(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],
) -> None:
    config, store, base, files, holdout_path = setup
    result = finetune_run(
        config,
        [store],
        base,
        files,
        load_holdout(holdout_path),
        "base_ft",
        lam=1.0,
        device_choice="cpu",
        epochs=2,
    )
    assert result.best_val_argo_rmse_c < result.base_val_argo_rmse_c
    assert result.best_epoch >= 1

    history = pd.read_csv(Path(config.paths.outputs) / "metrics" / "finetune_base_ft.csv")
    assert history["epoch"].tolist() == [0, 1, 2]
    assert history["val_argo_n_obs"].gt(0).all()

    model, run_id = load_checkpoint(config, result.best_path, torch.device("cpu"))
    assert run_id == "base_ft"

    # A second call resumes from last_ft.pt, finds the run finished and changes nothing.
    again = finetune_run(
        config,
        [store],
        base,
        files,
        load_holdout(holdout_path),
        "base_ft",
        lam=1.0,
        device_choice="cpu",
        epochs=2,
    )
    assert again.best_epoch == result.best_epoch


def test_held_out_float_never_becomes_a_target(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store, base, files, holdout_path = setup
    seen: list[set[str]] = []
    original = argo.profile_levels

    def spy(levels: pd.DataFrame, depths: list[float], exclude: set[str]) -> pd.DataFrame:
        out = original(levels, depths, exclude)
        seen.append(set(out["platform_number"]))
        return out

    monkeypatch.setattr("oceanscope_ml.finetune.profile_levels", spy)
    finetune_run(
        config,
        [store],
        base,
        files,
        load_holdout(holdout_path),
        "spy_ft",
        lam=0.1,
        device_choice="cpu",
        epochs=1,
    )
    assert len(seen) == 2 and all(seen)
    assert all(HELD_OUT not in floats for floats in seen)
