"""ARMOR3D input channels (DECISIONS 2026-11-08): side stores, planes, fine-tune and predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.armor_inputs import (
    ArmorInputs,
    build_side_year,
    enable_armor_input,
    side_days,
)
from oceanscope_ml.data.channels import channel_names
from oceanscope_ml.finetune import finetune_run, load_holdout
from oceanscope_ml.inference.predict import load_checkpoint, predict_store
from tests.synthetic_store import N_LAT, N_LON
from tests.test_finetune import setup  # noqa: F401  (fixture)

OFFSET_C = 0.3  # synthetic ARMOR3D = GLORYS + this


def _match_region(config: ExperimentConfig) -> None:
    """Point the config region at the synthetic store's grid (5-7.25 N, 60-62.75 E)."""
    config.region.lon_min, config.region.lon_max = 60.0, 60.0 + 0.25 * N_LON
    config.region.lat_min, config.region.lat_max = 5.0, 5.0 + 0.25 * N_LAT


def _raw_armor(store: Path, config: ExperimentConfig) -> xr.Dataset:
    """GLORYS of the synthetic store plus OFFSET_C, on a 1/8 degree grid, raw ARMOR3D names."""
    ds = xr.open_zarr(store)
    fine_lat = 5.0 + 0.0625 + 0.125 * np.arange(2 * N_LAT)
    fine_lon = 60.0 + 0.0625 + 0.125 * np.arange(2 * N_LON)
    thetao = np.nan_to_num(np.asarray(ds["thetao"].values), nan=20.0)
    fine = np.repeat(np.repeat(thetao, 2, axis=2), 2, axis=3) + OFFSET_C
    return xr.Dataset(
        {"to": (("time", "depth", "latitude", "longitude"), fine.astype(np.float32))},
        coords={
            "time": ds["time"].values,
            "depth": np.asarray(config.depths_m, dtype=np.float64),
            "latitude": fine_lat,
            "longitude": fine_lon,
        },
    )


def _add_store_armor(store: Path) -> None:
    ds = xr.open_zarr(store)
    armor = (ds["thetao"] + OFFSET_C).astype(np.float32).rename("armor3d")
    out = xr.Dataset({"armor3d": armor.load()})
    out.attrs = dict(ds.attrs)  # appending rewrites the group attributes
    out.to_zarr(store, mode="a")


def test_channels_follow_the_argo_channels(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
) -> None:
    config = setup[0]
    before = channel_names(config)
    enable_armor_input(config)
    after = channel_names(config)
    added = [n for n in after if n not in before]
    assert added == [f"armor3d_anom@{d:g}m" for d in config.depths_m] + ["armor3d_mask"]


def test_side_store_holds_strided_training_days_only(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
    tmp_path: Path,
) -> None:
    config, store = setup[0], setup[1]
    _match_region(config)
    days = side_days(config, 2000, 3)
    assert days.min() >= pd.Timestamp(config.splits.train[0])
    assert days.max() <= pd.Timestamp(config.splits.train[1])
    assert ((days - pd.Timestamp("1993-01-01")).days % 3 == 0).all()

    dest = build_side_year(config, 2000, 3, tmp_path / "armor", source=_raw_armor(store, config))
    side = xr.open_zarr(dest)
    assert pd.DatetimeIndex(side["time"].values).equals(days)
    glorys = xr.open_zarr(store)["thetao"].sel(time=days[0])
    diff = side["armor3d"].isel(time=0).values - glorys.values
    ocean = np.isfinite(glorys.values)
    assert np.allclose(diff[ocean], OFFSET_C, atol=1e-4)
    assert build_side_year(config, 2000, 3, tmp_path / "armor") == dest  # finished: skipped

    late = config.splits.train[1]
    config.splits.train = [config.splits.train[0], "1999-12-31"]
    with pytest.raises(ValueError, match="train-only"):
        build_side_year(config, 2000, 3, tmp_path / "armor2", source=_raw_armor(store, config))
    config.splits.train = [config.splits.train[0], late]


def test_planes_are_scaled_anomalies_and_the_test_year_is_sealed(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
    tmp_path: Path,
) -> None:
    config, store = setup[0], setup[1]
    _match_region(config)
    _add_store_armor(store)
    side = build_side_year(config, 2000, 3, tmp_path / "armor", source=_raw_armor(store, config))
    ds = xr.open_zarr(store)
    sigma = np.asarray(ds.attrs["norm_stats"]["sigma_d"])
    armor = ArmorInputs(config, ds, sigma, side_paths=[tmp_path / "armor"])
    train_day = side_days(config, 2000, 3)[0]
    off_day = train_day + pd.Timedelta(days=1)
    val_day = pd.Timestamp(config.splits.val[0])
    test_day = pd.Timestamp(config.splits.test[0])
    assert armor.has(train_day) and armor.has(val_day)
    assert not armor.has(off_day) and not armor.has(test_day)

    planes = armor.planes(val_day)
    n = len(config.depths_m)
    assert planes.shape == (n + 1, N_LAT, N_LON)
    ocean = np.asarray(ds["ocean_mask"].values, dtype=bool)
    clim = ds["clim"].isel(dayofyear=val_day.day_of_year - 1).values
    glorys = ds["thetao"].sel(time=val_day).values
    want = (glorys + OFFSET_C - clim) / sigma[:, None, None]
    assert np.allclose(planes[:n][:, ocean], want[:, ocean], atol=1e-3)
    assert planes[n][ocean].all() and not planes[n][~ocean].any()
    assert not armor.planes(off_day).any()

    assert ArmorInputs(config, ds, sigma, allow_test=True).has(test_day)
    bad = xr.open_zarr(side).assign_coords(time=[val_day] * xr.open_zarr(side).sizes["time"])
    bad.attrs = xr.open_zarr(side).attrs
    bad.to_zarr(tmp_path / "bad" / "armor3d_2000.zarr", mode="w")
    with pytest.raises(ValueError, match="validation period"):
        ArmorInputs(config, ds, sigma, side_paths=[tmp_path / "bad"])


def test_finetune_with_armor_inputs_and_predict(
    setup: tuple[ExperimentConfig, Path, Path, list[Path], Path],  # noqa: F811
    tmp_path: Path,
) -> None:
    config, store, base, files, holdout_path = setup
    config.inputs.setdefault("argo_input", {})["min_year"] = 2000  # synthetic years
    _match_region(config)
    _add_store_armor(store)
    build_side_year(config, 2000, 1, tmp_path / "armor", source=_raw_armor(store, config))
    holdout = load_holdout(holdout_path)
    with pytest.raises(ValueError, match="side stores"):
        finetune_run(config, [store], base, files, holdout, "base_ar0", lam=1.0,
                     device_choice="cpu", epochs=1, argo_input=True, armor_input=True)  # fmt: skip
    result = finetune_run(
        config, [store], base, files, holdout, "base_ar", lam=1.0, device_choice="cpu",
        epochs=1, argo_input=True, armor_input=True, armor_paths=[tmp_path / "armor"],
    )  # fmt: skip
    best = torch.load(result.best_path, weights_only=False)
    assert best["armor_input"] and best["loss"]["armor_input"]
    assert "armor3d_mask" in best["channel_names"]
    history = pd.read_csv(Path(config.paths.outputs) / "metrics" / "finetune_base_ar.csv")
    assert history["val_argo_n_obs"].gt(0).all()

    model, _ = load_checkpoint(config, result.best_path, torch.device("cpu"))
    dest = predict_store(
        config, store, result.best_path, out_dir=tmp_path / "o", device_choice="cpu",
        argo_paths=files, holdout=holdout,
    )  # fmt: skip
    assert np.isfinite(xr.open_zarr(dest)["temp_pred"].values).any()
    with pytest.raises(ValueError, match="other settings"):
        finetune_run(config, [store], base, files, holdout, "base_ar", lam=1.0,
                     device_choice="cpu", epochs=2, argo_input=True)  # fmt: skip
