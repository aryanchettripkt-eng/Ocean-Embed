"""P5-3 evaluation report generated from metric files. All fixtures here are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.evaluation.argo_validation import HEADLINE_COLUMNS
from oceanscope_ml.evaluation.evaluate import evaluate_predictions
from oceanscope_ml.evaluation.report import (
    ARGO_CAVEAT,
    GLORYS_CAVEAT,
    SYNTHETIC_BANNER,
    generate_report,
)
from oceanscope_ml.evaluation.temporal import temporal_checks

RUN = "synth_run"
MODEL = "SYNTHETIC_unet"
BACKEND_CAVEATS = Path(__file__).resolve().parents[2] / "backend" / "app" / "schemas" / "caveats.py"


def _metric_rows(depths: list[float], rmse: list[float], skill: list[float]) -> pd.DataFrame:
    rows = [
        {"model": MODEL, "split": "val", "depth_m": d, "n": 100, "mae_c": r * 0.8,
         "rmse_c": r, "bias_c": 0.01, "skill_vs_clim_pct": s}
        for d, r, s in zip(depths, rmse, skill, strict=True)
    ]  # fmt: skip
    rows.append({**rows[0], "depth_m": "all", "n": 100 * len(depths), "rmse_c": 0.555})
    return pd.DataFrame(rows)


def write_fixture_metrics(out: Path, depths: list[float], n_beat: int = 12) -> None:
    """Synthetic §6.3 and §5.8 files: the model beats climatology at ``n_beat`` depths."""
    metrics = out / "metrics"
    metrics.mkdir(parents=True)
    n = len(depths)
    skill = [20.0] * n_beat + [-5.0] * (n - n_beat)
    _metric_rows(depths, [0.4321] * n, skill).to_csv(metrics / f"{RUN}_val.csv", index=False)
    for name in ("climatology", "ridge"):
        base = _metric_rows(depths, [0.9] * n, [0.0] * n).assign(model=f"SYNTHETIC_{name}")
        base.to_csv(metrics / f"baseline_{name}_val.csv", index=False)
    season = pd.concat(
        [_metric_rows(depths, [0.5] * n, [1.0] * n).assign(season=s) for s in "DJF MAM".split()]
    )
    season.to_csv(metrics / f"{RUN}_val_by_season.csv", index=False)
    region = _metric_rows(depths, [0.6] * n, [1.0] * n).assign(region="bay_of_bengal")
    region.to_csv(metrics / f"{RUN}_val_by_region.csv", index=False)

    head = []
    for product, rmse, diff_high in [
        ("model", 0.5, 0.0), ("glorys12", 0.7, -0.05), ("grep_mean", 0.8, -0.1),
        ("armor3d", 0.9, -0.2),
    ]:  # fmt: skip
        head.append(
            {"period": "val", "product": product, "n_floats": 12, "n_obs": 3400,
             "rmse_c": rmse, "ci_low_c": rmse - 0.1, "ci_high_c": rmse + 0.1,
             "diff_vs_model_c": 0.5 - rmse, "diff_ci_low_c": 0.5 - rmse - 0.1,
             "diff_ci_high_c": diff_high}
        )  # fmt: skip
        head.append({"period": "test", "product": product, "n_floats": 0, "n_obs": 0})
    headline = pd.DataFrame(head, columns=HEADLINE_COLUMNS)
    headline.to_csv(metrics / f"{RUN}_headline.csv", index=False)

    temporal = []
    for i, d in enumerate([*depths, "all"]):
        jumpy = i < 3  # three flagged depths: the §10.2 trigger
        temporal.append(
            {"model": MODEL, "depth_m": d, "n": 50, "tend_rmse_pred_c": 0.9 if jumpy else 0.1,
             "tend_rmse_glorys_c": 0.2, "tend_err_rmse_c": 0.3, "lag1_autocorr_pred": 0.5,
             "lag1_autocorr_glorys": 0.9, "flagged": jumpy and d != "all"}
        )  # fmt: skip
    pd.DataFrame(temporal).to_csv(metrics / f"{RUN}_val_temporal.csv", index=False)
    bias = [
        {"model": m, "split": "val", "year": y, "depth_m": d, "n": 10,
         "bias_c": b0 + step * (y - 2018), "rmse_c": 0.5}
        for m, b0, step in [(MODEL, 0.0, -0.05), ("climatology", 0.1, 0.02)]
        for y in (2018, 2019, 2020)
        for d in depths
    ]  # fmt: skip
    pd.DataFrame(bias).to_csv(metrics / f"{RUN}_bias_by_year.csv", index=False)
    trend = [
        {"model": m, "depth_m": d, "n_years": 3, "slope_c_per_yr": slope}
        for m, slope in [(MODEL, -0.05), ("climatology", 0.02)]
        for d in depths
    ]
    pd.DataFrame(trend).to_csv(metrics / f"{RUN}_bias_trend.csv", index=False)
    physics = [
        {"check": "heat_content_monthly", "model": MODEL, "region": "bay_of_bengal",
         "metric": "rms_diff_j_m2", "value": 9.0e7, "n": 3, "flag": True},
        {"check": "heat_content_monthly", "model": "glorys", "region": "bay_of_bengal",
         "metric": "rms_diff_j_m2", "value": 4.0e7, "n": 3, "flag": None},
        {"check": "inversion_gt_0p5c", "model": MODEL, "region": "all",
         "metric": "frac_pred_inv_not_glorys", "value": 0.001, "n": 500, "flag": False},
    ]  # fmt: skip
    pd.DataFrame(physics).to_csv(metrics / f"{RUN}_physics.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps({"synthetic": True, "active_run_id": RUN, "test_evaluated": False})
    )


def test_caveats_match_the_backend_strings() -> None:
    namespace: dict[str, object] = {}
    exec(BACKEND_CAVEATS.read_text(encoding="utf-8"), namespace)  # constants only
    assert GLORYS_CAVEAT == namespace["GLORYS_CAVEAT"]
    assert ARGO_CAVEAT == namespace["ARGO_CAVEAT"]


def test_report_from_fixture_metrics(config: ExperimentConfig, tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    write_fixture_metrics(out, config.depths_m)
    text = generate_report(config, RUN, out).read_text(encoding="utf-8")

    assert text.startswith(f"# Evaluation report: {RUN}")
    assert SYNTHETIC_BANNER in text
    assert GLORYS_CAVEAT in text and ARGO_CAVEAT in text
    assert "Validation scores are optimistic" in text
    assert "Test year not evaluated" in text
    assert "**Validation dates: Met.**" in text
    assert "**Test year (sealed):** not evaluated." in text
    assert "0.432" in text and "0.555" in text  # numbers come from the CSV
    assert "at 12 of 15 depths; the gate needs 10. Passed." in text
    assert "**Flagged at 3 depth(s):**" in text and "§10.2" in text
    assert "-0.050" in text and "SYNTHETIC_unet slope (°C/yr)" in text
    assert "2019 (val)" in text
    assert "**Flagged:** heat_content_monthly / bay_of_bengal / rms_diff_j_m2." in text
    for missing in (f"ablation_{RUN}.csv", f"argo_{RUN}.csv"):
        assert f"`metrics/{missing}`" in text
    assert f"`figures/timeseries_{RUN}.png`" in text
    assert "Final test, sealed year" in text and "Not evaluated" in text


def test_m2_gate_failure_is_reported(config: ExperimentConfig, tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    write_fixture_metrics(out, config.depths_m, n_beat=9)
    text = generate_report(config, RUN, out).read_text(encoding="utf-8")
    assert "at 9 of 15 depths; the gate needs 10. Not passed" in text


def test_headline_not_met_when_an_interval_crosses_zero(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    out = tmp_path / "outputs"
    write_fixture_metrics(out, config.depths_m)
    path = out / "metrics" / f"{RUN}_headline.csv"
    head = pd.read_csv(path)
    head.loc[(head["period"] == "val") & (head["product"] == "armor3d"), "diff_ci_high_c"] = 0.02
    head.to_csv(path, index=False)
    text = generate_report(config, RUN, out).read_text(encoding="utf-8")
    assert "**Validation dates: Not met.**" in text


def test_empty_outputs_still_give_caveats(config: ExperimentConfig, tmp_path: Path) -> None:
    text = generate_report(config, RUN, tmp_path / "empty").read_text(encoding="utf-8")
    assert GLORYS_CAVEAT in text and ARGO_CAVEAT in text
    assert SYNTHETIC_BANNER not in text  # nothing says synthetic, so no banner
    assert text.count("Not produced yet.") >= 8
    assert f"`metrics/{RUN}_val.csv`" in text


def test_test_year_appears_only_from_final_test_file(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    out = tmp_path / "outputs"
    write_fixture_metrics(out, config.depths_m)
    test = _metric_rows(config.depths_m, [0.7777] * 15, [10.0] * 15).assign(split="test")
    test.to_csv(out / "metrics" / f"{RUN}_test.csv", index=False)
    text = generate_report(config, RUN, out).read_text(encoding="utf-8")
    assert "was scored once by `final_test`" in text
    assert "0.778" in text


def _write_val_predictions(store: Path, dest: Path, config: ExperimentConfig) -> Path:
    """A §6.2-shaped store for the validation dates: GLORYS + 0.25 °C. Synthetic only."""
    ds = xr.open_zarr(store).sel(time=slice(*config.splits.val))
    glorys = np.asarray(ds["thetao"].values, dtype=np.float32)
    ocean = np.asarray(ds["ocean_mask"].values, dtype=bool)
    coverage = np.broadcast_to(ocean.astype(np.int8), (glorys.shape[0], *ocean.shape)).copy()
    xr.Dataset(
        {
            "temp_pred": (("time", "depth", "lat", "lon"), (glorys + 0.25).astype(np.float16)),
            "temp_glorys": (("time", "depth", "lat", "lon"), glorys.astype(np.float16)),
            "coverage": (("time", "lat", "lon"), coverage),
        },
        coords={
            "time": ds["time"].values,
            "depth": ds["depth"].values,
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
        },  # fmt: skip
        attrs={"run_id": RUN, "model_name": MODEL},
    ).to_zarr(dest, mode="w", zarr_format=2)
    return dest


def test_end_to_end_on_synthetic_store(
    config: ExperimentConfig, store: Path, tmp_path: Path
) -> None:
    """Prediction store → evaluate → §5.8 checks → report, without torch."""
    out = tmp_path / "outputs"
    (out / "predictions").mkdir(parents=True)
    pred = _write_val_predictions(store, out / "predictions" / f"predictions_{RUN}.zarr", config)
    evaluate_predictions(config, pred, store, out_dir=out)
    temporal_checks(config, pred, store, out_dir=out)
    text = generate_report(config, RUN, out).read_text(encoding="utf-8")
    val = pd.read_csv(out / "metrics" / f"{RUN}_val.csv")
    assert f"{val['rmse_c'].iloc[0]:.3f}" in text
    assert SYNTHETIC_BANNER in text  # the model name carries SYNTHETIC
    assert f"![Model vs GLORYS at three points](../figures/timeseries_{RUN}.png)" in text
    for name in ("val_temporal", "bias_by_year", "bias_trend", "physics"):
        assert f"`metrics/{RUN}_{name}.csv`" not in text
    assert "### Physics sanity checks" in text and "2000 (val)" in text
    assert "climatology slope (°C/yr)" in text  # one year gives NaN slopes, still shown
