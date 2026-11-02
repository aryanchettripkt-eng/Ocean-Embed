"""Temporal consistency and physics sanity checks on a prediction store (PRD_AI_ML §5.8).

Usage (from ml/):

    python -m oceanscope_ml.evaluation.temporal --config configs/experiment.yaml \
        --predictions outputs/predictions/predictions_<run_id>.zarr \
        --store <parts...> [--stats <stats.zarr>] [--out-dir outputs]

Writes ML-internal files. They are not part of the §6 contract and are not listed in the
manifest:

- ``metrics/<run_id>_val_temporal.csv``: §5.8.1 day-to-day flicker per depth, for the U-Net and
  for the climatology baseline. Ridge needs the model to rebuild its daily values, so it is not
  included here.
- ``metrics/<run_id>_bias_by_year.csv``: §5.8.2 bias and RMSE per split, year and depth.
- ``metrics/<run_id>_bias_trend.csv``: §5.8.2 least-squares slope of the yearly bias (degC/yr).
- ``metrics/<run_id>_physics.csv``: §5.8.3 monthly heat content against ERA5 net heat flux,
  inversions and gross outliers.
- ``figures/timeseries_<run_id>.png``: §5.8.4 time series at three fixed points.

Only dates in the train or validation split are read. The test split is never read
(PRD_AI_ML §0 rule 3).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from PIL import Image, ImageDraw

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.data.transform import check_no_test_leakage
from oceanscope_ml.evaluation.evaluate import _Acc, region_masks

logger = logging.getLogger("oceanscope_ml.evaluation.temporal")

# Constants and thresholds fixed by PRD_AI_ML §5.8. They are not tunables.
RHO = 1025.0  # kg m-3
C_P = 3990.0  # J kg-1 K-1
HEAT_BOTTOM_M = 300.0
INVERSION_C = 0.5  # a level warmer than the one above it by more than this (degC)
OUTLIER_C = 3.0
SECONDS_PER_DAY = 86400.0
FLICKER_RATIO_FLAG = 1.5
FLICKER_FLAG_MIN_DEPTHS = 3  # at this many flagged depths, §10.2 (temporal model) is triggered
HEAT_RMS_RATIO_FLAG = 2.0
INVERSION_FLAG_FRAC = 0.01
BLOCK_DAYS = 16
CLIMATOLOGY = "climatology"
ONE_DAY = pd.Timedelta(days=1)
FLICKER_COLUMNS = [
    "model", "depth_m", "n", "tend_rmse_pred_c", "tend_rmse_glorys_c", "tend_err_rmse_c",
    "lag1_autocorr_pred", "lag1_autocorr_glorys", "flagged",
]  # fmt: skip
BIAS_COLUMNS = ["model", "split", "year", "depth_m", "n", "bias_c", "rmse_c"]
PHYSICS_COLUMNS = ["check", "model", "region", "metric", "value", "n", "flag"]


def heat_content_weights(depths: np.ndarray, bottom_m: float = HEAT_BOTTOM_M) -> np.ndarray:
    """Layer thickness (m) of each level in the 0..bottom_m column, 0 for deeper levels.

    A level owns the layer from the midpoint above it to the midpoint below it, clipped to
    [0, bottom_m]. The weights of the levels inside the column sum to bottom_m.
    """
    z = np.asarray(depths, dtype=np.float64)
    mids = (z[:-1] + z[1:]) / 2.0
    upper = np.concatenate([[0.0], mids])
    lower = np.concatenate([mids, [np.inf]])
    thickness = np.clip(np.minimum(lower, bottom_m) - upper, 0.0, None)
    return np.where(z <= bottom_m, thickness, 0.0)


def _corr_from_sums(n: float, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return float(num / den) if den > 1e-12 else float("nan")


def _split_labels(dates: pd.DatetimeIndex, config: ExperimentConfig) -> np.ndarray:
    return np.where(dates <= pd.Timestamp(config.splits.train[1]), "train", "val")


class _Flicker:
    """Running sums over consecutive-day pairs at each depth (PRD_AI_ML §5.8.1)."""

    def __init__(self, n_d: int) -> None:
        self.n = np.zeros(n_d)
        self.tp2 = np.zeros(n_d)  # sum of dPred^2
        self.tg2 = np.zeros(n_d)  # sum of dGLORYS^2
        self.te2 = np.zeros(n_d)  # sum of (dPred - dGLORYS)^2
        # Lag-1 sums of the anomaly (value minus clim): sx, sy, sxx, syy, sxy per depth.
        self.lag_p = np.zeros((5, n_d))
        self.lag_g = np.zeros((5, n_d))

    def add(self, prev: dict[str, np.ndarray], cur: dict[str, np.ndarray], w: np.ndarray) -> None:
        """``prev``/``cur`` hold pred, glorys, anom_p and anom_g as [pairs, depth, lat, lon]."""
        dp = np.where(w, cur["pred"] - prev["pred"], 0.0)
        dg = np.where(w, cur["glorys"] - prev["glorys"], 0.0)
        self.n += w.sum(axis=(0, 2, 3))
        self.tp2 += (dp**2).sum(axis=(0, 2, 3))
        self.tg2 += (dg**2).sum(axis=(0, 2, 3))
        self.te2 += ((dp - dg) ** 2).sum(axis=(0, 2, 3))
        for target, key in ((self.lag_p, "anom_p"), (self.lag_g, "anom_g")):
            x = np.where(w, prev[key], 0.0)
            y = np.where(w, cur[key], 0.0)
            target += np.stack([x, y, x * x, y * y, x * y]).sum(axis=(1, 3, 4))

    def rows(self, depths: list[float], model: str) -> list[dict[str, Any]]:
        selections: list[tuple[Any, Any]] = [(d, slice(i, i + 1)) for i, d in enumerate(depths)]
        selections.append(("all", slice(None)))
        out: list[dict[str, Any]] = []
        for label, sel in selections:
            n = float(self.n[sel].sum())
            row: dict[str, Any] = {"model": model, "depth_m": label, "n": int(n)}
            if n <= 0:
                row.update(
                    tend_rmse_pred_c=np.nan,
                    tend_rmse_glorys_c=np.nan,
                    tend_err_rmse_c=np.nan,
                    lag1_autocorr_pred=np.nan,
                    lag1_autocorr_glorys=np.nan,
                    flagged=False,
                )
            else:
                tp = float(np.sqrt(self.tp2[sel].sum() / n))
                tg = float(np.sqrt(self.tg2[sel].sum() / n))
                sp = [float(v) for v in self.lag_p[:, sel].sum(axis=1)]
                sg = [float(v) for v in self.lag_g[:, sel].sum(axis=1)]
                row.update(
                    tend_rmse_pred_c=round(tp, 4),
                    tend_rmse_glorys_c=round(tg, 4),
                    tend_err_rmse_c=round(float(np.sqrt(self.te2[sel].sum() / n)), 4),
                    lag1_autocorr_pred=_corr_from_sums(n, *sp),
                    lag1_autocorr_glorys=_corr_from_sums(n, *sg),
                    flagged=bool(tp > FLICKER_RATIO_FLAG * tg),
                )
            out.append(row)
        return out


@dataclass
class _Block:
    """Days ``start .. start + len - 1`` of the prediction store, with truth and climatology."""

    start: int
    dates: pd.DatetimeIndex
    pred: np.ndarray  # [days, depth, lat, lon], degC, NaN where not valid
    glorys: np.ndarray  # [days, depth, lat, lon], degC
    clim: np.ndarray  # [days, depth, lat, lon], degC, climatology of each day
    valid: np.ndarray  # [days, depth, lat, lon], prediction, truth and clim all valid
    flux: np.ndarray  # [days, lat, lon], ERA5 net heat flux, W m-2


@dataclass
class _Physics:
    """Counts and month-start heat contents for PRD_AI_ML §5.8.3."""

    profiles: float = 0.0
    pred_inv: float = 0.0
    glorys_inv: float = 0.0
    pred_inv_not_glorys: float = 0.0
    values: float = 0.0
    outliers: float = 0.0
    # date index -> (heat content of prediction, of GLORYS, cells valid on that day)
    heat: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    # [date index, region] -> region-mean ERA5 net heat flux (W m-2)
    flux_region: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def _load_block(
    pred_ds: xr.Dataset,
    parts: xr.Dataset,
    clim_all: np.ndarray,
    dates: pd.DatetimeIndex,
    start: int,
    stop: int,
) -> _Block:
    idx = slice(start, stop)
    block_dates = dates[start:stop]
    pred = np.asarray(pred_ds["temp_pred"].isel(time=idx).values, dtype=np.float32)
    glorys = np.asarray(parts["thetao"].isel(time=idx).values, dtype=np.float32)
    cov = np.asarray(pred_ds["coverage"].isel(time=idx).values)
    clim = clim_all[block_dates.day_of_year.values - 1]
    valid = np.isfinite(pred) & np.isfinite(glorys) & np.isfinite(clim)
    valid &= (cov == 1)[:, None]
    if "target_valid" in parts.data_vars:
        valid &= np.asarray(parts["target_valid"].isel(time=idx).values, dtype=bool)
    flux = np.asarray(parts["net_heat_flux"].isel(time=idx).values, dtype=np.float32)
    return _Block(start, block_dates, pred, glorys, clim, valid, flux)


def _pack(blk: _Block, pred: np.ndarray, sl: slice) -> dict[str, np.ndarray]:
    return {
        "pred": pred[sl],
        "glorys": blk.glorys[sl],
        "anom_p": pred[sl] - blk.clim[sl],
        "anom_g": blk.glorys[sl] - blk.clim[sl],
    }


def _flicker_inputs(
    blk: _Block, pred: np.ndarray, first_new: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray] | None:
    """Consecutive-day pairs whose later day is a new day of the block (index >= first_new)."""
    n = len(blk.dates)
    j0 = max(first_new, 1)
    if j0 >= n:
        return None
    prev_sl, cur_sl = slice(j0 - 1, n - 1), slice(j0, n)
    consecutive = np.array([blk.dates[j] - blk.dates[j - 1] == ONE_DAY for j in range(j0, n)])
    if not consecutive.any():
        return None
    w = blk.valid[prev_sl] & blk.valid[cur_sl] & consecutive[:, None, None, None]
    return _pack(blk, pred, prev_sl), _pack(blk, pred, cur_sl), w


def _region_mean(field_hw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the masked cells of [days, lat, lon]; NaN for a day with no finite cell."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(field_hw[:, mask], axis=1)


def _heat_content(temp: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Heat content (J m-2) of the column: [days, levels, lat, lon] -> [days, lat, lon]."""
    finite = np.where(np.isfinite(temp), temp, 0.0)
    return RHO * C_P * np.einsum("dzhw,z->dhw", finite, weights)


def _count_physics(blk: _Block, first_new: int, phys: _Physics) -> None:
    """Inversions and gross outliers over the new days of a block (PRD_AI_ML §5.8.3 b, c)."""
    valid = blk.valid[first_new:]
    pred = blk.pred[first_new:]
    glorys = blk.glorys[first_new:]
    profile = valid.all(axis=1)
    inv_p = (np.diff(pred, axis=1) > INVERSION_C).any(axis=1)
    inv_g = (np.diff(glorys, axis=1) > INVERSION_C).any(axis=1)
    phys.profiles += float(profile.sum())
    phys.pred_inv += float((inv_p & profile).sum())
    phys.glorys_inv += float((inv_g & profile).sum())
    phys.pred_inv_not_glorys += float((inv_p & ~inv_g & profile).sum())
    phys.values += float(valid.sum())
    phys.outliers += float(((np.abs(pred - glorys) > OUTLIER_C) & valid).sum())


def _monthly_budget(
    phys: _Physics,
    dates: pd.DatetimeIndex,
    regions: dict[str, np.ndarray],
) -> dict[str, dict[str, list[float]]]:
    """Per region: change in 0-300 m heat content of prediction and GLORYS, and ERA5 flux (J m-2).

    Uses consecutive full calendar months whose first day is in the store. Each region's
    heat-content change is averaged over the cells valid on both month starts.
    """
    out: dict[str, dict[str, list[float]]] = {
        name: {"pred": [], "glorys": [], "flux": []} for name in regions
    }
    starts = sorted(phys.heat)
    for a, b in zip(starts, starts[1:], strict=False):
        if (dates[b] - dates[a]).days != dates[a].days_in_month:
            continue
        if not np.all(np.diff(dates[a : b + 1]) == ONE_DAY):
            continue
        hc_pa, hc_ga, ok_a = phys.heat[a]
        hc_pb, hc_gb, ok_b = phys.heat[b]
        cells = ok_a & ok_b
        for r, (name, mask) in enumerate(regions.items()):
            sel = cells & mask
            flux_days = phys.flux_region[a:b, r]
            if not sel.any() or not np.all(np.isfinite(flux_days)):
                continue
            out[name]["pred"].append(float(np.mean((hc_pb - hc_pa)[sel])))
            out[name]["glorys"].append(float(np.mean((hc_gb - hc_ga)[sel])))
            out[name]["flux"].append(float(np.sum(flux_days)) * SECONDS_PER_DAY)
    return out


def _physics_rows(
    phys: _Physics,
    dates: pd.DatetimeIndex,
    regions: dict[str, np.ndarray],
    model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    budget = _monthly_budget(phys, dates, regions)
    for name, series in budget.items():
        flux = np.asarray(series["flux"])
        n_months = len(flux)
        rms: dict[str, float] = {}
        for who, key in ((model, "pred"), ("glorys", "glorys")):
            vals = np.asarray(series[key])
            corr = float(np.corrcoef(vals, flux)[0, 1]) if n_months >= 3 else np.nan
            rms[who] = float(np.sqrt(np.mean((vals - flux) ** 2))) if n_months else np.nan
            rows.append(
                _row("heat_content_monthly", who, name, "corr_with_era5_flux", corr, n_months)
            )  # noqa: E501
            rows.append(
                _row("heat_content_monthly", who, name, "rms_diff_j_m2", rms[who], n_months)
            )  # noqa: E501
        # Flag only the model's RMS row (PRD §5.8.3: model more than 2x GLORYS).
        model_rms_row = rows[-3]
        model_rms_row["flag"] = bool(n_months and rms[model] > HEAT_RMS_RATIO_FLAG * rms["glorys"])
    profiles = phys.profiles
    frac_not_glorys = phys.pred_inv_not_glorys / profiles if profiles else np.nan
    rows.append(
        _row(
            "inversion_gt_0p5c",
            model,
            "all",
            "frac_pred_inv_not_glorys",
            frac_not_glorys,
            profiles,
            flag=bool(profiles and frac_not_glorys > INVERSION_FLAG_FRAC),
        )
    )
    frac_pred = phys.pred_inv / profiles if profiles else np.nan
    frac_glo = phys.glorys_inv / profiles if profiles else np.nan
    rows.append(_row("inversion_gt_0p5c", model, "all", "frac_pred_inversion", frac_pred, profiles))
    rows.append(
        _row("inversion_gt_0p5c", "glorys", "all", "frac_glorys_inversion", frac_glo, profiles)
    )  # noqa: E501
    frac_out = phys.outliers / phys.values if phys.values else np.nan
    rows.append(
        _row(
            "gross_outlier_gt_3c", model, "all", "frac_cells_abs_diff_gt_3c", frac_out, phys.values
        )
    )
    return rows


def _row(check: str, model: str, region: str, metric: str, value: float, n: float,
         flag: bool | None = None) -> dict[str, Any]:  # fmt: skip
    return {"check": check, "model": model, "region": region, "metric": metric,
            "value": round(float(value), 6), "n": int(n), "flag": flag}  # fmt: skip


DEFAULT_FIGURE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "temporal_checks.yaml"


def _figure_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = raw.get("figure")
    if not section:
        raise ValueError(f"{path} needs a figure section")
    return dict(section)


def _draw_runs(
    draw: ImageDraw.ImageDraw, xy: list[tuple[float, float] | None], colour: tuple[int, int, int]
) -> None:  # noqa: E501
    run: list[tuple[float, float]] = []
    for point in [*xy, None]:
        if point is None:
            if len(run) > 1:
                draw.line(run, fill=colour, width=2)
            run = []
        else:
            run.append(point)


def timeseries_figure(pred: xr.Dataset, figure_config: Path, out_path: Path) -> Path:
    """PNG of ``days`` consecutive days at three points and depths: prediction against GLORYS."""
    fig_cfg = _figure_config(figure_config)
    dates = pd.DatetimeIndex(pd.to_datetime(pred["time"].values))
    start = pd.Timestamp(fig_cfg["start"])
    if start not in dates:
        logger.warning(
            "Figure start %s is not a prediction date; using %s", start.date(), dates[0].date()
        )  # noqa: E501
        start = dates[0]
    i0 = int(np.searchsorted(dates.values, start.to_datetime64()))
    i1 = min(i0 + int(fig_cfg["days"]), len(dates))
    depths = np.asarray(pred["depth"].values, dtype=float)
    lat = np.asarray(pred["lat"].values, dtype=float)
    lon = np.asarray(pred["lon"].values, dtype=float)
    level_idx = [int(np.argmin(np.abs(depths - float(d)))) for d in fig_cfg["depths_m"]]
    panels: list[list[tuple[str, float, np.ndarray, np.ndarray]]] = []
    for name, point in fig_cfg["points"].items():
        y = int(np.argmin(np.abs(lat - float(point["lat"]))))
        x = int(np.argmin(np.abs(lon - float(point["lon"]))))
        if abs(lat[y] - point["lat"]) > 1.0 or abs(lon[x] - point["lon"]) > 1.0:
            logger.warning("Point %s is more than 1 degree from the grid", name)
        row = []
        for li in level_idx:
            sel = {"time": slice(i0, i1), "depth": li, "lat": y, "lon": x}
            p = np.asarray(pred["temp_pred"].isel(sel).values, dtype=float)
            g = np.asarray(pred["temp_glorys"].isel(sel).values, dtype=float)
            row.append((name, float(depths[li]), p, g))
        panels.append(row)

    cell_w, cell_h, margin_l, margin_t = 380, 230, 70, 90
    width = margin_l + len(level_idx) * cell_w + 20
    height = margin_t + len(panels) * (cell_h + 50) + 20
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title = f"Time series, {dates[i0].date()} to {dates[i1 - 1].date()} ({i1 - i0} days)"
    draw.text((margin_l, 12), title, fill=(0, 0, 0))
    draw.text((margin_l, 40), "blue = GLORYS, orange = model (degC)", fill=(0, 0, 0))
    for r, row in enumerate(panels):
        for c, (name, depth, p, g) in enumerate(row):
            x0 = margin_l + c * cell_w
            y0 = margin_t + r * (cell_h + 50)
            pw, ph = cell_w - 60, cell_h - 40
            draw.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(120, 120, 120))
            draw.text((x0, y0 - 18), f"{name}, {depth:.0f} m", fill=(0, 0, 0))
            finite = np.concatenate([p[np.isfinite(p)], g[np.isfinite(g)]])
            if finite.size == 0:
                draw.text((x0 + 10, y0 + ph // 2), "no valid cells in window", fill=(0, 0, 0))
                continue
            lo, hi = float(finite.min()), float(finite.max())
            span = hi - lo if hi > lo else 1.0
            n = len(p)

            def to_xy(v: np.ndarray) -> list[tuple[float, float] | None]:
                pts: list[tuple[float, float] | None] = []
                for k, val in enumerate(v):
                    if np.isfinite(val):
                        px = x0 + (k / max(n - 1, 1)) * pw
                        py = y0 + ph - (val - lo) / span * ph
                        pts.append((px, py))
                    else:
                        pts.append(None)
                return pts

            _draw_runs(draw, to_xy(g), (31, 119, 180))
            _draw_runs(draw, to_xy(p), (255, 127, 14))
            draw.text((x0 + pw + 4, y0), f"{hi:.1f}", fill=(0, 0, 0))
            draw.text((x0 + pw + 4, y0 + ph - 10), f"{lo:.1f}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def temporal_checks(
    config: ExperimentConfig,
    predictions: Path,
    store_path: Path | list[Path],
    stats_path: Path | None = None,
    out_dir: Path | None = None,
    figure_config: Path = DEFAULT_FIGURE_CONFIG,
) -> dict[str, Path]:
    """Run §5.8 on one prediction store and write its ML-internal files; return their paths."""
    pred = xr.open_zarr(predictions)
    run_id = str(pred.attrs["run_id"])
    model = str(pred.attrs.get("model_name", "unet"))
    dates = pd.DatetimeIndex(pd.to_datetime(pred["time"].values))
    check_no_test_leakage(dates, config.splits.test)
    if len(dates) < 2 or dates.min() < pd.Timestamp(config.splits.train[0]) or (
        dates.max() > pd.Timestamp(config.splits.val[1])
    ):  # fmt: skip
        raise ValueError("The prediction store must hold train or validation dates only")

    stores = store_path if isinstance(store_path, list) else [store_path]
    parts = open_parts(stores, stats_path).sel(time=dates)
    depths = [float(d) for d in pred["depth"].values]
    n_d = len(depths)
    regions = region_masks(np.asarray(pred["lat"].values), np.asarray(pred["lon"].values))
    clim_all = np.asarray(parts["clim"].values, dtype=np.float32)
    weights = heat_content_weights(np.asarray(depths))
    heat_levels = weights > 0

    flicker = {model: _Flicker(n_d), CLIMATOLOGY: _Flicker(n_d)}
    bias: dict[tuple[str, str, int], _Acc] = {}
    phys = _Physics(flux_region=np.full((len(dates), len(regions)), np.nan))
    month_starts = {i for i, d in enumerate(dates) if d.day == 1}

    for b0 in range(0, len(dates), BLOCK_DAYS):
        b1 = min(b0 + BLOCK_DAYS, len(dates))
        s = max(b0 - 1, 0)  # one earlier day, so that pairs cross block edges
        blk = _load_block(pred, parts, clim_all, dates, s, b1)
        first_new = b0 - s
        new_dates = blk.dates[first_new:]

        for name, pred_arr in ((model, blk.pred), (CLIMATOLOGY, blk.clim)):
            inputs = _flicker_inputs(blk, pred_arr, first_new)
            if inputs is not None:
                flicker[name].add(*inputs)

        splits = _split_labels(new_dates, config)
        years = new_dates.year.values
        for name, pred_arr in ((model, blk.pred), (CLIMATOLOGY, blk.clim)):
            err = pred_arr[first_new:] - blk.glorys[first_new:]
            valid_new = blk.valid[first_new:]
            for split in np.unique(splits):
                for year in np.unique(years[splits == split]):
                    sel = (splits == split) & (years == year)
                    acc = bias.setdefault((name, str(split), int(year)), _Acc(n_d))
                    k = int(sel.sum())
                    err_k = err[sel].reshape(k, n_d, -1)
                    acc.add(err_k, np.zeros_like(err_k), valid_new[sel].reshape(k, n_d, -1))

        _count_physics(blk, first_new, phys)
        cell_ok: np.ndarray = np.asarray(blk.valid[:, heat_levels].all(axis=1))
        hc_p = _heat_content(blk.pred[:, heat_levels], weights[heat_levels])
        hc_g = _heat_content(blk.glorys[:, heat_levels], weights[heat_levels])
        for j in range(first_new, len(blk.dates)):
            gi = blk.start + j
            if gi in month_starts:
                phys.heat[gi] = (hc_p[j], hc_g[j], cell_ok[j])
        block_slice = slice(blk.start + first_new, blk.start + len(blk.dates))
        for r, mask in enumerate(regions.values()):
            phys.flux_region[block_slice, r] = _region_mean(blk.flux[first_new:], mask)

    metrics_dir = Path(out_dir or config.paths.outputs) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    temporal = pd.DataFrame([r for name, acc in flicker.items() for r in acc.rows(depths, name)])[
        FLICKER_COLUMNS
    ]
    paths["temporal"] = metrics_dir / f"{run_id}_val_temporal.csv"
    temporal.to_csv(paths["temporal"], index=False)
    flagged = temporal[(temporal["model"] == model) & (temporal["depth_m"] != "all")]
    n_flagged = int(flagged["flagged"].sum())
    logger.info(
        "Flicker: %d of %d depths flagged (tendency RMSE > %.1f x GLORYS); §10.2 trigger at %d",
        n_flagged, n_d, FLICKER_RATIO_FLAG, FLICKER_FLAG_MIN_DEPTHS,
    )  # fmt: skip

    bias_rows = []
    for (name, split, year), acc in bias.items():
        for row in acc.rows(depths, {"model": name, "split": split, "year": year}):
            if row["depth_m"] == "all":
                continue
            bias_rows.append({
                "model": name, "split": split, "year": year, "depth_m": row["depth_m"],
                "n": row["n"], "bias_c": row["bias_c"], "rmse_c": row["rmse_c"],
            })  # fmt: skip
    bias_df = pd.DataFrame(bias_rows, columns=BIAS_COLUMNS)
    paths["bias_by_year"] = metrics_dir / f"{run_id}_bias_by_year.csv"
    bias_df.to_csv(paths["bias_by_year"], index=False)
    trend_rows = []
    for (name, depth), group in bias_df.groupby(["model", "depth_m"], sort=False):
        yearly = group.groupby("year")["bias_c"].mean().dropna()  # train and val may share a year
        years = yearly.index.to_numpy(dtype=float)
        slope = float(np.polyfit(years, yearly.to_numpy(), 1)[0]) if len(yearly) >= 2 else np.nan
        trend_rows.append(
            {
                "model": name,
                "depth_m": depth,
                "n_years": int(len(yearly)),
                "slope_c_per_yr": round(slope, 5),
            }
        )
    paths["bias_trend"] = metrics_dir / f"{run_id}_bias_trend.csv"
    pd.DataFrame(trend_rows).to_csv(paths["bias_trend"], index=False)

    physics = pd.DataFrame(_physics_rows(phys, dates, regions, model), columns=PHYSICS_COLUMNS)
    paths["physics"] = metrics_dir / f"{run_id}_physics.csv"
    physics.to_csv(paths["physics"], index=False)

    paths["figure"] = timeseries_figure(
        pred,
        figure_config,
        Path(out_dir or config.paths.outputs) / "figures" / f"timeseries_{run_id}.png",
    )
    logger.info("Temporal checks written for run %s: %s", run_id, ", ".join(map(str, paths)))
    return paths


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRD §5.8 temporal and physics checks.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    temporal_checks(
        config,
        Path(args.predictions),
        [Path(p) for p in args.store],
        stats_path=Path(args.stats) if args.stats else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
