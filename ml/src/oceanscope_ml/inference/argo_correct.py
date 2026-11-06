"""ARGO residual correction of a prediction store (objective analysis of the model's error).

Usage (from ml/):

    python -m oceanscope_ml.inference.argo_correct --config configs/experiment.yaml \
        --predictions outputs/predictions/predictions_<run_id>.zarr \
        --argo <argo_YYYY-MM.nc...> --holdout configs/argo_eval_floats.json \
        --run-id <run_id>_oi [--out-dir outputs] [--params L,TAU,WINDOW,NOISE] [--per-depth]

The model learns to copy GLORYS, so on its own it cannot use the floats that are in the water
on day t. This step does: for every profile of a non-held-out float, the residual
r = T_argo - T_pred is taken at the profile's cell and day, on the 15 model depths. The
correction at a cell on day t is, per depth,

    c = sum(w_i * r_i) / (sum(w_i) + noise),
    w_i = exp(-d_i^2 / (2 L^2)) * exp(-(t - t_i) / tau),

over the profiles with t - window < t_i <= t (past and same day only, as in live use). d_i is
the distance in degrees (longitude scaled by cos(lat)). ``noise`` shrinks the correction where
floats are few: a profile at the same cell and day moves the field by r / (1 + noise).

L, tau, window and noise are picked on the validation period with non-held-out floats only:
20% of them (by float, fixed seed) are kept out of the residuals and scored, and the setting with
the lowest RMSE on those in the Bay of Bengal at 50-300 m is used (the §1.1 band). The final
store then uses all non-held-out floats. Held-out floats (PRD_AI_ML §7) are never read as
residuals, in tuning or in the final store. Only validation dates are read or written.

With ``--per-depth`` each depth gets its own setting: the one with the lowest RMSE on the tuning
floats at that depth (whole domain), or no correction at all where none beats the model by at
least 1%. This keeps the correction away from levels where the floats only add noise (the
surface level was made worse by a single shared setting on main2_ai).

Two guards against the failure seen on the held-out floats (the tuning floats said the 0.5 m
level improved, the held-out floats said it got worse, 0.34 -> 0.77 C):

- ``--min-depth M``: levels shallower than M metres are never corrected (their residuals are
  dropped before tuning).
- ``--max-depth M``: levels deeper than M metres are never corrected either (on the final half
  the correction made 643 m and 1062 m slightly worse).
- ``--outlier-sigma K``: at each depth, residuals further than K robust standard deviations
  (1.4826 x the median absolute deviation) from the median are dropped before tuning and
  before the final store, so a few bad profiles or frontal mismatches can't drive the field.

Writes ``<out>/predictions/predictions_<run_id>.zarr`` (same layout as ``inference.predict``,
``temp_pred`` corrected, ``temp_glorys`` and ``coverage`` copied) and
``<out>/metrics/argo_correct_tuning_<run_id>.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.argo import profile_levels, read_levels
from oceanscope_ml.data.transform import check_no_test_leakage
from oceanscope_ml.evaluation.argo_validation import _nearest_cell
from oceanscope_ml.evaluation.evaluate import BAY_OF_BENGAL_MIN_LON, EQUATORIAL_MAX_LAT

logger = logging.getLogger("oceanscope_ml.inference.argo_correct")

BAND_M = (50.0, 300.0)
TUNE_FRACTION = 0.2
TUNE_SEED = 42
BLOCK_DAYS = 16
MIN_GAIN = 0.01  # per-depth: a setting must beat no correction by 1% at that depth
GRID = {
    "length_deg": (0.5, 1.0, 2.0, 3.0),
    "tau_days": (5.0, 15.0, 45.0),
    "window_days": (10, 30),
    "noise": (0.25, 1.0, 3.0),
}
TUNING_COLUMNS = [
    "length_deg",
    "tau_days",
    "window_days",
    "noise",
    "rmse_c",
    "bob_band_rmse_c",
    "n_obs",
    "bob_band_n_obs",
]


@dataclass(frozen=True)
class OIParams:
    length_deg: float
    tau_days: float
    window_days: int
    noise: float


Plan = tuple[OIParams | None, ...]  # one setting per depth; None = no correction there


def uniform_plan(params: OIParams, n_depth: int) -> Plan:
    return (params,) * n_depth


def plan_json(plan: Plan) -> str:
    return json.dumps([None if p is None else asdict(p) for p in plan])


@dataclass
class Residuals:
    """Model-minus-ARGO residuals: one row per profile, values on the model depths."""

    platform: np.ndarray  # str [n]
    day: np.ndarray  # int days since 1970-01-01 [n]
    lat: np.ndarray  # [n]
    lon: np.ndarray  # [n]
    values: np.ndarray  # [n, D], T_argo - T_pred, NaN where missing

    def subset(self, keep: np.ndarray) -> Residuals:
        return Residuals(
            self.platform[keep], self.day[keep], self.lat[keep], self.lon[keep], self.values[keep]
        )

    def __len__(self) -> int:
        return int(self.day.size)


def _day_number(times: pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    days = pd.DatetimeIndex(times).normalize().values.astype("datetime64[D]")
    return days.astype(np.int64)


def residuals(profiles: pd.DataFrame, pred: xr.Dataset) -> Residuals:
    """Residuals at each profile's nearest cell and day, where the prediction is valid."""
    n_depth = int(pred.sizes["depth"])
    cols = [f"t_{d}" for d in range(n_depth)]
    lat_grid = np.asarray(pred["lat"].values, dtype=float)
    lon_grid = np.asarray(pred["lon"].values, dtype=float)
    pred_days = _day_number(pred["time"].values)
    empty = Residuals(
        np.array([], dtype=str), np.array([], dtype=np.int64), np.array([]), np.array([]),
        np.zeros((0, n_depth)),
    )  # fmt: skip
    if profiles.empty:
        return empty
    day = _day_number(profiles["day"].values)
    i_lat, lat_ok = _nearest_cell(lat_grid, profiles["lat"].to_numpy(float))
    i_lon, lon_ok = _nearest_cell(lon_grid, profiles["lon"].to_numpy(float))
    keep = lat_ok & lon_ok & np.isin(day, pred_days)
    rows = np.flatnonzero(keep)
    values = np.full((rows.size, n_depth), np.nan)
    pos = {int(d): k for k, d in enumerate(pred_days)}
    obs = profiles[cols].to_numpy(float)
    order = pd.Series(rows).groupby(day[rows]).indices
    for d, idx in order.items():
        r = rows[idx]
        t = pos[int(d)]
        field = pred.isel(time=t)
        temp = np.asarray(field["temp_pred"].transpose("depth", "lat", "lon").values, float)
        cov = np.asarray(field["coverage"].values)
        model = temp[:, i_lat[r], i_lon[r]].T
        ok = (cov[i_lat[r], i_lon[r]] == 1)[:, None]
        values[idx] = np.where(ok, obs[r] - model, np.nan)
    good = np.isfinite(values).any(axis=1)
    return Residuals(
        profiles["platform_number"].to_numpy(str)[rows][good],
        day[rows][good],
        profiles["lat"].to_numpy(float)[rows][good],
        profiles["lon"].to_numpy(float)[rows][good],
        values[good],
    )


def drop_shallow(res: Residuals, depths: np.ndarray, min_depth_m: float) -> Residuals:
    """Residuals with the levels shallower than ``min_depth_m`` set to NaN."""
    values = res.values.copy()
    values[:, depths < min_depth_m] = np.nan
    keep = np.isfinite(values).any(axis=1)
    out = Residuals(res.platform, res.day, res.lat, res.lon, values)
    return out.subset(keep)


def drop_deep(res: Residuals, depths: np.ndarray, max_depth_m: float) -> Residuals:
    """Residuals with the levels deeper than ``max_depth_m`` set to NaN."""
    values = res.values.copy()
    values[:, depths > max_depth_m] = np.nan
    keep = np.isfinite(values).any(axis=1)
    return Residuals(res.platform, res.day, res.lat, res.lon, values).subset(keep)


def drop_outliers(res: Residuals, k: float) -> tuple[Residuals, np.ndarray]:
    """Residuals with values beyond ``k`` robust sigmas of the per-depth median set to NaN.
    Returns them and the number dropped per depth."""
    values = res.values.copy()
    dropped = np.zeros(values.shape[1], dtype=int)
    for d in range(values.shape[1]):
        col = values[:, d]
        ok = np.isfinite(col)
        if ok.sum() < 5:
            continue
        med = float(np.median(col[ok]))
        mad = float(np.median(np.abs(col[ok] - med)))
        if mad == 0.0:
            continue
        bad = ok & (np.abs(col - med) > k * 1.4826 * mad)
        dropped[d] = int(bad.sum())
        col[bad] = np.nan
    keep = np.isfinite(values).any(axis=1)
    out = Residuals(res.platform, res.day, res.lat, res.lon, values)
    return out.subset(keep), dropped


def correction(
    res: Residuals, params: OIParams, day: int, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """[m, D] correction at points (lat, lon) on ``day`` from the residuals before or on it."""
    n_depth = res.values.shape[1]
    sel = (res.day <= day) & (res.day > day - params.window_days)
    if not sel.any() or lat.size == 0:
        return np.zeros((lat.size, n_depth))
    r_lat, r_lon, r_val = res.lat[sel], res.lon[sel], res.values[sel]
    dt = (day - res.day[sel]).astype(float)
    dy = lat[:, None] - r_lat[None, :]
    coslat = np.cos(np.deg2rad((lat[:, None] + r_lat[None, :]) / 2.0))
    dx = (lon[:, None] - r_lon[None, :]) * coslat
    w = (
        np.exp(-(dx**2 + dy**2) / (2.0 * params.length_deg**2))
        * np.exp(-dt / params.tau_days)[None]
    )
    finite = np.isfinite(r_val)
    num = w @ np.where(finite, r_val, 0.0)
    den = w @ finite.astype(float)
    return num / (den + params.noise)


def plan_correction(
    res: Residuals, plan: Plan, day: int, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """[m, D] correction with one setting per depth (zero where the plan says None)."""
    out = np.zeros((lat.size, len(plan)))
    for params in {p for p in plan if p is not None}:
        cols = [k for k, p in enumerate(plan) if p == params]
        out[:, cols] = correction(res, params, day, lat, lon)[:, cols]
    return out


def _band_mask(lat: np.ndarray, lon: np.ndarray, depths: np.ndarray) -> np.ndarray:
    cells = (lon >= BAY_OF_BENGAL_MIN_LON) & (lat >= EQUATORIAL_MAX_LAT)
    band = (depths >= BAND_M[0]) & (depths <= BAND_M[1])
    return cells[:, None] & band[None, :]


def split_floats(
    res: Residuals, fraction: float = TUNE_FRACTION, seed: int = TUNE_SEED
) -> set[str]:
    """The non-held-out floats kept out of the residuals for tuning (by float, fixed seed)."""
    floats = np.unique(res.platform)
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction * floats.size)))
    return {str(f) for f in rng.choice(floats, size=n, replace=False)}


def errors(fit: Residuals, tune: Residuals, params: OIParams | None) -> np.ndarray:
    """[n, D] remaining error at the tuning profiles; ``params=None`` is the model alone."""
    err = tune.values.copy()
    if params is not None:
        for d in np.unique(tune.day):
            idx = np.flatnonzero(tune.day == d)
            err[idx] -= correction(fit, params, int(d), tune.lat[idx], tune.lon[idx])
    return err


def _depth_rmse(err: np.ndarray) -> np.ndarray:
    finite = np.isfinite(err)
    sq = np.where(finite, err, 0.0) ** 2
    n = finite.sum(axis=0)
    return np.where(n > 0, np.sqrt(sq.sum(axis=0) / np.maximum(n, 1)), np.nan)


def _summary(err: np.ndarray, tune: Residuals, depths: np.ndarray) -> dict:
    finite = np.isfinite(err)
    band = finite & _band_mask(tune.lat, tune.lon, depths)
    return {
        "rmse_c": float(np.sqrt(np.mean(err[finite] ** 2))) if finite.any() else np.nan,
        "bob_band_rmse_c": float(np.sqrt(np.mean(err[band] ** 2))) if band.any() else np.nan,
        "n_obs": int(finite.sum()),
        "bob_band_n_obs": int(band.sum()),
    }


def score(fit: Residuals, tune: Residuals, params: OIParams | None, depths: np.ndarray) -> dict:
    """RMSE of the corrected model at the tuning profiles; ``params=None`` scores the model."""
    return _summary(errors(fit, tune, params), tune, depths)


def tune(
    res: Residuals, depths: np.ndarray, per_depth: bool = False
) -> tuple[Plan, pd.DataFrame, pd.DataFrame | None]:
    """Grid search on the tuning floats; returns the plan, the full table and, with
    ``per_depth``, the per-depth choices."""
    tune_floats = split_floats(res)
    is_tune = np.isin(res.platform, list(tune_floats))
    fit, held = res.subset(~is_tune), res.subset(is_tune)
    logger.info(
        "Tuning: %d profiles from %d floats as residuals, %d profiles from %d floats scored",
        len(fit), np.unique(fit.platform).size, len(held), len(tune_floats),
    )  # fmt: skip
    rows = [{"length_deg": 0.0, "tau_days": 0.0, "window_days": 0, "noise": 0.0}]
    base_err = errors(fit, held, None)
    rows[0].update(_summary(base_err, held, depths))
    settings: list[OIParams] = []
    depth_rmse: list[np.ndarray] = []
    logger.info("No correction: RMSE %.3f C, BoB 50-300 m %.3f C", rows[0]["rmse_c"],
                rows[0]["bob_band_rmse_c"])  # fmt: skip
    for values in itertools.product(*GRID.values()):
        params = OIParams(*values)  # type: ignore[arg-type]
        row: dict = asdict(params)
        err = errors(fit, held, params)
        row.update(_summary(err, held, depths))
        rows.append(row)
        settings.append(params)
        depth_rmse.append(_depth_rmse(err))
        logger.info("%s: RMSE %.3f C, BoB 50-300 m %.3f C", params, row["rmse_c"],
                    row["bob_band_rmse_c"])  # fmt: skip
    table = pd.DataFrame(rows, columns=TUNING_COLUMNS)
    grid = table.iloc[1:]
    best = grid.loc[grid["bob_band_rmse_c"].idxmin()]
    params = OIParams(
        float(best["length_deg"]), float(best["tau_days"]), int(best["window_days"]),
        float(best["noise"]),
    )  # fmt: skip
    if not per_depth:
        return uniform_plan(params, depths.size), table, None

    base_rmse = _depth_rmse(base_err)
    grid_rmse = np.vstack(depth_rmse)  # [settings, D]
    plan: list[OIParams | None] = []
    choice_rows = []
    for k, depth in enumerate(depths):
        col = np.where(np.isfinite(grid_rmse[:, k]), grid_rmse[:, k], np.inf)
        j = int(np.argmin(col))
        use = np.isfinite(base_rmse[k]) and col[j] < (1.0 - MIN_GAIN) * base_rmse[k]
        plan.append(settings[j] if use else None)
        choice_rows.append(
            {
                "depth_m": float(depth),
                **({f: (getattr(settings[j], f) if use else 0) for f in GRID}),
                "corrected": bool(use),
                "rmse_none_c": float(base_rmse[k]),
                "rmse_chosen_c": float(col[j] if use else base_rmse[k]),
                "n_obs": int(np.isfinite(base_err[:, k]).sum()),
            }
        )
        logger.info("  %.1f m: %s (RMSE %.3f -> %.3f C)", depth,
                    settings[j] if use else "no correction", base_rmse[k],
                    col[j] if use else base_rmse[k])  # fmt: skip
    final = tuple(plan)
    err = base_err.copy()
    for params_k in {p for p in final if p is not None}:
        cols = [k for k, p in enumerate(final) if p == params_k]
        err[:, cols] = errors(fit, held, params_k)[:, cols]
    summary = _summary(err, held, depths)
    logger.info("Per-depth plan: RMSE %.3f C, BoB 50-300 m %.3f C", summary["rmse_c"],
                summary["bob_band_rmse_c"])  # fmt: skip
    return final, table, pd.DataFrame(choice_rows)


def write_store(pred_path: Path, res: Residuals, plan: Plan, run_id: str, dest: Path) -> Path:
    """Copy the prediction store to ``dest`` with ``temp_pred`` corrected on valid cells."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(pred_path, dest)
    pred = xr.open_zarr(dest)
    dtype = pred["temp_pred"].dtype
    lat2, lon2 = np.meshgrid(pred["lat"].values, pred["lon"].values, indexing="ij")
    days = _day_number(pred["time"].values)
    n_time = days.size
    sq = np.zeros(int(pred.sizes["depth"]))
    n_cells = 0
    for b0 in range(0, n_time, BLOCK_DAYS):
        b1 = min(b0 + BLOCK_DAYS, n_time)
        block = pred.isel(time=slice(b0, b1)).load()
        temp = block["temp_pred"].transpose("time", "depth", "lat", "lon").values.astype(float)
        cov = block["coverage"].values
        for k in range(b1 - b0):
            valid = cov[k] == 1
            c = plan_correction(res, plan, int(days[b0 + k]), lat2[valid], lon2[valid])
            temp[k][:, valid] += c.T
            sq += (c**2).sum(axis=0)
            n_cells += int(valid.sum())
        out = xr.Dataset({"temp_pred": (("time", "depth", "lat", "lon"), temp.astype(dtype))})
        out.to_zarr(dest, region={"time": slice(b0, b1)}, mode="r+")
        logger.info("Corrected %d / %d days", b1, n_time)
    group = zarr.open_group(str(dest), mode="r+")
    base_run = str(group.attrs.get("run_id", ""))
    group.attrs.update(
        {
            "run_id": run_id,
            "base_run_id": base_run,
            "argo_correction": plan_json(plan),
        }
    )
    zarr.consolidate_metadata(str(dest))
    rms = np.sqrt(sq / max(n_cells, 1))
    logger.info("RMS correction per depth (C): %s", np.round(rms, 3).tolist())
    return dest


def run(
    config: ExperimentConfig,
    predictions: Path,
    argo_paths: list[Path],
    holdout: set[str],
    run_id: str,
    out_dir: Path | None = None,
    params: OIParams | None = None,
    per_depth: bool = False,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
    outlier_sigma: float | None = None,
) -> tuple[Path, Plan]:
    """Tune (unless ``params`` is given) and write the corrected store; returns its path and
    the per-depth plan used. ``min_depth_m`` and ``outlier_sigma`` are the guards described in
    the module docstring (both off by default)."""
    pred = xr.open_zarr(predictions)
    dates = pd.DatetimeIndex(pred["time"].values).normalize()
    check_no_test_leakage(dates, config.splits.test)
    val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
    if dates.size == 0 or dates.min() < val_start or dates.max() > val_end:
        raise ValueError("The prediction store must hold validation dates only")
    depths = np.asarray(pred["depth"].values, dtype=float)

    levels = read_levels(argo_paths, val_start, val_end)
    profiles = profile_levels(levels, [float(d) for d in depths], holdout)
    if profiles["platform_number"].isin(holdout).any():
        raise AssertionError("A held-out float reached the ARGO correction")
    res = residuals(profiles, pred)
    if np.isin(res.platform, list(holdout)).any():
        raise AssertionError("A held-out float reached the ARGO correction")
    logger.info("Residuals: %d profiles from %d non-held-out floats", len(res),
                np.unique(res.platform).size)  # fmt: skip
    if len(res) == 0:
        raise ValueError("No non-held-out ARGO profiles match the prediction store")
    if min_depth_m > 0:
        res = drop_shallow(res, depths, min_depth_m)
        logger.info("Levels shallower than %.1f m are not corrected", min_depth_m)
    if max_depth_m is not None:
        res = drop_deep(res, depths, max_depth_m)
        logger.info("Levels deeper than %.1f m are not corrected", max_depth_m)
    if outlier_sigma is not None:
        res, dropped = drop_outliers(res, outlier_sigma)
        logger.info("Outlier residuals dropped per depth (> %.1f robust sigma): %s",
                    outlier_sigma, dropped.tolist())  # fmt: skip
    if len(res) == 0:
        raise ValueError("No ARGO residuals left after the guards")
    for k, d in enumerate(depths):
        v = res.values[:, k]
        v = v[np.isfinite(v)]
        if v.size:
            logger.info("  %.1f m: mean residual (ARGO - model) %+.3f C, RMS %.3f C, n %d",
                        d, v.mean(), np.sqrt(np.mean(v**2)), v.size)  # fmt: skip

    out = Path(out_dir or config.paths.outputs)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    if params is None:
        plan, table, choices = tune(res, depths, per_depth=per_depth)
        table.round(6).to_csv(out / "metrics" / f"argo_correct_tuning_{run_id}.csv", index=False)
        if choices is not None:
            choices.round(6).to_csv(
                out / "metrics" / f"argo_correct_depths_{run_id}.csv", index=False
            )
    else:
        plan = uniform_plan(params, depths.size)
    if min_depth_m > 0:
        plan = tuple(None if d < min_depth_m else p for d, p in zip(depths, plan, strict=True))
    if max_depth_m is not None:
        plan = tuple(None if d > max_depth_m else p for d, p in zip(depths, plan, strict=True))
    logger.info("Using %s", plan_json(plan))
    dest = out / "predictions" / f"predictions_{run_id}.zarr"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_store(predictions, res, plan, run_id, dest)
    return dest, plan


def _load_holdout(path: Path) -> set[str]:
    floats = json.loads(Path(path).read_text(encoding="utf-8")).get("held_out_floats") or []
    if not floats:
        raise ValueError(f"{path} lists no held-out floats (PRD_AI_ML §7)")
    return {str(f).strip() for f in floats}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARGO residual correction of a prediction store.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--argo", nargs="+", required=True, help="Monthly ARGO NetCDF files")
    parser.add_argument("--holdout", required=True, help="argo_eval_floats.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    parser.add_argument(
        "--params", default=None, help="L_deg,tau_days,window_days,noise (skips tuning)"
    )
    parser.add_argument(
        "--per-depth", action="store_true", help="Tune one setting per depth (or none)"
    )
    parser.add_argument(
        "--min-depth", type=float, default=0.0, help="Never correct levels shallower than this (m)"
    )
    parser.add_argument(
        "--max-depth", type=float, default=None, help="Never correct levels deeper than this (m)"
    )
    parser.add_argument(
        "--outlier-sigma", type=float, default=None, help="Drop residuals beyond K robust sigma"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    params = None
    if args.params:
        a, b, c, d = args.params.split(",")
        params = OIParams(float(a), float(b), int(c), float(d))
    dest, used = run(
        config,
        Path(args.predictions),
        [Path(p) for p in args.argo],
        _load_holdout(Path(args.holdout)),
        args.run_id,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        params=params,
        per_depth=args.per_depth,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        outlier_sigma=args.outlier_sigma,
    )
    print(json.dumps({"store": str(dest), "plan": json.loads(plan_json(used))}))


if __name__ == "__main__":
    main()
