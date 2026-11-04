"""ARGO residual correction of a prediction store (objective analysis of the model's error).

Usage (from ml/):

    python -m oceanscope_ml.inference.argo_correct --config configs/experiment.yaml \
        --predictions outputs/predictions/predictions_<run_id>.zarr \
        --argo <argo_YYYY-MM.nc...> --holdout configs/argo_eval_floats.json \
        --run-id <run_id>_oi [--out-dir outputs] [--params L,TAU,WINDOW,NOISE]

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


def score(fit: Residuals, tune: Residuals, params: OIParams | None, depths: np.ndarray) -> dict:
    """RMSE of the corrected model at the tuning profiles; ``params=None`` scores the model."""
    err = tune.values.copy()
    if params is not None:
        for d in np.unique(tune.day):
            idx = np.flatnonzero(tune.day == d)
            err[idx] -= correction(fit, params, int(d), tune.lat[idx], tune.lon[idx])
    finite = np.isfinite(err)
    band = finite & _band_mask(tune.lat, tune.lon, depths)
    return {
        "rmse_c": float(np.sqrt(np.mean(err[finite] ** 2))) if finite.any() else np.nan,
        "bob_band_rmse_c": float(np.sqrt(np.mean(err[band] ** 2))) if band.any() else np.nan,
        "n_obs": int(finite.sum()),
        "bob_band_n_obs": int(band.sum()),
    }


def tune(res: Residuals, depths: np.ndarray) -> tuple[OIParams, pd.DataFrame]:
    """Grid search on the tuning floats; returns the best setting and the full table."""
    tune_floats = split_floats(res)
    is_tune = np.isin(res.platform, list(tune_floats))
    fit, held = res.subset(~is_tune), res.subset(is_tune)
    logger.info(
        "Tuning: %d profiles from %d floats as residuals, %d profiles from %d floats scored",
        len(fit), np.unique(fit.platform).size, len(held), len(tune_floats),
    )  # fmt: skip
    rows = [{"length_deg": 0.0, "tau_days": 0.0, "window_days": 0, "noise": 0.0}]
    rows[0].update(score(fit, held, None, depths))
    logger.info("No correction: RMSE %.3f C, BoB 50-300 m %.3f C", rows[0]["rmse_c"],
                rows[0]["bob_band_rmse_c"])  # fmt: skip
    for values in itertools.product(*GRID.values()):
        params = OIParams(*values)  # type: ignore[arg-type]
        row: dict = asdict(params)
        row.update(score(fit, held, params, depths))
        rows.append(row)
        logger.info("%s: RMSE %.3f C, BoB 50-300 m %.3f C", params, row["rmse_c"],
                    row["bob_band_rmse_c"])  # fmt: skip
    table = pd.DataFrame(rows, columns=TUNING_COLUMNS)
    grid = table.iloc[1:]
    best = grid.loc[grid["bob_band_rmse_c"].idxmin()]
    params = OIParams(
        float(best["length_deg"]), float(best["tau_days"]), int(best["window_days"]),
        float(best["noise"]),
    )  # fmt: skip
    return params, table


def write_store(pred_path: Path, res: Residuals, params: OIParams, run_id: str, dest: Path) -> Path:
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
            c = correction(res, params, int(days[b0 + k]), lat2[valid], lon2[valid])
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
            "argo_correction": json.dumps(asdict(params)),
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
) -> tuple[Path, OIParams]:
    """Tune (unless ``params`` is given) and write the corrected store; returns its path."""
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
    for k, d in enumerate(depths):
        v = res.values[:, k]
        v = v[np.isfinite(v)]
        if v.size:
            logger.info("  %.1f m: mean residual (ARGO - model) %+.3f C, RMS %.3f C, n %d",
                        d, v.mean(), np.sqrt(np.mean(v**2)), v.size)  # fmt: skip

    out = Path(out_dir or config.paths.outputs)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    if params is None:
        params, table = tune(res, depths)
        table.round(6).to_csv(out / "metrics" / f"argo_correct_tuning_{run_id}.csv", index=False)
    logger.info("Using %s", params)
    dest = out / "predictions" / f"predictions_{run_id}.zarr"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_store(predictions, res, params, run_id, dest)
    return dest, params


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
    )
    print(json.dumps({"store": str(dest), "params": asdict(used)}))


if __name__ == "__main__":
    main()
