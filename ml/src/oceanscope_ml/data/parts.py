"""Work with a store that was built as several parts (PRD_AI_ML §4.2).

The full 1993-2021 build does not fit in one 12 h Kaggle run, so it runs as parallel
parts, each its own Zarr store holding a range of years. This module:

- opens the parts as one dataset holding only the dates each part actually wrote,
- fits the train-only climatology and normalization stats across all parts into a
  separate stats store (the part stores are read-only inputs on Kaggle), and
- checks every product for every built year, printing OK / WARN / FAIL.

Usage:
    python -m oceanscope_ml.data.parts stats --parts A.zarr B.zarr --out stats.zarr
    python -m oceanscope_ml.data.parts check --parts A.zarr B.zarr --stats stats.zarr
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.data.climatology import fit_harmonic_climatology
from oceanscope_ml.data.preprocess import compute_config_hash, load_config
from oceanscope_ml.data.transform import check_no_test_leakage, compute_norm_stats

logger = logging.getLogger("oceanscope_ml.data.parts")

# Variables each product writes, and the range a value must fall in to be plausible.
PRODUCT_VARS: dict[str, list[str]] = {
    "GLORYS": ["thetao"],
    "SLA": ["sla", "adt"],
    "SST": ["sst"],
    "SSS": ["sss"],
    "OSCAR": ["u_current", "v_current"],
    "ERA5": ["u_wind", "v_wind", "wind_stress_curl", "net_heat_flux", "precip"],
    "GREP": ["grep_mean"],
    "ARMOR3D": ["armor3d"],
}
PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "thetao": (-3.0, 40.0),
    "grep_mean": (-3.0, 40.0),
    "armor3d": (-3.0, 40.0),
    "sst": (-3.0, 40.0),
    "sss": (0.0, 45.0),
    "sla": (-2.0, 2.0),
    "adt": (-3.0, 3.0),
    "u_current": (-5.0, 5.0),
    "v_current": (-5.0, 5.0),
    "u_wind": (-60.0, 60.0),
    "v_wind": (-60.0, 60.0),
}
# Mean share of ocean cells with a value: below FAIL_COVERAGE fails, below WARN_COVERAGE warns.
FAIL_COVERAGE = 0.5
WARN_COVERAGE = 0.9
# Fewer distinct train days of year than this cannot pin down an annual cycle (smoke
# builds hold two weeks per year), so the stats step falls back to a mean-only fit.
MIN_DOY_FOR_HARMONICS = 180


def _read_built_years(ds: xr.Dataset) -> list[int]:
    return sorted(int(y) for y in ds.attrs.get("built_years", []))


def built_dates(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Dates a store actually wrote: its built years, cut to `built_ranges` when recorded.

    Stores built before `built_ranges` existed hold whole years, so a built year means
    every date of that year in the store. A store with no `built_years` at all holds only
    real dates and is returned whole.
    """
    store_dates = pd.to_datetime(ds.coords["time"].values)
    if "built_years" not in ds.attrs:
        # Written by preprocess.py, which only holds the dates it processed.
        return store_dates
    ranges = dict(ds.attrs.get("built_ranges", {}))
    keep = np.zeros(len(store_dates), dtype=bool)
    for yr in _read_built_years(ds):
        if str(yr) in ranges:
            first, last = (pd.Timestamp(d) for d in ranges[str(yr)])
            keep |= (store_dates >= first) & (store_dates <= last)
        else:
            keep |= store_dates.year == yr
    return store_dates[keep]


def open_parts(store_paths: list[Path], stats_path: Path | None = None) -> xr.Dataset:
    """Open one or more part stores as one lazy dataset over the dates they wrote.

    `clim` and `norm_stats` come from `stats_path` when given. Otherwise a single store
    keeps its own `clim` and `norm_stats` (a store finalized on its own).
    """
    if not store_paths:
        raise ValueError("No part stores given.")

    year_owner: dict[int, Path] = {}
    time_parts: list[xr.Dataset] = []
    static: xr.Dataset | None = None
    first: xr.Dataset | None = None
    for path in store_paths:
        ds = xr.open_zarr(path)
        for yr in _read_built_years(ds):
            if yr in year_owner:
                raise ValueError(
                    f"Year {yr} is built in both {year_owner[yr]} and {path}. "
                    "Each year must come from exactly one part."
                )
            year_owner[yr] = path
        dates = built_dates(ds)
        if len(dates) == 0:
            logger.warning("Part %s has no built dates; skipping it.", path)
            continue
        time_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
        time_parts.append(ds[time_vars].sel(time=dates))
        if static is None:
            first = ds
            static = ds[[v for v in ds.data_vars if "time" not in ds[v].dims and v != "clim"]]

    if not time_parts or static is None or first is None:
        raise ValueError(f"None of the part stores {store_paths} has any built dates.")

    combined = xr.concat(time_parts, dim="time") if len(time_parts) > 1 else time_parts[0]
    combined = combined.sortby("time")
    for v in static.data_vars:
        combined[v] = static[v]

    attrs = {k: v for k, v in first.attrs.items() if k not in ("norm_stats", "split_dates")}
    attrs["built_years"] = sorted(year_owner)
    attrs["parts"] = [str(p) for p in store_paths]
    if stats_path is not None:
        stats = xr.open_zarr(stats_path)
        combined["clim"] = stats["clim"]
        for key in ("norm_stats", "split_dates", "clim_harmonics"):
            if key in stats.attrs:
                attrs[key] = stats.attrs[key]
    elif len(store_paths) == 1 and "clim" in first.data_vars:
        combined["clim"] = first["clim"]
        for key in ("norm_stats", "split_dates"):
            if key in first.attrs:
                attrs[key] = first.attrs[key]
    combined.attrs = attrs
    return combined


def build_stats(
    config: Any,
    store_paths: list[Path],
    out_path: Path,
    n_harmonics: int = 3,
) -> Path:
    """Fit the train-only climatology and norm stats across all parts into `out_path`."""
    ds = open_parts(store_paths)
    train_start = pd.Timestamp(config.splits.train[0])
    train_end = pd.Timestamp(config.splits.train[1])
    check_no_test_leakage(config.splits.train, config.splits.test)

    t_3d = ds["thetao"].sel(time=slice(train_start, train_end))
    train_times = pd.to_datetime(t_3d.coords["time"].values)
    if len(train_times) == 0:
        raise ValueError(
            f"The parts hold no dates in the train split {config.splits.train}; "
            f"built years are {ds.attrs['built_years']}."
        )
    check_no_test_leakage(train_times, config.splits.test)

    n_doy = len(np.unique(train_times.day_of_year))
    if n_doy < MIN_DOY_FOR_HARMONICS and n_harmonics > 0:
        logger.warning(
            "Train dates cover only %d days of the year; fitting a mean-only climatology "
            "instead of %d harmonics. Expected for a smoke build, not for a full build.",
            n_doy,
            n_harmonics,
        )
        n_harmonics = 0

    logger.info(
        "Fitting climatology (%d harmonics) on %d train dates from years %s...",
        n_harmonics,
        len(train_times),
        sorted(set(train_times.year)),
    )
    clim_366 = fit_harmonic_climatology(
        t_3d,
        train_times,
        n_harmonics=n_harmonics,
        train_split=config.splits.train,
        test_split=config.splits.test,
    )
    clim_da = xr.DataArray(
        clim_366.astype(np.float32),
        dims=("dayofyear", "depth", "lat", "lon"),
        coords={
            "dayofyear": np.arange(1, 367),
            "depth": ds.coords["depth"].values,
            "lat": ds.coords["lat"].values,
            "lon": ds.coords["lon"].values,
        },
    )

    ds_with_clim = ds.copy()
    ds_with_clim["clim"] = clim_da
    logger.info("Computing normalization statistics on train dates...")
    norm_stats = compute_norm_stats(
        ds_with_clim,
        train_split=config.splits.train,
        test_split=config.splits.test,
    )

    out = clim_da.to_dataset(name="clim")
    out.attrs = {
        "dataset_version": str(config.dataset_version),
        "config_hash": compute_config_hash(config.raw_dict),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "built_years": ds.attrs["built_years"],
        "parts": ds.attrs["parts"],
        "train_years": sorted(int(y) for y in set(train_times.year)),
        "n_train_dates": int(len(train_times)),
        "clim_harmonics": int(n_harmonics),
        "norm_stats": norm_stats,
        "split_dates": {
            "train": list(config.splits.train),
            "val": list(config.splits.val),
            "test": list(config.splits.test),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_zarr(out_path, mode="w")
    logger.info("Wrote climatology and normalization statistics to %s", out_path)
    return out_path


def _row(item: str, year: int | str, status: str, detail: str) -> dict[str, Any]:
    return {"item": item, "year": year, "status": status, "detail": detail}


def _check_product_year(
    ds: xr.Dataset, product: str, yr: int, dates: pd.DatetimeIndex, ocean: np.ndarray
) -> dict[str, Any]:
    present = [v for v in PRODUCT_VARS[product] if v in ds.data_vars]
    if not present:
        return _row(product, yr, "FAIL", "no variables in store")
    worst_cov, worst_var = 1.0, present[0]
    problems: list[str] = []
    for v in present:
        da = ds[v].sel(time=dates)
        surf = da.isel(depth=0) if "depth" in da.dims else da
        vals = np.asarray(surf.values)[:, ocean]
        per_day = np.isfinite(vals).mean(axis=1) if vals.size else np.zeros(len(dates))
        empty_days = int(np.sum(per_day == 0.0))
        cov = float(per_day.mean()) if len(per_day) else 0.0
        if cov < worst_cov:
            worst_cov, worst_var = cov, v
        if empty_days:
            problems.append(f"{v}: {empty_days} empty day(s)")
        if v in PLAUSIBLE_RANGE and np.isfinite(vals).any():
            lo, hi = PLAUSIBLE_RANGE[v]
            vmin = float(da.min().compute())
            vmax = float(da.max().compute())
            if vmin < lo or vmax > hi:
                problems.append(f"{v}: range {vmin:.3g}..{vmax:.3g} outside {lo}..{hi}")
    detail = f"coverage {worst_cov:.0%}" + (f" ({worst_var})" if len(present) > 1 else "")
    if problems or worst_cov < FAIL_COVERAGE:
        return _row(product, yr, "FAIL", "; ".join([detail, *problems]))
    if worst_cov < WARN_COVERAGE:
        return _row(product, yr, "WARN", detail)
    return _row(product, yr, "OK", detail)


def check_parts(
    store_paths: list[Path],
    stats_path: Path | None = None,
    expected_years: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Check every product for every built year in every part, plus the stats store."""
    rows: list[dict[str, Any]] = []
    ref_mask: np.ndarray | None = None
    seen_years: set[int] = set()
    for path in store_paths:
        try:
            ds = xr.open_zarr(path)
        except Exception as e:
            rows.append(_row(f"part {path.name}", "-", "FAIL", f"cannot open: {e}"))
            continue
        if "ocean_mask" not in ds:
            rows.append(_row(f"part {path.name}", "-", "FAIL", "no ocean_mask"))
            continue
        ocean = np.asarray(ds["ocean_mask"].values, dtype=bool)
        if ref_mask is None:
            ref_mask = ocean
        elif not np.array_equal(ref_mask, ocean):
            rows.append(_row(f"part {path.name}", "-", "FAIL", "ocean_mask differs from part 1"))
        for static in ("bathymetry", "woa_clim_profile"):
            if static not in ds:
                rows.append(_row(static, "-", "FAIL", f"missing in {path.name}"))

        smoke_days = ds.attrs.get("smoke_days")
        dates_all = built_dates(ds)
        for yr in _read_built_years(ds):
            seen_years.add(yr)
            dates = dates_all[dates_all.year == yr]
            n_year = 366 if pd.Timestamp(f"{yr}-12-31").day_of_year == 366 else 365
            want = min(int(smoke_days), n_year) if smoke_days else n_year
            if len(dates) != want:
                rows.append(_row("dates", yr, "FAIL", f"{len(dates)} days written, want {want}"))
            else:
                rows.append(_row("dates", yr, "OK", f"{len(dates)} days"))
            if len(dates) == 0:
                continue
            for product in PRODUCT_VARS:
                rows.append(_check_product_year(ds, product, yr, dates, ocean))

    for yr in sorted(set(expected_years or []) - seen_years):
        rows.append(_row("year", yr, "FAIL", "not built in any part"))

    if stats_path is not None:
        try:
            stats = xr.open_zarr(stats_path)
            clim = np.asarray(stats["clim"].isel(depth=0, dayofyear=0).values)
            cov = float(np.isfinite(clim[ref_mask]).mean()) if ref_mask is not None else 0.0
            status = "OK" if cov >= WARN_COVERAGE else "FAIL"
            rows.append(
                _row(
                    "climatology",
                    "-",
                    status,
                    f"surface coverage {cov:.0%}, {stats.attrs.get('clim_harmonics')} harmonics",
                )
            )
            ns = stats.attrs.get("norm_stats", {})
            bad = [
                ch
                for ch, s in ns.get("channels", {}).items()
                # WOA stats are per depth (lists); the other channels are scalars.
                if not (
                    np.all(np.isfinite(s["mean"]))
                    and np.all(np.isfinite(s["std"]))
                    and np.all(np.asarray(s["std"]) > 0)
                )
            ]
            sigma = ns.get("sigma_d", [])
            n_ch = len(ns.get("channels", {}))
            if bad or not sigma or not all(np.isfinite(sigma)):
                rows.append(_row("norm stats", "-", "FAIL", f"bad channels {bad}"))
            else:
                rows.append(_row("norm stats", "-", "OK", f"{n_ch} channels, {len(sigma)} depths"))
        except Exception as e:
            rows.append(_row("stats store", "-", "FAIL", f"cannot read: {e}"))
    return rows


def print_check(rows: list[dict[str, Any]]) -> int:
    """Print the check table and return the number of FAIL rows."""
    print(f"{'item':<14} {'year':<6} {'status':<6} detail")
    print("-" * 72)
    for r in rows:
        print(f"{r['item']:<14} {str(r['year']):<6} {r['status']:<6} {r['detail']}")
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    n_warn = sum(r["status"] == "WARN" for r in rows)
    print("-" * 72)
    print("ALL OK" if n_fail == 0 else f"{n_fail} FAIL", f"({n_warn} WARN)")
    return n_fail


def main() -> None:
    """CLI entrypoint for oceanscope_ml.data.parts."""
    from oceanscope_ml.data.build_store import parse_year_spec

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="Fit climatology and norm stats across parts.")
    p_stats.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    p_stats.add_argument("--parts", type=Path, nargs="+", required=True)
    p_stats.add_argument("--out", type=Path, required=True)

    p_check = sub.add_parser("check", help="Print OK/WARN/FAIL per product and year.")
    p_check.add_argument("--parts", type=Path, nargs="+", required=True)
    p_check.add_argument("--stats", type=Path, default=None)
    p_check.add_argument("--years", type=str, default=None, help="Years that must be built.")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.cmd == "stats":
        build_stats(load_config(args.config), args.parts, args.out)
    else:
        years = parse_year_spec(args.years) if args.years else None
        n_fail = print_check(check_parts(args.parts, args.stats, years))
        sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
