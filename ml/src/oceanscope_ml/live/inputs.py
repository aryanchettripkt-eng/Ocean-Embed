"""Input fields for live mode and the NRT check (PRD_AI_ML §12).

Each input product is fetched for a range of days, reduced to the 0.25° target grid with the
same reduce functions the training store used (``oceanscope_ml.data.reduce``), and checked
day by day. A day is usable only if every input the model reads exists for that day and for
its whole history window; otherwise it is skipped with the reason (no partial maps).

The products and dataset IDs come from ``live.sources`` (near-real-time, live mode) or
``live_check.reprocessed`` (the reprocessed products the training store used) in
``configs/live.yaml``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.channels import channel_list
from oceanscope_ml.data.reduce import (
    TargetGrid,
    reduce_adt,
    reduce_currents,
    reduce_era5_fluxes,
    reduce_era5_winds,
    reduce_sla,
    reduce_sss,
    reduce_sst,
)

logger = logging.getLogger("oceanscope_ml.live.inputs")

# Share of ocean cells that must hold a value for a day of one input to count as present.
# Same threshold as the store check's FAIL level (oceanscope_ml.data.parts.FAIL_COVERAGE).
MIN_OCEAN_FRACTION = 0.5

# Store variables each product provides.
PRODUCT_VARS: dict[str, tuple[str, ...]] = {
    "sla_adt": ("sla", "adt"),
    "sst": ("sst",),
    "sss": ("sss",),
    "currents": ("u_current", "v_current"),
    "era5": ("u_wind", "v_wind", "wind_stress_curl", "net_heat_flux", "precip"),
}
SERVICES = ("copernicusmarine", "earthdata", "cds")
# Newest ERA5T days to give up on, one at a time, when CDS refuses a request.
CDS_MAX_DROPPED_DAYS = 3

# fetch(product, spec, start, end) -> raw dataset, or None when the source has no data then.
Fetcher = Callable[[str, dict[str, Any], pd.Timestamp, pd.Timestamp], "xr.Dataset | None"]


@dataclass
class InputCheck:
    """Which days of each input are present, and the chosen target day."""

    coverage: pd.DataFrame  # index: day, columns: store variable, value: ocean fraction valid
    last_day: dict[str, str | None]  # newest day each variable is present
    target: pd.Timestamp | None
    skipped: dict[str, list[str]]  # newer day -> why it was skipped


def target_grid(config: ExperimentConfig) -> TargetGrid:
    r = config.region
    return TargetGrid.from_bounds(
        r.lon_min, r.lon_max, r.lat_min, r.lat_max, float(config.grid.resolution_deg)
    )


def model_input_vars(config: ExperimentConfig) -> list[str]:
    """Daily store variables the model reads, in config order (core then forcing)."""
    inputs = config.inputs
    if inputs.get("indices_enabled"):
        raise NotImplementedError(
            "Live inputs for the climate indices are not built (inputs.indices_enabled is on)"
        )
    if (inputs.get("argo_input") or {}).get("enabled"):
        raise NotImplementedError(
            "Live ARGO input channels are not built (inputs.argo_input.enabled is on)"
        )
    return [str(v) for v in list(inputs.get("core", [])) + list(inputs.get("forcing", []))]


def var_lookbacks(config: ExperimentConfig) -> dict[str, int]:
    """Days of history before the target day that each daily variable's channels read."""
    lookback: dict[str, int] = {v: 0 for v in model_input_vars(config)}
    for c in channel_list(config):
        if c.kind in ("dynamic", "long_history"):
            lookback[c.source] = max(lookback.get(c.source, 0), c.lag + c.window_days - 1)
    return lookback


def products_for(variables: list[str], sources: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The products (from ``sources``) that provide ``variables``; fails if one is missing."""
    needed: dict[str, dict[str, Any]] = {}
    for var in variables:
        product = next((p for p, vs in PRODUCT_VARS.items() if var in vs), None)
        if product is None:
            raise KeyError(f"No live product provides input variable {var!r}")
        if product not in sources:
            raise KeyError(f"Config has no source for product {product!r} (needed for {var!r})")
        spec = dict(sources[product])
        if spec.get("service") not in SERVICES:
            raise ValueError(f"Source {product!r} has unknown service {spec.get('service')!r}")
        if not spec.get("dataset_id"):
            raise ValueError(f"Source {product!r} has no dataset_id")
        needed[product] = spec
    return needed


def reduce_product(
    product: str,
    raw: xr.Dataset | None,
    grid: TargetGrid,
    days: pd.DatetimeIndex,
    ocean_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce one product's raw fields to [day, lat, lon] arrays on the target grid."""
    shape = (len(days), len(grid.lats), len(grid.lons))
    if raw is None or raw.sizes.get("time", raw.sizes.get("valid_time", 1)) == 0:
        return {v: np.full(shape, np.nan, dtype=np.float32) for v in PRODUCT_VARS[product]}
    t = days.values
    if product == "sla_adt":
        out = {"sla": reduce_sla(raw, grid, target_time=t)["sla"].values}
        out["adt"] = reduce_adt(raw, grid, target_time=t)["adt"].values
    elif product == "sst":
        out = {"sst": reduce_sst(raw, grid, target_time=t)["sst"].values}
    elif product == "sss":
        out = {"sss": reduce_sss(raw, grid, target_time=t)["sss"].values}
    elif product == "currents":
        red = reduce_currents(raw, grid, target_time=t)
        out = {"u_current": red["u_current"].values, "v_current": red["v_current"].values}
    elif product == "era5":
        winds = reduce_era5_winds(raw, grid, target_time=t, ocean_mask=ocean_mask)
        flux = reduce_era5_fluxes(raw, grid, target_time=t)
        out = {v: winds[v].values for v in ("u_wind", "v_wind", "wind_stress_curl")}
        out.update({v: flux[v].values for v in ("net_heat_flux", "precip")})
    else:
        raise KeyError(f"Unknown product {product!r}")
    return {k: np.asarray(v, dtype=np.float32).reshape(shape) for k, v in out.items()}


def load_inputs(
    config: ExperimentConfig,
    sources: dict[str, Any],
    days: pd.DatetimeIndex,
    ocean_mask: np.ndarray,
    fetch: Fetcher,
    product_end: dict[str, pd.Timestamp] | None = None,
) -> xr.Dataset:
    """Every model input variable on ``days`` (NaN where a source has no data).

    ``product_end`` caps the last day requested from a product (ERA5 on CDS errors when asked
    for days it does not have yet).
    """
    grid = target_grid(config)
    variables = model_input_vars(config)
    fields: dict[str, Any] = {}
    for product, spec in products_for(variables, sources).items():
        end = days[-1]
        if product_end and product in product_end:
            end = min(end, product_end[product])
        raw = None
        if end >= days[0]:
            logger.info(
                "Fetching %s (%s) %s..%s", product, spec["dataset_id"], days[0].date(), end.date()
            )
            raw = fetch(product, spec, days[0], end)
        for var, values in reduce_product(product, raw, grid, days, ocean_mask).items():
            if var in variables:
                fields[var] = (("time", "lat", "lon"), values)
    return xr.Dataset(fields, coords={"time": days, "lat": grid.lats, "lon": grid.lons})


def daily_coverage(inputs: xr.Dataset, ocean_mask: np.ndarray) -> pd.DataFrame:
    """Share of ocean cells with a value, per day and variable."""
    n_ocean = max(int(ocean_mask.sum()), 1)
    cols = {}
    for var in inputs.data_vars:
        values = np.asarray(inputs[var].values)
        cols[str(var)] = np.isfinite(values[:, ocean_mask]).sum(axis=1) / n_ocean
    return pd.DataFrame(cols, index=pd.DatetimeIndex(inputs["time"].values))


def check_inputs(
    coverage: pd.DataFrame,
    lookbacks: dict[str, int],
    newest: pd.Timestamp,
    oldest: pd.Timestamp,
) -> InputCheck:
    """Pick the newest day in [oldest, newest] whose inputs and histories are all present."""
    present = coverage >= MIN_OCEAN_FRACTION
    last_day: dict[str, str | None] = {}
    for var in coverage.columns:
        days_present = coverage.index[present[var].values]
        last_day[str(var)] = str(days_present.max().date()) if len(days_present) else None

    skipped: dict[str, list[str]] = {}
    day = newest
    while day >= oldest:
        missing: list[str] = []
        for var, lb in lookbacks.items():
            need = pd.date_range(day - pd.Timedelta(days=lb), day, freq="D")
            lacking = [d for d in need if d not in present.index or not bool(present.at[d, var])]
            if lacking:
                if lacking[-1] == day:
                    missing.append(f"{var} missing on the day")
                else:
                    missing.append(f"{var} missing {len(lacking)} history day(s)")
        if not missing:
            return InputCheck(coverage, last_day, day, skipped)
        skipped[str(day.date())] = missing
        day -= pd.Timedelta(days=1)
    return InputCheck(coverage, last_day, None, skipped)


def write_window_store(
    path: Path,
    inputs: xr.Dataset,
    static: xr.Dataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Path:
    """A small store over [start, end] that ``OceanScopeDataset(split="live")`` reads.

    It holds the daily inputs plus the static fields of the training store. ``clim`` and
    ``norm_stats`` come from the kit's stats store when the dataset opens it.
    """
    window = inputs.sel(time=slice(start, end))
    expected = pd.date_range(start, end, freq="D")
    if not pd.DatetimeIndex(window["time"].values).equals(expected):
        raise ValueError(f"Inputs do not cover every day {start.date()}..{end.date()}")
    for v in ("lat", "lon"):
        if not np.allclose(window[v].values, static[v].values):
            raise ValueError(f"Kit static fields are on a different {v} grid than the config")
    ds = window.copy()
    for v in ("ocean_mask", "bathymetry", "woa_clim_profile"):
        ds[v] = static[v]
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", zarr_format=2, consolidated=True)
    return path


# ---------------------------------------------------------------- real fetchers


def fetch_remote(
    product: str,
    spec: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bbox: tuple[float, float, float, float],
    scratch: Path,
) -> xr.Dataset | None:
    """Fetch one product's raw fields for [start, end] from its service.

    Credentials come from the environment (COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD,
    EARTHDATA_USERNAME/PASSWORD, and the CDS key in ~/.cdsapirc or CDSAPI_URL/CDSAPI_KEY).
    """
    service = spec["service"]
    if service == "copernicusmarine":
        return _fetch_copernicus(spec, start, end, bbox)
    if service == "earthdata":
        return _fetch_earthdata(spec, start, end, bbox, scratch / product)
    if service == "cds":
        return _fetch_cds_daily(spec, start, end, bbox, scratch / product)
    raise ValueError(f"Unknown service {service!r} for {product}")


def _fetch_copernicus(
    spec: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset | None:
    import copernicusmarine

    from oceanscope_ml.data.build_store import _load_with_retries

    lon_min, lon_max, lat_min, lat_max = bbox
    kwargs: dict[str, Any] = {
        "dataset_id": spec["dataset_id"],
        "minimum_longitude": lon_min - 1.0,
        "maximum_longitude": lon_max + 1.0,
        "minimum_latitude": lat_min - 1.0,
        "maximum_latitude": lat_max + 1.0,
    }
    if spec.get("variables"):
        kwargs["variables"] = list(spec["variables"])
    if spec.get("dataset_version"):
        kwargs["dataset_version"] = str(spec["dataset_version"])
    if spec.get("max_depth") is not None:
        kwargs["maximum_depth"] = float(spec["max_depth"])
    # Opened lazily without a time range, then cut to the days that exist: asking the
    # toolbox for days past the end of an NRT product is an error, not an empty result.
    ds = copernicusmarine.open_dataset(**kwargs)
    times = pd.to_datetime(ds["time"].values).normalize()
    keep = np.flatnonzero((times >= start) & (times <= end))
    if len(keep) == 0:
        logger.warning("%s has no days in %s..%s", spec["dataset_id"], start.date(), end.date())
        return None
    return _load_with_retries(ds.isel(time=keep), spec["dataset_id"])


def _fetch_earthdata(
    spec: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bbox: tuple[float, float, float, float],
    out_dir: Path,
) -> xr.Dataset | None:
    import earthaccess

    from oceanscope_ml.data.build_store import _with_retries

    out_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy="environment")
    short_name = spec["dataset_id"]
    results = _with_retries(
        lambda: earthaccess.search_data(
            short_name=short_name,
            temporal=(str(start.date()), str(end.date())),
            bounding_box=(bbox[0], bbox[2], bbox[1], bbox[3]),
        ),
        f"{short_name} search",
    )
    if not results:
        logger.warning("%s has no granules in %s..%s", short_name, start.date(), end.date())
        return None
    files = _with_retries(
        lambda: earthaccess.download(results, str(out_dir)), f"{short_name} download"
    )
    if len(files) == 1:
        return xr.open_dataset(files[0])
    return xr.open_mfdataset(sorted(files), combine="nested", concat_dim="time")


def _fetch_cds_daily(
    spec: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bbox: tuple[float, float, float, float],
    out_dir: Path,
) -> xr.Dataset | None:
    import cdsapi

    from oceanscope_ml.data.build_store import ERA5_DAILY_VARIABLES

    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    # ERA5T's newest day is not always out at the expected lag: CDS refuses the whole request
    # when one day is missing. Drop the newest day and ask again, a few times, before failing.
    parts: list[xr.Dataset] = []
    for drop in range(CDS_MAX_DROPPED_DAYS + 1):
        last = end - pd.Timedelta(days=drop)
        if last < start:
            break
        try:
            parts = [
                _cds_retrieve(client, spec, request, out_dir, ERA5_DAILY_VARIABLES)
                for request in cds_requests(start, last, bbox)
            ]
            break
        except Exception as e:
            if drop == CDS_MAX_DROPPED_DAYS or last - pd.Timedelta(days=1) < start:
                raise
            logger.warning("CDS refused days up to %s (%s); trying a day earlier", last.date(), e)
    if not parts:
        return None
    time_dim = "valid_time" if "valid_time" in parts[0].dims else "time"
    ds = xr.concat(parts, dim=time_dim) if len(parts) > 1 else parts[0]
    ds.attrs["daily_statistic"] = "daily_mean"
    ds.attrs["is_daily_mean"] = True
    return ds


def _cds_retrieve(
    client: Any, spec: dict[str, Any], request: dict[str, Any], out_dir: Path, variables: list
) -> xr.Dataset:
    from oceanscope_ml.data.build_store import _open_cds_download

    y, m, d = request["year"], request["month"], request["day"]
    target = out_dir / f"era5_{y}_{m}_{d[0]}-{d[-1]}.nc"
    if not (target.exists() and target.stat().st_size > 0):
        logger.info("CDS request %s-%s days %s", y, m, d)
        client.retrieve(spec["dataset_id"], {**request, "variable": variables}, str(target))
    return _open_cds_download(target)


def cds_requests(
    start: pd.Timestamp, end: pd.Timestamp, bbox: tuple[float, float, float, float]
) -> list[dict[str, Any]]:
    """One ERA5 daily-statistics request per calendar month in [start, end]."""
    lon_min, lon_max, lat_min, lat_max = bbox
    days = pd.date_range(start, end, freq="D")
    requests = []
    for (year, month), group in pd.Series(days, index=days).groupby([days.year, days.month]):
        requests.append(
            {
                "product_type": "reanalysis",
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d.day:02d}" for d in group],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": [lat_max + 1.0, lon_min - 1.0, lat_min - 1.0, lon_max + 1.0],
                "format": "netcdf",
            }
        )
    return requests
