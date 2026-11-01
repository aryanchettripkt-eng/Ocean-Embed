"""ARGO profiles: monthly ERDDAP download and profiles on the model depths (PRD_AI_ML §5.9, §7).

``fetch_month`` downloads one month of profile levels inside the region from the Argo GDAC
ERDDAP (``ArgoFloats``) as CSV and writes a NetCDF file with one row per level and numeric QC
flags, the layout ``evaluation.argo_validation`` reads. Months that touch the sealed test period
are refused.

``profile_levels`` keeps QC flags 1-2, drops listed floats (the held-out evaluation floats), and
interpolates each profile linearly to the model depths, with no extrapolation. Pressure in dbar
is used as depth in metres, as in ``argo_validation``.
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig

logger = logging.getLogger("oceanscope_ml.data.argo")

ERDDAP_CSV = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
VARIABLES = (
    "platform_number",
    "cycle_number",
    "time",
    "latitude",
    "longitude",
    "pres",
    "temp",
    "temp_qc",
    "pres_qc",
    "position_qc",
)
NUMERIC = VARIABLES[1:2] + VARIABLES[3:]
QC_GOOD = (1, 2)
MAX_PRES_DBAR = 1100


def _refuse_test(month: pd.Period, config: ExperimentConfig) -> None:
    test_start = pd.Timestamp(config.splits.test[0])
    if month.end_time >= test_start:
        raise ValueError(f"ARGO month {month} reaches the sealed test period (PRD_AI_ML §0)")


def month_query(month: pd.Period, config: ExperimentConfig) -> str:
    region = config.region
    start = month.start_time.strftime("%Y-%m-%dT00:00:00Z")
    end = month.end_time.strftime("%Y-%m-%dT23:59:59Z")
    query = (
        f"{','.join(VARIABLES)}&time>={start}&time<={end}"
        f"&latitude>={region.lat_min}&latitude<={region.lat_max}"
        f"&longitude>={region.lon_min}&longitude<={region.lon_max}&pres<={MAX_PRES_DBAR}"
    )
    return f"{ERDDAP_CSV}?{urllib.parse.quote(query, safe='&,=')}"


def csv_to_dataset(text: str) -> xr.Dataset:
    """ERDDAP CSV (a header row, then a units row) to a NetCDF-ready dataset on dim ``row``."""
    if text.strip():
        df = pd.read_csv(StringIO(text), skiprows=[1], dtype={"platform_number": str})
    else:
        df = pd.DataFrame(columns=list(VARIABLES))
    df["platform_number"] = df["platform_number"].astype(str).str.strip()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    ds = xr.Dataset({c: ("row", df[c].to_numpy()) for c in VARIABLES})
    ds["platform_number"] = ds["platform_number"].astype(str)
    return ds


def fetch_month(
    month: pd.Period, config: ExperimentConfig, out_dir: Path, retries: int = 5
) -> Path:
    """Download one month to ``out_dir/argo_<YYYY-MM>.nc`` (kept and reused if it exists)."""
    _refuse_test(month, config)
    path = Path(out_dir) / f"argo_{month}.nc"
    if path.exists():
        return path
    url = month_query(month, config)
    text: str | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                text = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # ERDDAP's answer when no rows match
                text = ""
                break
            logger.warning("%s: HTTP %s, retry %d", month, exc.code, attempt + 1)
        except Exception as exc:  # noqa: BLE001 - network errors of any kind are retried
            logger.warning("%s: %s, retry %d", month, exc, attempt + 1)
        time.sleep(10 * 2**attempt)
    if text is None:
        raise RuntimeError(f"ERDDAP did not answer for {month}: {url}")
    ds = csv_to_dataset(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    ds.to_netcdf(tmp)
    tmp.rename(path)
    n_floats = len(set(ds["platform_number"].values.tolist()))
    logger.info("%s: %d levels, %d floats", month, ds.sizes["row"], n_floats)
    return path


def read_levels(paths: list[Path], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """QC 1-2 levels from the monthly files with start <= time < end + 1 day."""
    frames = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            df = ds[list(VARIABLES)].to_dataframe().reset_index(drop=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(VARIABLES))
    df = pd.concat(frames, ignore_index=True)
    df["platform_number"] = df["platform_number"].astype(str).str.strip()
    t = pd.to_datetime(df["time"])
    keep = (t >= start) & (t < end + pd.Timedelta(days=1))
    for col in ("temp_qc", "pres_qc", "position_qc"):
        keep &= pd.to_numeric(df[col], errors="coerce").isin(QC_GOOD)
    keep &= np.isfinite(df["temp"]) & np.isfinite(df["pres"]) & (df["pres"] >= 0)
    return df.loc[keep].reset_index(drop=True)


def profile_levels(
    levels: pd.DataFrame, depths_m: list[float], exclude_floats: set[str]
) -> pd.DataFrame:
    """One row per profile: position, day and temperature at each model depth (NaN outside).

    Columns: platform_number, cycle, day, lat, lon, t_0 ... t_<D-1>.
    """
    depths = np.asarray(depths_m, dtype=float)
    cols = [f"t_{i}" for i in range(depths.size)]
    df = levels[~levels["platform_number"].isin(exclude_floats)]
    rows: list[dict[str, Any]] = []
    for (pn, cycle), prof in df.groupby(["platform_number", "cycle_number"], sort=False):
        prof = prof.sort_values("pres").drop_duplicates("pres")
        if len(prof) < 2:
            continue
        z = prof["pres"].to_numpy(float)
        temp = prof["temp"].to_numpy(float)
        values = np.interp(depths, z, temp, left=np.nan, right=np.nan)
        if not np.isfinite(values).any():
            continue
        first = prof.iloc[0]
        row: dict[str, Any] = {
            "platform_number": str(pn),
            "cycle": int(cycle),
            "day": pd.Timestamp(first["time"]).normalize(),
            "lat": float(prof["latitude"].median()),
            "lon": float(prof["longitude"].median()),
        }
        row.update(zip(cols, values.tolist(), strict=True))
        rows.append(row)
    return pd.DataFrame(rows, columns=["platform_number", "cycle", "day", "lat", "lon", *cols])
