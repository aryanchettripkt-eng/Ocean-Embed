"""Where does each run go wrong? ARGO errors split by ocean regime (evaluation only).

Usage (from ml/):

    python -m oceanscope_ml.evaluation.regimes \
        --matches argo_matches_<run_a>_dev.parquet argo_matches_<run_b>_dev.parquet ... \
        --events ../backend/app/data/events.json --run-id <label> [--out-dir outputs] \
        [--no-fetch]

Each ``--matches`` file is one run's ARGO match table (``evaluation.argo_validation``), on the
same floats. The first file also gives GLORYS12, GREP and ARMOR3D. Every observation gets:

- ``season``: SW monsoon (Jun-Sep), NE monsoon (Dec-Feb), inter-monsoon (Mar-May, Oct-Nov)
- ``iod``: positive (DMI >= 0.4), negative (<= -0.4) or neutral, from the NOAA PSL monthly DMI
- ``enso``: El Nino (Nino 3.4 >= 0.5), La Nina (<= -0.5) or neutral, from NOAA PSL monthly
- ``cyclone``: inside a curated cyclone's box (+2 deg) from its start to 7 days after its end
- ``anomaly``: warm / normal / cold tercile of the observed temperature against the mean of all
  observations at that level and calendar month in the file

The indices are downloaded (Kaggle: Internet on); with ``--no-fetch`` or when a download
fails, those regimes are "unknown". For each regime value it reports RMSE (°C) per product, overall
and in the Bay of Bengal at 50-300 m, with the number of observations and floats. Regimes are
for reading the errors; nothing here picks a run or a setting.

Writes ``<out>/metrics/regimes_<run_id>.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from oceanscope_ml.evaluation.argo_validation import (
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    _in_bay_of_bengal,
)

logger = logging.getLogger("oceanscope_ml.evaluation.regimes")

DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
IOD_THRESHOLD = 0.4
ENSO_THRESHOLD = 0.5
CYCLONE_PAD_DEG = 2.0
CYCLONE_AFTER_DAYS = 7
KEYS = ["platform_number", "cycle", "time", "depth_m", "temp_obs_c"]
REFERENCES = {"glorys12": "temp_glorys_c", "grep": "temp_grep_c", "armor3d": "temp_armor3d_c"}
COLUMNS = ["regime", "value", "group", "product", "n_obs", "n_floats", "rmse_c", "bias_c"]


def parse_psl_monthly(text: str) -> pd.Series:
    """A NOAA PSL monthly table (year + 12 values per line; -99.99-like values missing)."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 13 and parts[0].isdigit() and len(parts[0]) == 4:
            year = int(parts[0])
            for m, v in enumerate(parts[1:], start=1):
                try:
                    x = float(v)
                except ValueError:
                    continue
                if x > -90:
                    out[pd.Period(year=year, month=m, freq="M")] = x
    return pd.Series(out, dtype=float)


def fetch_index(url: str) -> pd.Series | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            series = parse_psl_monthly(resp.read().decode("utf-8", errors="replace"))
        return series if not series.empty else None
    except Exception as exc:  # the regimes are optional; the run must not fail on a download
        logger.warning("Could not fetch %s: %s", url, exc)
        return None


def season(month: pd.Series) -> pd.Series:
    labels = {6: "sw_monsoon", 7: "sw_monsoon", 8: "sw_monsoon", 9: "sw_monsoon",
              12: "ne_monsoon", 1: "ne_monsoon", 2: "ne_monsoon"}  # fmt: skip
    return month.map(labels).fillna("inter_monsoon")


def phase(index: pd.Series | None, periods: pd.Series, thr: float, pos: str, neg: str) -> pd.Series:
    if index is None:
        return pd.Series("unknown", index=periods.index)
    values = periods.map(index)
    out = np.where(values >= thr, pos, np.where(values <= -thr, neg, "neutral"))
    return pd.Series(np.where(values.isna(), "unknown", out), index=periods.index)


def cyclone_flag(df: pd.DataFrame, events: list[dict]) -> pd.Series:
    flag = pd.Series("no", index=df.index)
    t = pd.to_datetime(df["time"]).dt.normalize()
    for e in events:
        if e.get("kind") != "cyclone":
            continue
        box = e["bbox"]
        start = pd.Timestamp(e["start"])
        end = pd.Timestamp(e["end"]) + pd.Timedelta(days=CYCLONE_AFTER_DAYS)
        inside = (
            (t >= start) & (t <= end)
            & (df["lat"] >= box["lat_min"] - CYCLONE_PAD_DEG)
            & (df["lat"] <= box["lat_max"] + CYCLONE_PAD_DEG)
            & (df["lon"] >= box["lon_min"] - CYCLONE_PAD_DEG)
            & (df["lon"] <= box["lon_max"] + CYCLONE_PAD_DEG)
        )  # fmt: skip
        flag[inside] = "yes"
    return flag


def anomaly_tercile(df: pd.DataFrame) -> pd.Series:
    month = pd.to_datetime(df["time"]).dt.month
    mean = df.groupby([df["depth_m"], month])["temp_obs_c"].transform("mean")
    anom = df["temp_obs_c"] - mean
    lo, hi = np.nanpercentile(anom, [100 / 3, 200 / 3])
    return pd.Series(np.where(anom <= lo, "cold", np.where(anom >= hi, "warm", "normal")),
                     index=df.index)  # fmt: skip


def combine(files: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """One row per observation all runs cover, with one prediction column per run and the
    references from the first file."""
    names = list(files)
    base = None
    for k, (name, m) in enumerate(files.items()):
        df = m.copy()
        df["platform_number"] = df["platform_number"].astype(str)
        df["dup"] = df.groupby(KEYS).cumcount()
        keep = KEYS + ["dup", "temp_pred_c"]
        if k == 0:
            keep += ["lat", "lon"] + [c for c in REFERENCES.values() if c in df]
        df = df[keep].rename(columns={"temp_pred_c": name})
        base = df if base is None else base.merge(df, on=KEYS + ["dup"], how="inner")
    assert base is not None
    products = names + [p for p, c in REFERENCES.items() if c in base]
    base = base.rename(columns={c: p for p, c in REFERENCES.items()})
    return base.reset_index(drop=True), products


def label(df: pd.DataFrame, events: list[dict], dmi: pd.Series | None,
          nino: pd.Series | None) -> pd.DataFrame:  # fmt: skip
    t = pd.to_datetime(df["time"])
    periods = t.dt.to_period("M")
    df = df.copy()
    df["season"] = season(t.dt.month)
    df["iod"] = phase(dmi, periods, IOD_THRESHOLD, "positive", "negative")
    df["enso"] = phase(nino, periods, ENSO_THRESHOLD, "el_nino", "la_nina")
    df["cyclone"] = cyclone_flag(df, events)
    df["anomaly"] = anomaly_tercile(df)
    return df


def table(df: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    band = (
        _in_bay_of_bengal(df["lat"].to_numpy(float), df["lon"].to_numpy(float))
        & (df["depth_m"] >= HEADLINE_DEPTH_MIN_M)
        & (df["depth_m"] <= HEADLINE_DEPTH_MAX_M)
    )
    rows = []
    for regime in ("season", "iod", "enso", "cyclone", "anomaly"):
        for value in sorted(df[regime].unique()):
            for group, sel in (("all", np.ones(len(df), bool)), ("bob_50_300", band)):
                sub = df[(df[regime] == value).to_numpy() & sel]
                for p in products:
                    err = (sub[p] - sub["temp_obs_c"]).to_numpy(float)
                    err = err[np.isfinite(err)]
                    rows.append(
                        {
                            "regime": regime,
                            "value": value,
                            "group": group,
                            "product": p,
                            "n_obs": int(err.size),
                            "n_floats": int(sub["platform_number"].nunique()),
                            "rmse_c": float(np.sqrt(np.mean(err**2))) if err.size else np.nan,
                            "bias_c": float(err.mean()) if err.size else np.nan,
                        }
                    )
    return pd.DataFrame(rows, columns=COLUMNS)


def run(
    match_paths: list[Path], events_path: Path | None, run_id: str, out_dir: Path, fetch: bool
) -> Path:
    files = {}
    for p in match_paths:
        name = p.stem.removeprefix("argo_matches_")
        files[name] = pd.read_parquet(p)
    df, products = combine(files)
    events = []
    if events_path is not None and events_path.exists():
        events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
    dmi = fetch_index(DMI_URL) if fetch else None
    nino = fetch_index(NINO34_URL) if fetch else None
    out = table(label(df, events, dmi, nino), products)
    dest = out_dir / "metrics" / f"regimes_{run_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(dest, index=False)
    return dest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARGO errors split by ocean regime.")
    parser.add_argument("--matches", nargs="+", required=True)
    parser.add_argument("--events", default=None, help="events.json with curated cyclones")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--no-fetch", action="store_true", help="Skip the index downloads")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    print(
        run(
            [Path(p) for p in args.matches],
            Path(args.events) if args.events else None,
            args.run_id,
            Path(args.out_dir),
            fetch=not args.no_fetch,
        )
    )


if __name__ == "__main__":
    main()
