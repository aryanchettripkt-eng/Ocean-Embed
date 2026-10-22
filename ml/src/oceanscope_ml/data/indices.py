"""Climate indices retrieval and processing (PRD_AI_ML §3, §4.1).

Covers Niño 3.4 (NOAA PSL), MJO RMM1/RMM2 (BoM), and DMI (Dipole Mode Index).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.data"
MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"


def fetch_nino34(url: str = NINO34_URL) -> pd.DataFrame:
    """Fetch and parse monthly/daily Niño 3.4 index from NOAA PSL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Niño 3.4 from {url}: {exc}") from exc

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame(columns=["date", "nino34"])

    # Typical PSL table format: Header line with start/end years,
    # then Year followed by 12 monthly values
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 13 and parts[0].isdigit():
            year = int(parts[0])
            for month in range(1, 13):
                val_str = parts[month]
                try:
                    val = float(val_str)
                    if val > -90.0:  # Missing values often -99.99
                        date_str = f"{year}-{month:02d}-15"
                        rows.append({"date": date_str, "nino34": val})
                except ValueError:
                    continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_mjo(url: str = MJO_URL) -> pd.DataFrame:
    """Fetch and parse Wheeler-Hendon daily MJO RMM1 and RMM2 from BoM."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch MJO from {url}: {exc}") from exc

    # Header is typically 2 lines, followed by: year, month, day, RMM1, RMM2, phase, amplitude, ...
    rows = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].isdigit():
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                rmm1 = float(parts[3])
                rmm2 = float(parts[4])
                if rmm1 > -90 and rmm2 > -90:
                    rows.append(
                        {
                            "date": f"{year}-{month:02d}-{day:02d}",
                            "mjo_rmm1": rmm1,
                            "mjo_rmm2": rmm2,
                        }
                    )
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_dmi_from_sst(
    sst_west: pd.Series,
    sst_east: pd.Series,
    clim_west: float = 28.0,
    clim_east: float = 28.5,
) -> pd.Series:
    """Compute Dipole Mode Index (IOD): West box anomaly minus East box anomaly.

    West box: 50-70°E, 10°S-10°N.
    East box: 90-110°E, 10°S-0°N.
    """
    anom_west = sst_west - clim_west
    anom_east = sst_east - clim_east
    return anom_west - anom_east


def save_indices(
    out_dir: Path | str,
    nino_df: pd.DataFrame | None = None,
    mjo_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Save processed index CSVs to out_dir."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    results = {}

    if nino_df is not None and not nino_df.empty:
        nino_path = p / "nino34.csv"
        nino_df.to_csv(nino_path, index=False)
        results["nino34"] = str(nino_path)

    if mjo_df is not None and not mjo_df.empty:
        mjo_path = p / "mjo.csv"
        mjo_df.to_csv(mjo_path, index=False)
        results["mjo"] = str(mjo_path)

    return results
