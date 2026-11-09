"""ARMOR3D as an input to the U-Net (DECISIONS 2026-11-08, Aryan's choice on the card).

PRD_AI_ML §3 lists ARMOR3D as an external baseline only. On 2026-11-08 Aryan chose to try it
as an input, to correct an observation-based analysis instead of reconstructing from the surface
alone. ARMOR3D assimilates every ARGO float, the held-out ones included, so a model with these
channels compares with GLORYS12 on equal terms (both have seen the floats) but no longer
independently of the floats or of ARMOR3D itself.

Channels (``channels._armor_channels``, after the ARGO ones): ``armor3d_anom@<d>m`` for the
15 depths, the ARMOR3D temperature minus the training climatology, divided by ``sigma_d``
(the same scaling as the targets), and ``armor3d_mask`` (1 where ARMOR3D has a value).

Where the values come from:
- validation and test years: the ``armor3d`` variable of the part stores (built for those
  years only, DECISIONS 2026-10-28). The test year is read only with ``allow_test``.
- training years: side stores written by ``python -m oceanscope_ml.data.armor_inputs build``
  (one ``armor3d_<year>.zarr`` per year, every ``stride``-th day, training dates only).

Usage (from ml/, needs COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD):

    python -m oceanscope_ml.data.armor_inputs build --config configs/experiment.yaml \
        --years 2004 2005 --stride 3 --out-dir /kaggle/working/armor3d
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config

logger = logging.getLogger("oceanscope_ml.data.armor_inputs")

ARMOR_PREFIX = "armor3d_"
ARMOR_MASK = "armor3d_mask"
SIDE_ORIGIN = pd.Timestamp("1993-01-01")  # day stride counts from here, so parts line up
COMPLETE_ATTR = "armor3d_side_complete"


def enable_armor_input(config: ExperimentConfig) -> None:
    """Turn the ARMOR3D input channels on in a loaded config (``inputs.armor3d_input``)."""
    armor_cfg = dict(config.inputs.get("armor3d_input", {}) or {})
    armor_cfg["enabled"] = True
    config.inputs["armor3d_input"] = armor_cfg


def is_armor_channel(name: str) -> bool:
    return str(name).startswith(ARMOR_PREFIX)


def side_store_paths(paths: list[Path]) -> list[Path]:
    """Side stores from files or folders: a folder contributes its ``armor3d_<year>.zarr``."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.name.startswith("armor3d_") and p.suffix == ".zarr":
            out.append(p)
        elif p.is_dir():
            out += sorted(p.glob("armor3d_*.zarr"))
    return sorted(set(out))


def _armor_ranges(config: ExperimentConfig, allow_test: bool) -> list[tuple[pd.Timestamp, ...]]:
    """Date ranges whose part-store ``armor3d`` is filled: validation, plus test with allow_test."""
    splits = [config.splits.val] + ([config.splits.test] if allow_test else [])
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in splits]


class ArmorInputs:
    """ARMOR3D planes for one day: ``[n_depth + 1, lat, lon]`` (anomalies, then the mask)."""

    def __init__(
        self,
        config: ExperimentConfig,
        ds: xr.Dataset,
        sigma_d: np.ndarray,
        side_paths: list[Path] | None = None,
        allow_test: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.n_depth = len(config.depths_m)
        self.sigma = np.asarray(sigma_d, dtype=np.float64)[:, None, None]
        if self.sigma.shape[0] != self.n_depth:
            raise ValueError("sigma_d must have one value per depth")
        if "armor3d" not in ds or "clim" not in ds:
            raise KeyError("The store needs armor3d and clim (attach the stats store)")
        self._lat = np.asarray(ds["lat"].values)
        self._lon = np.asarray(ds["lon"].values)
        val_start = pd.Timestamp(config.splits.val[0])
        test_start = pd.Timestamp(config.splits.test[0])

        times = pd.to_datetime(ds["time"].values).normalize()
        ranges = _armor_ranges(config, allow_test)
        self._main: dict[pd.Timestamp, int] = {}
        for t, day in enumerate(times):
            if any(a <= day <= b for a, b in ranges) and (allow_test or day < test_start):
                self._main[day] = t

        self.side_paths = side_store_paths(list(side_paths or []))
        self._side: dict[pd.Timestamp, tuple[int, int]] = {}
        for k, path in enumerate(self.side_paths):
            side = xr.open_zarr(path)
            if not side.attrs.get(COMPLETE_ATTR):
                raise ValueError(f"{path} is not a finished ARMOR3D side store")
            if not (
                np.allclose(np.asarray(side["lat"].values), self._lat)
                and np.allclose(np.asarray(side["lon"].values), self._lon)
            ):
                raise ValueError(f"{path} is on another grid than the part stores")
            if not np.allclose(np.asarray(side["depth"].values), np.asarray(config.depths_m)):
                raise ValueError(f"{path} has other depths than the config")
            side_times = pd.to_datetime(side["time"].values).normalize()
            if len(side_times) and side_times.max() >= val_start:
                raise ValueError(f"{path} reaches the validation period (side stores: train only)")
            for t, day in enumerate(side_times):
                self._side.setdefault(day, (k, t))
        for day in self._side:
            self._main.pop(day, None)
        self._build_cache(ds, cache_dir)
        logger.info(
            "ARMOR3D inputs: %d days from the part stores, %d from %d side stores",
            len(self._main),
            len(self._side),
            len(self.side_paths),
        )

    def _build_cache(self, ds: xr.Dataset, cache_dir: Path | None) -> None:
        """Write every day's planes once into a float16 memory map (run M stalled without it).

        Reading one day from Zarr decompresses whole chunks of ``clim``, the part stores and
        the side stores (seconds per sample), and the DataLoader workers then shared lazy dask
        handles. The cache reads each source once, in order, so a sample is one small slice.
        """
        days = sorted(set(self._side) | set(self._main))
        self._row = {day: i for i, day in enumerate(days)}
        folder = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp())
        folder.mkdir(parents=True, exist_ok=True)
        self._cache_path = folder / f"armor3d_planes_{os.getpid()}_{id(self)}.npy"
        self._planes: np.ndarray | None = None
        shape = (max(len(days), 1), self.n_depth + 1, len(self._lat), len(self._lon))
        mm = np.lib.format.open_memmap(self._cache_path, mode="w+", dtype=np.float16, shape=shape)
        if days:
            start = time.perf_counter()
            clim = np.asarray(ds["clim"].values, dtype=np.float32)  # [dayofyear, depth, lat, lon]
            sigma = self.sigma.astype(np.float32)

            def write(day: pd.Timestamp, raw: np.ndarray, out: np.ndarray = mm) -> None:
                anomaly = (raw.astype(np.float32) - clim[day.day_of_year - 1]) / sigma
                ok = np.isfinite(anomaly)
                row = out[self._row[day]]
                row[: self.n_depth] = np.where(ok, anomaly, 0.0)
                row[self.n_depth] = ok.any(axis=0)

            by_store: dict[int, list[tuple[int, pd.Timestamp]]] = {}
            for day, (k, t) in self._side.items():
                by_store.setdefault(k, []).append((t, day))
            for k, items in by_store.items():
                values = np.asarray(xr.open_zarr(self.side_paths[k])["armor3d"].values)
                for t, day in items:
                    write(day, values[t])
                del values
            blocks: list[list[tuple[int, pd.Timestamp]]] = []
            for t, day in sorted((t, day) for day, t in self._main.items()):
                if not blocks or t - blocks[-1][0][0] >= 32:  # 32 store days per read
                    blocks.append([])
                blocks[-1].append((t, day))
            for block in blocks:
                lo, hi = block[0][0], block[-1][0] + 1
                values = np.asarray(ds["armor3d"].isel(time=slice(lo, hi)).values)
                for t, day in block:
                    write(day, values[t - lo])
            del clim
            logger.info(
                "ARMOR3D planes cached for %d days in %.1f min (%s)",
                len(days),
                (time.perf_counter() - start) / 60.0,
                self._cache_path,
            )
        mm.flush()
        del mm

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_planes"] = None  # the memory map reopens in each worker
        return state

    def has(self, day: pd.Timestamp) -> bool:
        return pd.Timestamp(day).normalize() in self._row

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self._row))

    def planes(self, day: pd.Timestamp) -> np.ndarray:
        """Anomaly planes and the mask; all zero (mask 0) on a day without ARMOR3D."""
        day = pd.Timestamp(day).normalize()
        if day not in self._row:
            return np.zeros((self.n_depth + 1, len(self._lat), len(self._lon)), dtype=np.float32)
        if self._planes is None:
            self._planes = np.load(self._cache_path, mmap_mode="r")
        return np.asarray(self._planes[self._row[day]], dtype=np.float32)


# ------------------------------------------------------------------ side stores


def side_days(config: ExperimentConfig, year: int, stride: int) -> pd.DatetimeIndex:
    """Every ``stride``-th day of ``year`` (counted from 1993-01-01) inside the training split."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    start, end = (pd.Timestamp(d) for d in config.splits.train)
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    days = days[(days >= start) & (days <= end)]
    return days[((days - SIDE_ORIGIN).days % stride) == 0]


def build_side_year(
    config: ExperimentConfig, year: int, stride: int, out_dir: Path, source: Any = None
) -> Path:
    """Write ``armor3d_<year>.zarr`` (training days only); skips a year already finished.

    ``source`` (tests) replaces the Copernicus stream with a dataset in the raw ARMOR3D layout.
    """
    from oceanscope_ml.data.build_store import stream_copernicus
    from oceanscope_ml.data.download import PRODUCT_SPECS
    from oceanscope_ml.data.reduce import TargetGrid, reduce_armor3d

    days = side_days(config, year, stride)
    if len(days) == 0:
        raise ValueError(f"No training days in {year}: ARMOR3D side stores are train-only")
    dest = Path(out_dir) / f"armor3d_{year}.zarr"
    if dest.exists():
        try:
            if xr.open_zarr(dest).attrs.get(COMPLETE_ATTR):
                logger.info("ARMOR3D %d already written: %s", year, dest)
                return dest
        except (OSError, ValueError, KeyError):
            pass
        shutil.rmtree(dest)
    grid = TargetGrid.from_bounds(
        lon_min=config.region.lon_min,
        lon_max=config.region.lon_max,
        lat_min=config.region.lat_min,
        lat_max=config.region.lat_max,
        resolution_deg=config.grid.resolution_deg,
    )
    depths = [float(d) for d in config.depths_m]
    if source is None:
        source = stream_copernicus(
            dataset_id=PRODUCT_SPECS["armor3d"]["dataset_id"],
            bbox=(
                float(grid.lons.min()),
                float(grid.lons.max()),
                float(grid.lats.min()),
                float(grid.lats.max()),
            ),
            dates=(str(days[0].date()), str(days[-1].date())),
            variables=["to"],
            depths=(0.0, 1100.0),
            bracket_levels=depths,
        )
    src_days = pd.to_datetime(source["time"].values).normalize()
    keep = np.flatnonzero(src_days.isin(days))
    if len(keep) == 0:
        raise ValueError(f"ARMOR3D returned none of the {len(days)} days asked for {year}")
    logger.info("ARMOR3D %d: %d of %d days (stride %d)", year, len(keep), len(days), stride)
    reduced = reduce_armor3d(source.isel(time=keep), grid, depths)
    reduced = reduced.assign_coords(time=src_days[keep].values)
    reduced.attrs.update(
        {
            "product": PRODUCT_SPECS["armor3d"]["dataset_id"],
            "year": int(year),
            "stride_days": int(stride),
            "note": "ARMOR3D input for training years only (DECISIONS 2026-11-08)",
        }
    )
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_zarr(tmp, mode="w")
    import zarr

    zarr.open_group(str(tmp), mode="r+").attrs[COMPLETE_ATTR] = True
    zarr.consolidate_metadata(str(tmp))
    tmp.rename(dest)
    logger.info("Wrote %s", dest)
    return dest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write ARMOR3D side stores for training years")
    b.add_argument("--config", default="configs/experiment.yaml")
    b.add_argument("--years", type=int, nargs="+", required=True)
    b.add_argument("--stride", type=int, default=3, help="keep every N-th day (default 3)")
    b.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    for year in args.years:
        build_side_year(config, year, args.stride, args.out_dir)


if __name__ == "__main__":
    main()
