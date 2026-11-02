"""Export the static fields (bathymetry, WOA monthly profile) from a built store.

The result is a small netCDF that ``build_store --static-file`` reads, so a streamed
build doesn't depend on the GEBCO and WOA servers.

Usage:
    python -m oceanscope_ml.data.export_static --store data/processed/oceanscope_v1.zarr \
        --out data/processed/static_fields.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

STATIC_VARS = ("bathymetry", "woa_clim_profile")


def export_static(store: Path, out: Path) -> Path:
    """Write bathymetry and woa_clim_profile from ``store`` to the netCDF ``out``."""
    ds = xr.open_zarr(store)
    missing = [v for v in STATIC_VARS if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Store {store} has no {missing}")
    static = ds[list(STATIC_VARS)].load()
    if bool(np.isnan(static["bathymetry"].values).all()):
        raise ValueError(f"bathymetry in {store} is all NaN")
    if bool(np.isnan(static["woa_clim_profile"].values).all()):
        raise ValueError(f"woa_clim_profile in {store} is all NaN")
    static.attrs = {"source_store": str(store)}
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in STATIC_VARS}
    out.parent.mkdir(parents=True, exist_ok=True)
    static.to_netcdf(out, encoding=encoding)
    return out


def main() -> None:
    """CLI entry point for oceanscope_ml.data.export_static."""
    parser = argparse.ArgumentParser(description="Export static fields for build_store")
    parser.add_argument("--store", required=True, help="Path to an existing oceanscope Zarr store")
    parser.add_argument("--out", required=True, help="Output netCDF path")
    args = parser.parse_args()
    out = export_static(Path(args.store), Path(args.out))
    print(f"[OK] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
