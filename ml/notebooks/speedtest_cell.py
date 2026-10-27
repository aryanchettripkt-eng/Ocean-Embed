# Speed test: lazy open_dataset (current code) vs copernicusmarine.subset, one month of GLORYS.
import time
from pathlib import Path

import copernicusmarine
import numpy as np

DEPTHS = [0.494, 5.078, 9.573, 18.496, 29.445, 47.374, 65.807, 92.326, 130.666,
          186.126, 222.475, 266.040, 380.213, 643.567, 1062.440]
common = dict(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao"],
    minimum_longitude=39.125, maximum_longitude=100.875,
    minimum_latitude=-0.875, maximum_latitude=30.875,
    start_datetime="2017-01-01", end_datetime="2017-01-31",
    minimum_depth=0.494, maximum_depth=1062.44,
)

# A) current method: lazy open, select 15 levels, load
t0 = time.time()
ds = copernicusmarine.open_dataset(**common)
dep = ds["depth"].values
idx = [int(np.argmin(np.abs(dep - d))) for d in DEPTHS]
arr = ds["thetao"].isel(depth=idx).values
ta = time.time() - t0
print(f"A) lazy open_dataset, 15 levels: {ta:.0f} s, {arr.nbytes / 1e9:.2f} GB in memory")

# B) subset: download one file (all levels in range), then load 15 levels
out = Path("/tmp/speedtest")
out.mkdir(exist_ok=True)
t0 = time.time()
copernicusmarine.subset(**common, output_directory=str(out), output_filename="glorys_jan2017.nc")
tb_dl = time.time() - t0
import xarray as xr  # noqa: E402

dsb = xr.open_dataset(out / "glorys_jan2017.nc")
arrb = dsb["thetao"].isel(depth=idx).values
tb = time.time() - t0
size = (out / "glorys_jan2017.nc").stat().st_size / 1e9
print(f"B) subset download {tb_dl:.0f} s ({size:.2f} GB file), total with load {tb:.0f} s")
print(f"Speed-up B vs A: {ta / tb:.1f}x  -> paste these 3 lines to Claude")
