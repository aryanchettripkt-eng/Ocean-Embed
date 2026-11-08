"""Kaggle: fetch ARMOR3D for training years, as input side stores (DECISIONS 2026-11-08).

The part stores hold ARMOR3D only for 2018-2021 (DECISIONS 2026-10-28). A model with ARMOR3D
input channels also needs it on its training days, so this writes one
``/kaggle/working/armor3d/armor3d_<year>.zarr`` per year: every OCEANSCOPE_ARMOR_STRIDE-th day
(default 3), training dates only, reduced to the 0.25 degree grid and 15 depths like the part
stores. A year already written in an attached earlier version of this notebook is copied, not
fetched again.

Run with Save Version -> "Save & Run All (Commit)". No GPU. Internet on. Secrets:
COPERNICUSMARINE_SERVICE_USERNAME, COPERNICUSMARINE_SERVICE_PASSWORD.
Run as: %env OCEANSCOPE_ARMOR_YEARS=2004-2010, then
%run /kaggle/working/ocean-embed/ml/scripts/kaggle_armor3d_fetch.py

Settings (env): OCEANSCOPE_ARMOR_YEARS (a range like 2004-2010, or a list like 2004,2005),
OCEANSCOPE_ARMOR_STRIDE (default 3).
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
OUT = Path("/kaggle/working/armor3d")
SPEC = os.environ.get("OCEANSCOPE_ARMOR_YEARS", "").strip()
STRIDE = int(os.environ.get("OCEANSCOPE_ARMOR_STRIDE", "3"))
SECRETS = ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")

if "-" in SPEC:
    a, b = (int(x) for x in SPEC.split("-", 1))
    YEARS = list(range(a, b + 1))
else:
    YEARS = [int(x) for x in SPEC.split(",") if x.strip()]
if not YEARS:
    raise SystemExit("Set OCEANSCOPE_ARMOR_YEARS, e.g. %env OCEANSCOPE_ARMOR_YEARS=2004-2010")


def run(cmd, cwd=ML):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not os.path.isdir(REPO):
    run(
        "git clone -b ml/argo-improvements "
        f"https://github.com/aryanchettripkt-eng/ocean-embed.git {REPO}",
        cwd="/kaggle/working",
    )
else:  # pull the branch the notebook cloned, never merge another one into it
    run("git pull --ff-only origin $(git rev-parse --abbrev-ref HEAD)", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")
sys.path.insert(0, f"{ML}/src")

from kaggle_secrets import UserSecretsClient  # noqa: E402

sec = UserSecretsClient()
for key in SECRETS:
    os.environ[key] = sec.get_secret(key)

import logging  # noqa: E402

import xarray as xr  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.armor_inputs import COMPLETE_ATTR, build_side_year  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
config = load_config(f"{ML}/configs/experiment.yaml")
OUT.mkdir(parents=True, exist_ok=True)

# Years finished in an attached earlier version are copied instead of fetched again.
earlier = {}
for root, dirs, _ in os.walk("/kaggle/input"):
    name = os.path.basename(root)
    if name.startswith("armor3d_") and name.endswith(".zarr"):
        earlier[name] = Path(root)
        dirs[:] = []

for year in YEARS:
    name = f"armor3d_{year}.zarr"
    if name in earlier and xr.open_zarr(earlier[name]).attrs.get(COMPLETE_ATTR):
        if xr.open_zarr(earlier[name]).attrs.get("stride_days") == STRIDE:
            shutil.copytree(earlier[name], OUT / name, dirs_exist_ok=True)
            print(f"ARMOR3D {year}: copied from {earlier[name]}", flush=True)
            continue
    start = time.perf_counter()
    build_side_year(config, year, STRIDE, OUT)
    print(f"ARMOR3D {year}: done in {(time.perf_counter() - start) / 60:.0f} min", flush=True)

for year in YEARS:
    side = xr.open_zarr(OUT / f"armor3d_{year}.zarr")
    print(f"armor3d_{year}.zarr: {side.sizes['time']} days, attrs ok: {side.attrs[COMPLETE_ATTR]}")
print(f"\nARMOR3D FETCH DONE for {YEARS} (stride {STRIDE})")
