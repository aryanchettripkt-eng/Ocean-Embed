"""Kaggle: ARGO check for one run (PRD_AI_ML §1.1, §7), then evaluate again and publish.

Run in a fresh notebook session (Internet on, secret HF_TOKEN attached) with these inputs:
  - the saved version of the training run (it holds outputs/ and stats.zarr),
  - the part stores (GREP and ARMOR3D are read from them at the ARGO points).
Settings: env OCEANSCOPE_RUN (default main2), OCEANSCOPE_PUBLISH=1 to upload at the end.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_argo_check.py

Only the validation years are downloaded (2018-2020). Nothing from the sealed 2021 test year is
requested. The model was trained with argo_input off, so no ARGO float was ever a model input;
the held-out floats are drawn here from the validation-year floats (20%, seed 42, by region).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
OUT = Path("/kaggle/working/outputs")
ARGO_DIR = Path("/kaggle/working/argo")
RUN = os.environ.get("OCEANSCOPE_RUN", "main2")
PUBLISH = os.environ.get("OCEANSCOPE_PUBLISH", "0") == "1"
ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
FIRST_MONTH, LAST_MONTH = "2018-01", "2020-12"  # validation years only, never 2021
VARS = "platform_number,cycle_number,time,latitude,longitude,pres,temp,temp_qc,pres_qc,position_qc"


def run(cmd, cwd=ML):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not os.path.isdir(REPO):
    run(
        f"git clone -b p3 https://github.com/aryanchettripkt-eng/ocean-embed.git {REPO}",
        cwd="/kaggle/working",
    )
else:
    run("git pull origin p3", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")
os.environ["PYTHONPATH"] = f"{ML}/src"
sys.path.insert(0, f"{ML}/src")
if PUBLISH:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.inspect import build_real_argo_holdout  # noqa: E402

CFG = "configs/experiment.yaml"
config = load_config(f"{ML}/{CFG}")
region = config.region

# 1) Find the part stores, the run's saved outputs and its stats in /kaggle/input.
stores, run_outputs, stats = [], None, None
for root, dirs, _ in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    if root.endswith("oceanscope_v1.zarr"):
        stores.append(root)
        dirs[:] = []
    elif root.endswith(f"predictions_{RUN}.zarr"):
        run_outputs = Path(root).parent.parent
        dirs[:] = []
    elif root.endswith("stats.zarr"):
        stats = root
        dirs[:] = []
    elif root.endswith(".zarr"):
        dirs[:] = []
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Run outputs:", run_outputs)
print("Stats:", stats)
if not stores or run_outputs is None or stats is None:
    raise SystemExit("Attach the part stores and the saved version of the training run as inputs.")

# Copy the run's outputs to a writable place (evaluate rewrites the manifest there).
if not (OUT / "predictions" / f"predictions_{RUN}.zarr").exists():
    shutil.copytree(run_outputs, OUT, dirs_exist_ok=True)
PRED = OUT / "predictions" / f"predictions_{RUN}.zarr"
print("Copied outputs:", sorted(p.name for p in OUT.iterdir()))


# 2) Download the ARGO profiles month by month from the Argo GDAC ERDDAP (CSV, then a
#    NetCDF file per month with plain numeric QC flags, the layout argo_validation reads).
def fetch_month(month: pd.Period) -> Path:
    nc_path = ARGO_DIR / f"argo_{month}.nc"
    if nc_path.exists():
        return nc_path
    start = month.start_time.strftime("%Y-%m-%dT00:00:00Z")
    end = month.end_time.strftime("%Y-%m-%dT23:59:59Z")
    query = (
        f"{VARS}&time>={start}&time<={end}"
        f"&latitude>={region.lat_min}&latitude<={region.lat_max}"
        f"&longitude>={region.lon_min}&longitude<={region.lon_max}&pres<=1100"
    )
    url = f"{ERDDAP}?{urllib.parse.quote(query, safe='&,=')}"
    text = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                text = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # ERDDAP's answer for "no rows match"
                text = ""
                break
            print(f"  {month}: HTTP {exc.code}, retry {attempt + 1}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {month}: {exc}, retry {attempt + 1}", flush=True)
        time.sleep(10 * 2**attempt)
    if text is None:
        raise RuntimeError(f"ERDDAP did not answer for {month}: {url}")
    if text:
        df = pd.read_csv(StringIO(text), skiprows=[1], dtype={"platform_number": str})
    else:
        df = pd.DataFrame(columns=VARS.split(","))
    df["platform_number"] = df["platform_number"].astype(str).str.strip()
    for col in (
        "cycle_number",
        "latitude",
        "longitude",
        "pres",
        "temp",
        "temp_qc",
        "pres_qc",
        "position_qc",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    ds = xr.Dataset({c: ("row", df[c].to_numpy()) for c in df.columns})
    ds["platform_number"] = ds["platform_number"].astype(str)
    tmp = nc_path.with_suffix(".tmp")
    ds.to_netcdf(tmp)
    tmp.rename(nc_path)
    print(f"  {month}: {len(df):,} levels, {df['platform_number'].nunique()} floats", flush=True)
    return nc_path


ARGO_DIR.mkdir(parents=True, exist_ok=True)
months = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
argo_files = [fetch_month(m) for m in months]
if any(m.year >= 2021 for m in months):
    raise SystemExit("The 2021 test year is sealed")

# 3) Held-out floats: 20% by region, seed 42 (PRD §7). Kept if it already exists, so
#    reruns use the same floats.
holdout_path = ARGO_DIR / "argo_eval_floats.json"
if not holdout_path.exists():
    # One row per profile (not per level) keeps the builder fast and counts profiles, not levels.
    cols = ["platform_number", "cycle_number", "time", "latitude", "longitude"]
    prof = []
    for f in argo_files:
        with xr.open_dataset(f) as ds:
            df = ds[cols].to_dataframe()
            prof.append(df.drop_duplicates(["platform_number", "cycle_number"]))
    prof = pd.concat(prof, ignore_index=True)
    prof_path = ARGO_DIR / "profiles_summary.nc"
    xr.Dataset({c: ("row", prof[c].to_numpy()) for c in cols}).to_netcdf(prof_path)
    holdout = build_real_argo_holdout([prof_path], seed=int(config.train.get("seed", 42)))
    holdout["source"] = f"Argo GDAC ERDDAP ArgoFloats, {FIRST_MONTH}..{LAST_MONTH}, QC not filtered"
    holdout_path.write_text(json.dumps(holdout, indent=2), encoding="utf-8")
holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
print(
    "Held-out floats:",
    holdout.get("total_held_out"),
    "of",
    holdout.get("total_floats"),
    "by region:",
    holdout.get("held_out_by_region"),
)
if not holdout.get("held_out_floats"):
    raise SystemExit("No held-out floats; check the ARGO download above.")

# 4) ARGO validation, then evaluate again so the manifest lists the ARGO files.
S = " ".join(stores)
FILES = " ".join(str(p) for p in argo_files)
run(
    f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --argo {FILES} --holdout {holdout_path} --out-dir {OUT}"
)
run(f"cat {OUT}/metrics/argo_{RUN}.csv")
run(f"cat {OUT}/metrics/{RUN}_headline.csv")
run(
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --out-dir {OUT}"
)
shutil.copy(holdout_path, OUT / "metrics" / "argo_eval_floats.json")

# 5) Publish (dry run always; upload only with OCEANSCOPE_PUBLISH=1).
run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT} --dry-run")
if PUBLISH:
    run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT}")
print("\nARGO CHECK DONE")
