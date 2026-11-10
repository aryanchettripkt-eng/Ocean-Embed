"""Kaggle: what near-real-time ARMOR3D exists, and how close is it to the reprocessed one?

Step 1 of running main3_aix30a live (thread "Continue model improvements against ARGO",
2026-11-10). The model was trained on the daily reprocessed (MY) ARMOR3D. Live mode needs a
near-real-time (NRT) feed. This script reads nothing from the test year and writes nothing that
training or the live site uses; it only prints:

1. every dataset of MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012 with its time coverage, time step,
   grid spacing, depths and variables;
2. for each NRT dataset, the latest days available (the delay against today);
3. on days where NRT and MY overlap: NRT minus MY over the Bay of Bengal at 50-300 m
   (RMSE and bias per depth), after putting NRT on the MY grid. That is the size of the input
   change the model would see in live mode.

Run with Save Version -> "Save & Run All (Commit)". No GPU. Internet on. Secrets:
COPERNICUSMARINE_SERVICE_USERNAME, COPERNICUSMARINE_SERVICE_PASSWORD.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_armor_nrt_check.py
"""

import datetime as dt
import json
import os
import subprocess
import sys

import numpy as np

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
PRODUCT = "MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012"
MY_DATASET = "cmems_obs-mob_glo_phy_my_0.125deg_P1D-m"  # what the model was trained on
BOX = dict(minimum_latitude=5.0, maximum_latitude=22.0, minimum_longitude=80.0,
           maximum_longitude=95.0)  # Bay of Bengal  # fmt: skip
SECRETS = ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")


def run(cmd, cwd=ML):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if os.path.isdir(REPO):
    run("git pull --ff-only origin $(git rev-parse --abbrev-ref HEAD)", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")

try:
    from kaggle_secrets import UserSecretsClient

    sec = UserSecretsClient()
    for name in SECRETS:
        os.environ[name] = sec.get_secret(name)
except Exception as exc:  # noqa: BLE001
    print("Secrets not read:", exc)

import copernicusmarine as cm  # noqa: E402

print("copernicusmarine", cm.__version__ if hasattr(cm, "__version__") else "")


# ---------------------------------------------------------------- 1. catalogue
def _ms(v):
    try:
        return dt.datetime.fromtimestamp(float(v) / 1000, dt.UTC).date().isoformat()
    except (TypeError, ValueError):
        return v


def _note_coordinate(info, c):
    cid = c.get("coordinate_id")
    if cid == "time" and info["time"] is None:
        lo, hi = _ms(c.get("minimum_value")), _ms(c.get("maximum_value"))
        info["time"] = (lo, hi, c.get("step"), c.get("coordinate_unit"))
    if cid in ("latitude", "lat") and info["lat_step"] is None:
        info["lat_step"] = c.get("step")
    if cid in ("depth", "elevation") and info["depths"] is None:
        vals = c.get("values") or []
        info["depths"] = len(vals) or (c.get("minimum_value"), c.get("maximum_value"))


cat = cm.describe(product_id=PRODUCT)
data = json.loads(cat.model_dump_json()) if hasattr(cat, "model_dump_json") else cat
datasets = {}
for product in data.get("products", []):
    for d in product.get("datasets", []):
        did = d.get("dataset_id")
        info = {"variables": set(), "time": None, "lat_step": None, "depths": None}
        for ver in d.get("versions", []):
            for part in ver.get("parts", []):
                for svc in part.get("services", []):
                    for var in svc.get("variables", []):
                        info["variables"].add(var.get("short_name"))
                        for c in var.get("coordinates", []):
                            _note_coordinate(info, c)
        datasets[did] = info

print("\n=== 1. Datasets in", PRODUCT)
for did, info in datasets.items():
    print(f"\n{did}")
    print("  variables:", sorted(v for v in info["variables"] if v))
    print("  time (start, end, step, unit):", info["time"])
    print("  latitude step:", info["lat_step"], " depths:", info["depths"])

nrt_ids = [d for d in datasets if "nrt" in d.lower()]
print("\nNRT datasets:", nrt_ids or "NONE")
if not nrt_ids:
    raise SystemExit("No NRT dataset found: send me this log.")


def temp_var(did):
    names = [v for v in datasets[did]["variables"] if v]
    for want in ("to", "thetao", "temperature", "temp"):
        if want in names:
            return want
    raise SystemExit(f"{did}: no temperature variable in {names}")


def open_box(did, start, end, depth_max=1100.0):
    return cm.open_dataset(dataset_id=did, variables=[temp_var(did)], start_datetime=start,
                           end_datetime=end, minimum_depth=0.0, maximum_depth=depth_max,
                           **BOX)  # fmt: skip


# ---------------------------------------------------------------- 2. latest NRT days
today = dt.date.today()
print(f"\n=== 2. Latest NRT days (today {today})")
nrt_info = {}
for did in nrt_ids:
    try:
        ds = open_box(did, (today - dt.timedelta(days=60)).isoformat(), today.isoformat())
    except Exception as exc:  # noqa: BLE001
        print(f"{did}: could not open the last 60 days: {exc}")
        continue
    times = np.asarray(ds["time"].values).astype("datetime64[D]")
    if len(times) == 0:
        print(f"{did}: no days in the last 60 days")
        continue
    last = times[-1].astype(dt.date)
    gaps = np.unique(np.diff(times).astype(int)) if len(times) > 1 else []
    lat = np.asarray(ds["latitude"].values)
    print(f"{did}: {len(times)} time steps in the last 60 days, last {last} "
          f"(delay {(today - last).days} days), spacing between steps {list(gaps)} days, "
          f"grid {abs(lat[1] - lat[0]):.3f} deg, {ds.sizes.get('depth')} depths")  # fmt: skip
    print("  depths (m):", np.round(np.asarray(ds["depth"].values), 1).tolist())
    nrt_info[did] = ds

# ---------------------------------------------------------------- 3. NRT vs MY overlap
print("\n=== 3. NRT minus MY on overlapping days, Bay of Bengal")
my_end = datasets.get(MY_DATASET, {}).get("time")
print("MY coverage:", my_end)
for did in nrt_ids:
    t = datasets[did]["time"]
    if not t or not my_end:
        print(f"{did}: no coverage info, skipped")
        continue
    lo = max(dt.date.fromisoformat(str(t[0])[:10]), dt.date.fromisoformat(str(my_end[0])[:10]))
    hi = min(dt.date.fromisoformat(str(t[1])[:10]), dt.date.fromisoformat(str(my_end[1])[:10]))
    if lo > hi:
        print(f"{did}: no overlap with MY ({t[0]}..{t[1]} vs {my_end[0]}..{my_end[1]})")
        continue
    start = max(lo, hi - dt.timedelta(days=90))
    print(f"{did}: overlap {lo}..{hi}; comparing {start}..{hi}")
    try:
        nrt = open_box(did, start.isoformat(), hi.isoformat(), 320.0)
        my = open_box(MY_DATASET, start.isoformat(), hi.isoformat(), 320.0)
    except Exception as exc:  # noqa: BLE001
        print("  could not open:", exc)
        continue
    nv, mv = nrt[temp_var(did)], my[temp_var(MY_DATASET)]
    step = np.diff(np.asarray(nrt["time"].values).astype("datetime64[D]")).astype(int)
    weekly = len(step) and int(np.median(step)) >= 7
    rows = []
    for tt in np.asarray(nrt["time"].values)[:12]:
        a = nv.sel(time=tt)
        if weekly:  # compare with the MY mean of the 7 days centred on the NRT stamp
            b = mv.sel(time=slice(tt - np.timedelta64(3, "D"), tt + np.timedelta64(3, "D")))
            b = b.mean("time")
        else:
            b = mv.sel(time=tt, method="nearest")
        a = a.interp(latitude=b["latitude"], longitude=b["longitude"], depth=b["depth"])
        d = (a - b).sel(depth=slice(45, 305))
        rows.append(d)
    if not rows:
        print("  no matching days")
        continue
    import xarray as xr

    diff = xr.concat(rows, "t")
    rmse = np.sqrt((diff**2).mean(("t", "latitude", "longitude"), skipna=True))
    bias = diff.mean(("t", "latitude", "longitude"), skipna=True)
    print(f"  {len(rows)} NRT steps compared ({'weekly vs 7-day MY mean' if weekly else 'daily'})")
    for z, r, b_ in zip(diff["depth"].values, rmse.values, bias.values, strict=True):
        print(f"  {z:7.1f} m  RMSE {r:.3f} C  bias {b_:+.3f} C")
    allr = float(np.sqrt(np.nanmean(diff.values**2)))
    print(f"  Bay of Bengal 50-300 m, all: RMSE {allr:.3f} C")

print("\nARMOR3D NRT CHECK DONE (nothing written, nothing published)")
