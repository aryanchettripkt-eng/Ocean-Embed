"""Kaggle: ensemble of saved runs plus per-depth ARGO correction, and their ARGO checks (no GPU).

Run with Save Version -> "Save & Run All (Commit)". Inputs: the part stores, the saved main2
training version (stats.zarr) and the saved versions that hold each member's validation
predictions (predictions_<member>.zarr). Internet on (only used for ARGO files not attached).
Never attach the 2021 final-test version: test dates are refused.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_ensemble_correct.py

Settings (env):
- OCEANSCOPE_MEMBERS: comma-separated runs to average (default main2_ai,main2_ft,main3)
- OCEANSCOPE_RUN: ensemble name (default ens3); the corrected one is <name>_oi
- OCEANSCOPE_SINGLE: one member to also correct per depth on its own, for comparison
  (default main2_ai; empty to skip); it is written as <member>_oid

Nothing is published by this script.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
MEMBERS = [
    m.strip()
    for m in os.environ.get("OCEANSCOPE_MEMBERS", "main2_ai,main2_ft,main3").split(",")
    if m.strip()
]
RUN = os.environ.get("OCEANSCOPE_RUN", "ens3")
SINGLE = os.environ.get("OCEANSCOPE_SINGLE", "main2_ai").strip()
OUT = Path("/kaggle/working/outputs_ens")
ARGO_DIR = Path("/kaggle/working/argo")
CFG = "configs/experiment.yaml"


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

import logging  # noqa: E402

import pandas as pd  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.argo import fetch_month  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
config = load_config(f"{ML}/{CFG}")

# 1) Inputs: part stores, stats, each member's prediction store, attached ARGO month files.
stores, stats, found, attached = [], None, {m: [] for m in MEMBERS}, {}
for root, dirs, files in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    name = os.path.basename(root)
    if name == "oceanscope_v1.zarr":
        stores.append(root)
        dirs[:] = []
    elif name == "stats.zarr":
        stats = root
        dirs[:] = []
    elif name.startswith("predictions_") and name.endswith(".zarr"):
        member = name[len("predictions_") : -len(".zarr")]
        if member in found:
            found[member].append(root)
        dirs[:] = []
    elif name.endswith(".zarr"):
        dirs[:] = []
    else:
        for f in files:
            if f.startswith("argo_") and f.endswith(".nc") and len(f) == len("argo_2018-01.nc"):
                attached[f[5:12]] = Path(root) / f
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats, "\nAttached ARGO months:", len(attached))
for m, paths in found.items():
    print(f"Member {m}:", *paths, sep="\n  ")
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the saved main2 training version (stats.zarr).")
missing = [m for m, p in found.items() if not p]
twice = [m for m, p in found.items() if len(p) > 1]
if missing:
    raise SystemExit(f"No predictions_<run>.zarr found for: {', '.join(missing)}")
if twice:
    raise SystemExit(f"More than one store for: {', '.join(twice)}. Attach one version each.")
PREDS = {m: p[0] for m, p in found.items()}
S = " ".join(stores)

# 2) ARGO months of the validation period (2021 is sealed and refused by fetch_month).
val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
months = pd.period_range(val_start.to_period("M"), val_end.to_period("M"), freq="M")
argo_files = []
for m in months:
    p = attached.get(str(m))
    argo_files.append(p if p is not None else fetch_month(m, config, ARGO_DIR))
print(f"ARGO months ready: {len(argo_files)}", flush=True)
FILES = " ".join(str(p) for p in argo_files)
holdout = Path(ML) / "configs" / "argo_eval_floats.json"
OUT.mkdir(parents=True, exist_ok=True)
P = OUT / "predictions"

# 3) Ensemble, then the per-depth correction of the ensemble (and of one member alone).
run(
    f"python -m oceanscope_ml.inference.ensemble --config {CFG} "
    f"--predictions {' '.join(PREDS[m] for m in MEMBERS)} --run-id {RUN} --out-dir {OUT}"
)
jobs = [(P / f"predictions_{RUN}.zarr", f"{RUN}_oi")]
if SINGLE:
    if SINGLE not in PREDS:
        raise SystemExit(f"OCEANSCOPE_SINGLE={SINGLE} is not one of the members")
    jobs.append((Path(PREDS[SINGLE]), f"{SINGLE}_oid"))
for src, rid in jobs:
    run(
        f"python -m oceanscope_ml.inference.argo_correct --config {CFG} --predictions {src} "
        f"--argo {FILES} --holdout {holdout} --run-id {rid} --out-dir {OUT} --per-depth"
    )

# 4) ARGO check on the held-out floats and validation against GLORYS.
checked = [RUN, f"{RUN}_oi"] + ([f"{SINGLE}_oid"] if SINGLE else [])
for rid in checked:
    pred = P / f"predictions_{rid}.zarr"
    run(
        f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {pred} "
        f"--store {S} --stats {stats} --argo {FILES} --holdout {holdout} --out-dir {OUT}"
    )
    run(
        f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {pred} "
        f"--store {S} --stats {stats} --out-dir {OUT}"
    )
M = OUT / "metrics"
names = [f"argo_correct_depths_{RUN}_oi.csv"]
if SINGLE:
    names.append(f"argo_correct_depths_{SINGLE}_oid.csv")
for rid in checked:
    names += [f"{rid}_headline.csv", f"argo_{rid}.csv", f"{rid}_val.csv"]
for name in names:
    if (M / name).exists():
        run(f"cat {M / name}")
print("\nENSEMBLE + CORRECTION DONE (nothing published)")
