"""Kaggle step D: score the run picked on the dev half ONCE on the final half of the held-out
floats (DECISIONS 2026-11-06). No GPU. Nothing is published, nothing reads 2021.

Inputs: the part stores, the saved main2 training version (stats.zarr), the saved version that
holds predictions_<RUN>.zarr (run E for ens3w_rm_oi) and the saved version that holds the live
model's predictions_<BASE>.zarr for reference (run A or E hold ens3). Internet on.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_final_half.py

Settings (env): OCEANSCOPE_RUN (the picked run, default ens3w_rm_oi), OCEANSCOPE_BASE (the live
model, scored on the same floats for reference only, default ens3).
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
RUN = os.environ.get("OCEANSCOPE_RUN", "ens3w_rm_oi")
BASE = os.environ.get("OCEANSCOPE_BASE", "ens3")
OUT = Path("/kaggle/working/outputs_final")
ARGO_DIR = Path("/kaggle/working/argo")
CFG = "configs/experiment.yaml"


def run(cmd, cwd=ML):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not os.path.isdir(REPO):
    run(
        "git clone -b ml/argo-improvements "
        f"https://github.com/aryanchettripkt-eng/ocean-embed.git {REPO}",
        cwd="/kaggle/working",
    )
else:
    run("git pull --ff-only origin $(git rev-parse --abbrev-ref HEAD)", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")
sys.path.insert(0, f"{ML}/src")

import pandas as pd  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.argo import fetch_month  # noqa: E402

config = load_config(f"{ML}/{CFG}")
stores, stats, found, attached = [], None, {RUN: [], BASE: []}, {}
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
        rid = name[len("predictions_") : -len(".zarr")]
        if rid in found:
            found[rid].append(root)
        dirs[:] = []
    elif name.endswith(".zarr"):
        dirs[:] = []
    else:
        for f in files:
            if f.startswith("argo_") and f.endswith(".nc") and len(f) == len("argo_2018-01.nc"):
                attached[f[5:12]] = Path(root) / f
stores.sort()
print("Stores:", *stores, sep="\n  ")
for rid, paths in found.items():
    print(f"{rid}:", *paths, sep="\n  ")
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the saved main2 training version (stats.zarr).")
if len(found[RUN]) != 1:
    raise SystemExit(f"Attach exactly one version holding predictions_{RUN}.zarr")
val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
months = pd.period_range(val_start.to_period("M"), val_end.to_period("M"), freq="M")
argo_files = [attached.get(str(m)) or fetch_month(m, config, ARGO_DIR) for m in months]
FILES = " ".join(str(p) for p in argo_files)
holdout = Path(ML) / "configs" / "argo_eval_floats.json"
S = " ".join(stores)

scored = [RUN] + ([BASE] if found[BASE] else [])
for rid in scored:
    run(
        f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} "
        f"--predictions {found[rid][0]} --store {S} --stats {stats} --argo {FILES} "
        f"--holdout {holdout} --out-dir {OUT} --half final"
    )
for rid in scored:
    run(f"cat {OUT}/metrics/{rid}_final_headline.csv")
    run(f"cat {OUT}/metrics/argo_{rid}_final.csv")
print(f"\nFINAL HALF SCORED ONCE for {RUN} (nothing published)")
