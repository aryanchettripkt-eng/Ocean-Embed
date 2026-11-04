"""Kaggle: ARGO residual correction of a published run, and its ARGO check (no GPU needed).

Run with Save Version -> "Save & Run All (Commit)". Inputs: the part stores and the saved version
of the fine-tune run (it holds outputs_ft/predictions/predictions_<run>.zarr and, usually, the
ARGO month files in argo/). Internet on (only used if the ARGO files are not attached).
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_argo_correct.py

Settings (env): OCEANSCOPE_RUN (base run, default main2_ft). The new run is <run>_oi. Nothing is
published by this script.

Steps: find inputs -> ARGO 2018-2020 (attached or downloaded; never 2021) -> tune and write the
corrected store (non-held-out floats only) -> ARGO check of the base and the corrected run on the
held-out floats -> evaluate the corrected run against GLORYS -> print the tables.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
BASE = os.environ.get("OCEANSCOPE_RUN", "main2_ft")
RUN = f"{BASE}_oi"
OUT = Path("/kaggle/working/outputs_oi")
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

# 1) Inputs: part stores, stats, the base run's prediction store, attached ARGO month files.
stores, stats, base_pred, attached = [], None, None, {}
for root, dirs, files in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    if root.endswith("oceanscope_v1.zarr"):
        stores.append(root)
        dirs[:] = []
    elif root.endswith(f"predictions_{BASE}.zarr"):
        base_pred = root
        dirs[:] = []
    elif root.endswith("stats.zarr"):
        stats = root
        dirs[:] = []
    elif root.endswith(".zarr"):
        dirs[:] = []
    else:
        for f in files:
            if f.startswith("argo_") and f.endswith(".nc") and len(f) == len("argo_2018-01.nc"):
                attached[f[5:12]] = Path(root) / f
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats, "\nBase predictions:", base_pred, "\nAttached ARGO months:", len(attached))
if not stores or stats is None or base_pred is None:
    raise SystemExit(f"Attach the part stores and the saved version that holds {BASE}'s outputs.")
S = " ".join(stores)

# 2) ARGO months of the validation period (2021 is sealed and refused by fetch_month).
val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
months = pd.period_range(val_start.to_period("M"), val_end.to_period("M"), freq="M")
argo_files = []
for m in months:
    p = attached.get(str(m))
    argo_files.append(p if p is not None else fetch_month(m, config, ARGO_DIR))
print(f"ARGO months ready: {len(argo_files)} ({len(months) - len(attached)} downloaded)")
FILES = " ".join(str(p) for p in argo_files)
holdout = Path(ML) / "configs" / "argo_eval_floats.json"

# 3) Tune and write the corrected store.
OUT.mkdir(parents=True, exist_ok=True)
run(
    f"python -m oceanscope_ml.inference.argo_correct --config {CFG} --predictions {base_pred} "
    f"--argo {FILES} --holdout {holdout} --run-id {RUN} --out-dir {OUT}"
)
PRED = OUT / "predictions" / f"predictions_{RUN}.zarr"

# 4) ARGO check on the held-out floats, base and corrected, on the same files.
for pred in (base_pred, PRED):
    run(
        f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {pred} "
        f"--store {S} --stats {stats} --argo {FILES} --holdout {holdout} --out-dir {OUT}"
    )
# 5) Validation against GLORYS for the corrected run.
run(
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --out-dir {OUT}"
)
M = OUT / "metrics"
for name in (
    f"argo_correct_tuning_{RUN}.csv",
    f"{BASE}_headline.csv",
    f"{RUN}_headline.csv",
    f"argo_{BASE}.csv",
    f"argo_{RUN}.csv",
    f"{RUN}_val.csv",
):
    if (M / name).exists():
        run(f"cat {M / name}")
print("\nARGO CORRECTION DONE (nothing published)")
