"""Kaggle: train a new run from scratch with ARGO input channels, recency weighting and/or a year
channel, then predict, ARGO-check and evaluate it (PRD_AI_ML §5.1, §5.8.2, §7, P4-6). No publish.

Run with Save Version -> "Save & Run All (Commit)". Inputs: the part stores and the saved main2
training version (for stats.zarr and the baseline metrics). GPU on, Internet on.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_train_argo.py

Settings (env):
- OCEANSCOPE_RUN: run name (default main3)
- OCEANSCOPE_ARGO_INPUT: 1 (default) for ARGO input channels, 0 for none
- OCEANSCOPE_ARGO_LAMBDA: weight of the loss on hidden ARGO profiles (default 1.0)
- OCEANSCOPE_RECENCY: weight of the last training year relative to the first (default 2.0)
- OCEANSCOPE_YEAR_INPUT: 1 to add the year channel (default 0)
- OCEANSCOPE_EPOCHS: epoch ceiling (default 25; early stopping as in main2)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
RUN = os.environ.get("OCEANSCOPE_RUN", "main3")
ARGO_INPUT = os.environ.get("OCEANSCOPE_ARGO_INPUT", "1") == "1"
LAMBDA = float(os.environ.get("OCEANSCOPE_ARGO_LAMBDA", "1.0"))
RECENCY = float(os.environ.get("OCEANSCOPE_RECENCY", "2.0"))
YEAR = os.environ.get("OCEANSCOPE_YEAR_INPUT", "0") == "1"
EPOCHS = int(os.environ.get("OCEANSCOPE_EPOCHS", "25"))
OUT = Path(f"/kaggle/working/outputs_{RUN}")
ARGO_DIR = Path("/kaggle/working/argo")
CACHE = f"/tmp/cache_{RUN}"
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

# 1) Inputs: part stores, the stats store and main2's baseline metrics.
stores, stats, baselines = [], None, []
for root, dirs, files in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    if root.endswith("oceanscope_v1.zarr"):
        stores.append(root)
        dirs[:] = []
    elif root.endswith("stats.zarr"):
        stats = root
        dirs[:] = []
    elif root.endswith(".zarr"):
        dirs[:] = []
    else:
        baselines += [
            Path(root) / f for f in files if f.startswith("baseline_") and f.endswith("_val.csv")
        ]
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats, "\nBaselines:", *baselines, sep="\n  ")
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the saved main2 training version (stats.zarr).")
S = " ".join(stores)
print(f"Run {RUN}: ARGO input {ARGO_INPUT} (lambda {LAMBDA:g}), recency {RECENCY:g}, year {YEAR}")

# 2) ARGO 2004-2020 (2021 is sealed and refused by fetch_month).
holdout = Path(ML) / "configs" / "argo_eval_floats.json"
months = pd.period_range("2004-01", "2020-12", freq="M")
argo_files = [fetch_month(m, config, ARGO_DIR) for m in months]
print(f"ARGO months ready: {len(argo_files)}", flush=True)
FILES = " ".join(str(p) for p in argo_files)
VAL_FILES = " ".join(str(p) for p, m in zip(argo_files, months, strict=True) if m.year >= 2018)
ARGO_ARGS = f"--argo {FILES} --holdout {holdout}" if ARGO_INPUT else ""

# 3) Train from scratch (resumes from last.pt if this version is rerun).
run(
    f"python -m oceanscope_ml.train --config {CFG} --store {S} --stats {stats} "
    f"--cache-dir {CACHE} --device cuda --epochs {EPOCHS} --run-id {RUN} {ARGO_ARGS} "
    f"--argo-lambda {LAMBDA:g} --recency {RECENCY:g}" + (" --year-input" if YEAR else "")
)

# 4) Predict the validation years, ARGO check on the held-out floats, evaluate.
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(exist_ok=True)
for b in baselines:
    shutil.copy(b, OUT / "metrics" / b.name)
PRED = OUT / "predictions" / f"predictions_{RUN}.zarr"
run(
    f"python -m oceanscope_ml.inference.predict --config {CFG} --store {S} --stats {stats} "
    f"--cache-dir {CACHE} --checkpoint checkpoints/{RUN}/best.pt --run-id {RUN} "
    f"--out-dir {OUT} {ARGO_ARGS}"
)
run(
    f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --argo {VAL_FILES} --holdout {holdout} --out-dir {OUT}"
)
run(
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --out-dir {OUT}"
)
for name in (f"{RUN}_val.csv", f"argo_{RUN}.csv", f"{RUN}_headline.csv"):
    run(f"cat {OUT}/metrics/{name}")
print(f"\nTRAIN {RUN} DONE (nothing published)")
