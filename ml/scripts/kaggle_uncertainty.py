"""Kaggle: honest error bars for main3_aix30a (Bayesian last layer fitted with Batch-and-Match).

One run, Save Version -> "Save & Run All (Commit)". GPU T4 x2 or P100, Internet on (ARGO months
from the Argo ERDDAP). Run as:
    %env OCEANSCOPE_OWNER_GO=1
    %run /kaggle/working/ocean-embed/ml/scripts/kaggle_uncertainty.py

Inputs (Add Input -> Your Work -> Notebooks): the part stores (including the one with 2021), the
stats version (stats.zarr) and the run M notebook (checkpoints/main3_aix30a_l3/best_ft.pt).

What it does (evaluation.uncertainty):
1. Predicts the validation years (2018-2020) with the frozen model and keeps, at every held-out
   float profile, the 32 features that feed the U-Net's 1x1 head.
2. With OCEANSCOPE_OWNER_GO=1 it does the same for 2021. This is a second look at the sealed
   test year, asked for by Aryan (DECISIONS 2026-11-13). The mean prediction is the same model
   that was scored once; nothing is fitted on 2021, and the headline result does not change.
3. Fits the per-depth Bayesian last layer on 2018-2020 only and reports 68% / 95% coverage per
   depth band with float-bootstrap intervals, for the constant error bar, the last layer at its
   posterior mean, and the full BaM posterior. Also a dev-half -> final-half check inside
   2018-2020. Reliability plots are saved as PNGs.

Settings (env): OCEANSCOPE_SINGLE (default main3_aix30a), OCEANSCOPE_OWNER_GO=1 for 2021.
Nothing is published.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
NAME = os.environ.get("OCEANSCOPE_SINGLE", "main3_aix30a").strip()
GO = os.environ.get("OCEANSCOPE_OWNER_GO", "") == "1"
OUT = Path("/kaggle/working/outputs_uncert")
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
else:  # pull the branch the notebook cloned, never merge another one into it
    run("git pull --ff-only origin $(git rev-parse --abbrev-ref HEAD)", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")
os.environ["PYTHONPATH"] = f"{ML}/src"
sys.path.insert(0, f"{ML}/src")

import pandas as pd  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.argo import fetch_month  # noqa: E402

config = load_config(f"{ML}/{CFG}")

# 1) Inputs.
stores, stats, ckpts = [], None, []
for root, dirs, files in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    name = os.path.basename(root)
    if name == "oceanscope_v1.zarr":
        stores.append(root)
        dirs[:] = []
        continue
    if name == "stats.zarr":
        stats = stats or root
        dirs[:] = []
        continue
    if name.endswith(".zarr"):
        dirs[:] = []
        continue
    if (name == NAME or name.startswith(f"{NAME}_l")) and "best_ft.pt" in files:
        ckpts.append(Path(root) / "best_ft.pt")
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats)
print("Checkpoints:", ckpts)
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the stats version (stats.zarr).")
if len(ckpts) != 1:
    raise SystemExit(f"Need exactly one {NAME} best_ft.pt (attach the run M notebook once)")
ckpt = ckpts[0]

# 2) ARGO months: the validation years plus the 30-day input window before them, and with the
#    owner's go the test year with its window.
months = list(pd.period_range("2017-12", "2020-12", freq="M"))
val_files = [fetch_month(m, config, ARGO_DIR) for m in months]
test_files = []
if GO:
    test_files = [fetch_month(m, config, ARGO_DIR, allow_test=True)
                  for m in pd.period_range("2020-12", "2021-12", freq="M")]  # fmt: skip
print(f"ARGO months ready: {len(val_files)} validation, {len(test_files)} for 2021", flush=True)
holdout = Path(ML) / "configs" / "argo_eval_floats.json"
common = f"--config {CFG} --checkpoint {ckpt} --store {' '.join(stores)} --stats {stats} " \
         f"--holdout {holdout} --out-dir {OUT}"  # fmt: skip

# 3) Matches + head features.
run(f"python -m oceanscope_ml.evaluation.uncertainty collect {common} --split val "
    f"--argo {' '.join(map(str, val_files))}")  # fmt: skip
if GO:
    run(f"python -m oceanscope_ml.evaluation.uncertainty collect {common} --split test "
        f"--argo {' '.join(map(str, test_files))} --owner-go")  # fmt: skip

# 4) Fit on 2018-2020, check coverage.
check = f"--check {OUT / 'uncert_test.parquet'}" if GO else ""
run(f"python -m oceanscope_ml.evaluation.uncertainty analyse --fit {OUT / 'uncert_val.parquet'} "
    f"{check} --out-dir {OUT}")  # fmt: skip
run(f"rm -rf {OUT / 'uncertainty'}")  # the two prediction stores; the tables are what we need
run(f"ls -la {OUT} {OUT / 'metrics'}")
print("\nUNCERTAINTY RUN DONE (nothing published). Send the log; the parquet and CSV files are "
      "in outputs_uncert.")  # fmt: skip
