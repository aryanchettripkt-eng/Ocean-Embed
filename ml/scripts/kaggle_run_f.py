"""Kaggle run F: richer tree-model correction (v2 features), no correction below 600 m.

Run with Save Version -> "Save & Run All (Commit)". No GPU. Inputs: the part stores, the saved
main2 training version (stats.zarr) and the saved run E version, whose weighted ensemble
predictions_<RUN>w.zarr is reused. Without run E, attach each member's prediction version
instead and the weighted ensemble is rebuilt the same way. Internet on
(ARGO files, and the IOD / ENSO indices for the regime table). Never attach the 2021
final-test version: test dates are refused.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_run_f.py

Steps (DECISIONS 2026-11-06, run F; nothing is published, nothing reads 2021):
1. weighted ensemble <RUN>w (from run E if attached, else rebuilt the same way)
2. <RUN>w_rm2: tree correction with the v2 features, no training or correction below 600 m
3. <RUN>w_rm2_oi: the guarded per-depth ARGO correction on top, also stopped below 600 m
4. ARGO check of both on the dev half of the held-out floats, and validation vs GLORYS
5. errors by regime for both
6. one summary table (run E's <RUN>w_rm_oi scored 0.816 C on the dev half: the number to beat)

Settings (env): OCEANSCOPE_MEMBERS (default main2_ai,main2_ft,main3), OCEANSCOPE_RUN (default
ens3), OCEANSCOPE_HALF (default dev), OCEANSCOPE_MIN_DEPTH (default 1.0),
OCEANSCOPE_OUTLIER_SIGMA (default 4), OCEANSCOPE_MAX_DEPTH (default 600),
OCEANSCOPE_TREE_WINDOW (days the tree's nearby-float features look back, default 10).
"""

import json
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
HALF = os.environ.get("OCEANSCOPE_HALF", "dev").strip()
MIN_DEPTH = float(os.environ.get("OCEANSCOPE_MIN_DEPTH", "1.0") or 0)
OUTLIER = os.environ.get("OCEANSCOPE_OUTLIER_SIGMA", "4").strip()
MAX_DEPTH = float(os.environ.get("OCEANSCOPE_MAX_DEPTH", "600") or 0)
TREE_WINDOW = int(os.environ.get("OCEANSCOPE_TREE_WINDOW", "10") or 10)
DEEP = f" --max-depth {MAX_DEPTH:g}" if MAX_DEPTH > 0 else ""
if HALF not in ("dev", "all", "final"):
    raise SystemExit("OCEANSCOPE_HALF must be dev, all or final")
GUARDS = (f" --min-depth {MIN_DEPTH:g}" if MIN_DEPTH > 0 else "") + (
    f" --outlier-sigma {OUTLIER}" if OUTLIER else ""
)
H = "" if HALF == "all" else f"_{HALF}"
OUT = Path("/kaggle/working/outputs_f")
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

import logging  # noqa: E402

import pandas as pd  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.argo import fetch_month  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
config = load_config(f"{ML}/{CFG}")
print(f"Run F on {RUN} = {', '.join(MEMBERS)}; guards:{GUARDS or ' none'};"
      f"{DEEP or ' no depth cut'}; {HALF} half")  # fmt: skip

# 1) Inputs: part stores, stats, each member's prediction store, attached ARGO month files.
stores, stats, found, attached = [], None, {m: [] for m in [*MEMBERS, f"{RUN}w"]}, {}
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
REUSE = bool(found[f"{RUN}w"])
missing = [] if REUSE else [m for m in MEMBERS if not found[m]]
twice = [m for m, p in found.items() if len(p) > 1 and (not REUSE or m == f"{RUN}w")]
if missing:
    raise SystemExit(
        f"No predictions_<run>.zarr found for: {', '.join(missing)}. Attach the run E version, "
        "or each member's predictions."
    )
if twice:
    raise SystemExit(f"More than one store for: {', '.join(twice)}. Attach one version each.")
PREDS = [] if REUSE else [found[m][0] for m in MEMBERS]
S = " ".join(stores)

val_start, val_end = (pd.Timestamp(d) for d in config.splits.val)
months = pd.period_range(val_start.to_period("M"), val_end.to_period("M"), freq="M")
argo_files = [attached.get(str(m)) or fetch_month(m, config, ARGO_DIR) for m in months]
print(f"ARGO months ready: {len(argo_files)}", flush=True)
FILES = " ".join(str(p) for p in argo_files)
holdout = Path(ML) / "configs" / "argo_eval_floats.json"
OUT.mkdir(parents=True, exist_ok=True)
P = OUT / "predictions"
M = OUT / "metrics"

# 2) Weighted ensemble: reuse run E's, or rebuild it the same way.
WSTORE = P / f"predictions_{RUN}w.zarr"
if REUSE:
    WSTORE = Path(found[f"{RUN}w"][0])
    print("Reusing run E's weighted ensemble:", WSTORE)
else:
    run(
        f"python -m oceanscope_ml.inference.ensemble_weights --config {CFG} "
        f"--predictions {' '.join(PREDS)} --argo {FILES} --holdout {holdout} "
        f"--run-id {RUN}w --out-dir {OUT}"
    )
    weights = json.loads((M / f"ensemble_weights_{RUN}w.json").read_text())["weights"]
    print("Weights:", dict(zip(MEMBERS, weights)))
    run(
        f"python -m oceanscope_ml.inference.ensemble --config {CFG} --predictions "
        f"{' '.join(PREDS)} --run-id {RUN}w --out-dir {OUT} "
        f"--weights {','.join(f'{w:g}' for w in weights)}"
    )

# 3) v2 tree correction, then the guarded per-depth correction on top; both stop at MAX_DEPTH.
run(
    f"python -m oceanscope_ml.inference.residual_model --config {CFG} "
    f"--predictions {WSTORE} --store {S} --stats {stats} --argo {FILES} --holdout {holdout} "
    f"--run-id {RUN}w_rm2 --out-dir {OUT} --features v2{DEEP} --window-days {TREE_WINDOW}"
)
run(
    f"python -m oceanscope_ml.inference.argo_correct --config {CFG} "
    f"--predictions {P / f'predictions_{RUN}w_rm2.zarr'} --argo {FILES} --holdout {holdout} "
    f"--run-id {RUN}w_rm2_oi --out-dir {OUT} --per-depth{GUARDS}{DEEP}"
)

# 4) ARGO check on the chosen half, and validation against GLORYS.
checked = [f"{RUN}w_rm2", f"{RUN}w_rm2_oi"]
for rid in checked:
    pred = P / f"predictions_{rid}.zarr"
    run(
        f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {pred} "
        f"--store {S} --stats {stats} --argo {FILES} --holdout {holdout} --out-dir {OUT} "
        f"--half {HALF}"
    )
    run(
        f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {pred} "
        f"--store {S} --stats {stats} --out-dir {OUT}"
    )

# 5) Errors by regime.
run(
    "python -m oceanscope_ml.evaluation.regimes --matches "
    + " ".join(str(M / f"argo_matches_{rid}{H}.parquet") for rid in checked)
    + f" --events {REPO}/backend/app/data/events.json --run-id {RUN}_f{H} --out-dir {OUT}"
)

for name in (
    f"residual_model_cv_{RUN}w_rm2.csv",
    f"argo_correct_depths_{RUN}w_rm2_oi.csv",
    *[f"argo_{rid}{H}.csv" for rid in checked],
    *[f"{rid}_val.csv" for rid in checked],
):
    if (M / name).exists():
        run(f"cat {M / name}")
reg = pd.read_csv(M / f"regimes_{RUN}_f{H}.csv")
print("\nRegimes, Bay of Bengal 50-300 m (RMSE C):")
band = reg[reg["group"] == "bob_50_300"]
print(band.pivot_table(index=["regime", "value"], columns="product", values="rmse_c").round(3)
      .to_string())  # fmt: skip

# 6) Summary: Bay of Bengal 50-300 m RMSE on the chosen half, every run vs the references.
print(f"\nSUMMARY: Bay of Bengal 50-300 m, held-out floats ({HALF} half), validation years")
print("(run E, dev half: ens3w_rm_oi 0.816 C, GLORYS12 0.756, GREP 0.855, ARMOR3D 0.639)")
summary = []
for rid in checked:
    h = pd.read_csv(M / f"{rid}{H}_headline.csv")
    h = h[h["period"] == "val"].set_index("product")
    m = h.loc["model"]
    row = {"run": rid, "n_floats": m["n_floats"], "model": m["rmse_c"], "ci_low": m["ci_low_c"],
           "ci_high": m["ci_high_c"]}  # fmt: skip
    row.update({p: h.loc[p, "rmse_c"] for p in ("glorys12", "grep", "armor3d") if p in h.index})
    summary.append(row)
print(pd.DataFrame(summary).round(3).to_string(index=False))
print("\nRUN F DONE (nothing published)")
