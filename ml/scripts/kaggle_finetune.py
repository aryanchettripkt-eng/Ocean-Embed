"""Kaggle: ARGO fine-tuning of a trained run, in one go (PRD_AI_ML §5.9, §7, §1.1).

Run with Save Version -> "Save & Run All (Commit)". Inputs: the part stores and the saved version
of the base training run (it holds outputs/, stats.zarr and checkpoints/<run>/best.pt). GPU on,
Internet on. Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_finetune.py

Settings (env): OCEANSCOPE_RUN (base run, default main2), OCEANSCOPE_FT_LAMBDAS (default
"0.03,0.1,0.3"), OCEANSCOPE_FT_EPOCHS (default 6), OCEANSCOPE_PUBLISH=1 to upload at the end.

Steps: ARGO 2004-2020 from the Argo ERDDAP (never 2021) -> held-out floats (the committed list,
checked against a rebuild) -> one fine-tune per lambda -> the lambda with the lowest validation
ARGO RMSE on non-held-out floats -> predict validation -> evaluate -> ARGO check on the held-out
floats -> publish only with OCEANSCOPE_PUBLISH=1.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
BASE = os.environ.get("OCEANSCOPE_RUN", "main2")
RUN = f"{BASE}_ft"
LAMBDAS = [float(x) for x in os.environ.get("OCEANSCOPE_FT_LAMBDAS", "0.03,0.1,0.3").split(",")]
EPOCHS = int(os.environ.get("OCEANSCOPE_FT_EPOCHS", "6"))
PUBLISH = os.environ.get("OCEANSCOPE_PUBLISH", "0") == "1"
OUT = Path("/kaggle/working/outputs_ft")
ARGO_DIR = Path("/kaggle/working/argo")
CACHE = "/tmp/cache_ft"
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
if PUBLISH:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

import logging  # noqa: E402

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.argo import fetch_month  # noqa: E402
from oceanscope_ml.data.inspect import build_real_argo_holdout  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
config = load_config(f"{ML}/{CFG}")

# 1) Inputs: part stores, the base run's outputs, stats and best.pt.
stores, base_outputs, stats, base_ckpt = [], None, None, None
for root, dirs, files in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    if root.endswith("oceanscope_v1.zarr"):
        stores.append(root)
        dirs[:] = []
    elif root.endswith(f"predictions_{BASE}.zarr"):
        base_outputs = Path(root).parent.parent
        dirs[:] = []
    elif root.endswith("stats.zarr"):
        stats = root
        dirs[:] = []
    elif root.endswith(".zarr"):
        dirs[:] = []
    elif root.endswith(f"checkpoints/{BASE}") and "best.pt" in files:
        base_ckpt = Path(root) / "best.pt"
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Base outputs:", base_outputs, "\nStats:", stats, "\nBase checkpoint:", base_ckpt)
if not stores or base_outputs is None or stats is None or base_ckpt is None:
    raise SystemExit(f"Attach the part stores and the saved version of the {BASE} run.")
S = " ".join(stores)

# 2) ARGO profiles, 2004-2020 (input-era floats start in 2004; 2021 is sealed).
months = pd.period_range("2004-01", "2020-12", freq="M")
argo_files = []
for m in months:
    argo_files.append(fetch_month(m, config, ARGO_DIR))
    print(f"  ARGO {m} ready", flush=True)
FILES = " ".join(str(p) for p in argo_files)
val_files = [p for p, m in zip(argo_files, months, strict=True) if m.year >= 2018]

# 3) Held-out floats: the committed list, checked against a rebuild from the same download.
cols = ["platform_number", "cycle_number", "time", "latitude", "longitude"]
prof = []
for f in val_files:
    with xr.open_dataset(f) as ds:
        df = ds[cols].to_dataframe()
        prof.append(df.drop_duplicates(["platform_number", "cycle_number"]))
summary = ARGO_DIR / "profiles_summary_val.nc"
merged = pd.concat(prof, ignore_index=True)
xr.Dataset({c: ("row", merged[c].to_numpy()) for c in cols}).to_netcdf(summary)
rebuilt = build_real_argo_holdout([summary], seed=int(config.train.get("seed", 42)))
committed = Path(ML) / "configs" / "argo_eval_floats.json"
if committed.exists():
    holdout_path = committed
    listed = set(json.loads(committed.read_text(encoding="utf-8"))["held_out_floats"])
    same = listed == set(rebuilt["held_out_floats"])
    print(f"Held-out floats: {len(listed)} (committed list); rebuild matches: {same}")
else:
    holdout_path = ARGO_DIR / "argo_eval_floats.json"
    holdout_path.write_text(json.dumps(rebuilt, indent=2), encoding="utf-8")
    print(f"Held-out floats: {rebuilt['total_held_out']} (rebuilt, no committed list)")

# 4) One fine-tune per lambda.
results = {}
for lam in LAMBDAS:
    rid = f"{RUN}_l{lam:g}"
    run(
        f"python -m oceanscope_ml.finetune --config {CFG} --store {S} --stats {stats} "
        f"--cache-dir {CACHE} --base {base_ckpt} --argo {FILES} --holdout {holdout_path} "
        f"--lambda {lam:g} --run-id {rid} --epochs {EPOCHS} --patience 2 --device cuda"
    )
    hist = pd.read_csv(Path(ML) / config.paths.outputs / "metrics" / f"finetune_{rid}.csv")
    print(hist.to_string(index=False))
    results[lam] = (hist["val_argo_rmse_c"].min(), rid, hist["val_argo_rmse_c"].iloc[0])

best_lam = min(results, key=lambda k: results[k][0])
best_rmse, best_rid, base_rmse = results[best_lam]
print(f"\nBase {BASE}: val ARGO RMSE (non-held-out floats) {base_rmse:.3f} C")
for lam, (rmse, rid, _) in sorted(results.items()):
    print(f"  lambda {lam:g}: best {rmse:.3f} C ({rid})")
print(f"Picked lambda {best_lam:g}: {best_rmse:.3f} C", flush=True)

# 5) Predict, evaluate and ARGO-check the picked run under the name RUN.
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "metrics").mkdir(exist_ok=True)
for name in ("baseline_climatology_val.csv", "baseline_ridge_val.csv"):
    src = base_outputs / "metrics" / name
    if src.exists():
        shutil.copy(src, OUT / "metrics" / name)
ckpt = Path(ML) / config.paths.checkpoints / best_rid / "best_ft.pt"
PRED = OUT / "predictions" / f"predictions_{RUN}.zarr"
run(
    f"python -m oceanscope_ml.inference.predict --config {CFG} --store {S} --stats {stats} "
    f"--cache-dir {CACHE} --checkpoint {ckpt} --run-id {RUN} --out-dir {OUT}"
)
run(
    f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --argo {' '.join(str(p) for p in val_files)} "
    f"--holdout {holdout_path} --out-dir {OUT}"
)
run(
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --out-dir {OUT}"
)
for name in (f"{RUN}_val.csv", f"argo_{RUN}.csv", f"{RUN}_headline.csv"):
    run(f"cat {OUT}/metrics/{name}")
shutil.copy(holdout_path, OUT / "metrics" / "argo_eval_floats.json")

# 6) Publish (dry run always; upload only with OCEANSCOPE_PUBLISH=1).
run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT} --dry-run")
if PUBLISH:
    run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT}")
print("\nFINE-TUNE DONE")
