"""Kaggle: publish a saved ensemble run (from kaggle_ensemble_correct.py) as the active run.

Run with Save Version -> "Save & Run All (Commit)". Inputs: the part stores, the saved main2
training version (stats.zarr and the baseline metrics) and the saved ensemble version (it holds
outputs_ens/predictions/predictions_<run>.zarr and its ARGO check files). No GPU. Internet on.
Secret HF_TOKEN (write token) only when publishing.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_publish_ensemble.py

Settings (env): OCEANSCOPE_RUN (default ens3), OCEANSCOPE_PUBLISH=1 to upload at the end
(otherwise only the dry run). Validation years only: the 2021 test year is not read, and the
manifest says test_evaluated false.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
RUN = os.environ.get("OCEANSCOPE_RUN", "ens3")
PUBLISH = os.environ.get("OCEANSCOPE_PUBLISH", "0") == "1"
OUT = Path(f"/kaggle/working/outputs_{RUN}")
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
if PUBLISH:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# 1) Inputs.
stores, stats, preds, metrics = [], None, [], {}
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
    elif name == f"predictions_{RUN}.zarr":
        preds.append(root)
        dirs[:] = []
    elif name.endswith(".zarr"):
        dirs[:] = []
    else:
        for f in files:
            wanted = f in (
                f"argo_{RUN}.csv",
                f"argo_matches_{RUN}.parquet",
                f"{RUN}_headline.csv",
            ) or (f.startswith("baseline_") and f.endswith("_val.csv"))
            if wanted:
                metrics.setdefault(f, Path(root) / f)
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats, "\nPredictions:", *preds, "\nMetrics:", *metrics.values(), sep="\n  ")
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the saved main2 training version (stats.zarr).")
if len(preds) != 1:
    raise SystemExit(f"Need exactly one predictions_{RUN}.zarr (found {len(preds)}).")
for need in (f"argo_{RUN}.csv", f"{RUN}_headline.csv"):
    if need not in metrics:
        raise SystemExit(f"{need} not found: attach the saved ensemble version.")
S = " ".join(stores)

# 2) Fresh outputs folder with only this run's files.
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "predictions").mkdir(parents=True)
(OUT / "metrics").mkdir()
PRED = OUT / "predictions" / f"predictions_{RUN}.zarr"
shutil.copytree(preds[0], PRED)
for name, src in metrics.items():
    shutil.copy(src, OUT / "metrics" / name)
shutil.copy(
    Path(ML) / "configs" / "argo_eval_floats.json", OUT / "metrics" / "argo_eval_floats.json"
)

# 3) Validation metrics and manifest, then publish.
run(
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --out-dir {OUT}"
)
run(f"cat {OUT}/manifest.json")
for name in (f"{RUN}_val.csv", f"{RUN}_headline.csv"):
    run(f"cat {OUT}/metrics/{name}")
run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT} --dry-run")
if PUBLISH:
    run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT}")
    print(f"\nPublished run {RUN}")
print("\nPUBLISH ENSEMBLE DONE" + ("" if PUBLISH else " (dry run only)"))
