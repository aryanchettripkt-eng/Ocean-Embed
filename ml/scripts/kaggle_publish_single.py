"""Kaggle: publish one fine-tuned model as the site's active run, then upload its live kit.

For main3_aix30a (DECISIONS 2026-11-10, Aryan: "Switch to main3_aix30a"). The model's validation
prediction store already exists in its fine-tune notebook; this scores it against ALL held-out
floats of the validation years (the fine-tune run scored only the dev half), writes the
validation metrics and manifest, publishes (with OCEANSCOPE_PUBLISH=1), and builds and uploads
the live kit for the same run name (with OCEANSCOPE_UPLOAD=1). The daily live run refuses a kit
that is not the published run, so both go up in the same notebook, publish first.

Run with Save Version -> "Save & Run All (Commit)". No GPU. Internet on (ARGO 2018-2020 from the
Argo ERDDAP). Secret HF_TOKEN (write) only to publish or upload.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_publish_single.py

Inputs: the part stores, the stats notebook (stats.zarr, and the baseline metrics if it has
them), and the model's fine-tune notebook (predictions_<RUN>.zarr and
checkpoints/<RUN>_l<lambda>/best_ft.pt).

Settings (env): OCEANSCOPE_RUN (default main3_aix30a), OCEANSCOPE_PUBLISH=1,
OCEANSCOPE_UPLOAD=1. Without them it is a dry run. The 2021 test year is not read.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
RUN = os.environ.get("OCEANSCOPE_RUN", "main3_aix30a").strip()
PUBLISH = os.environ.get("OCEANSCOPE_PUBLISH", "0") == "1"
UPLOAD = os.environ.get("OCEANSCOPE_UPLOAD", "0") == "1"
OUT = Path(f"/kaggle/working/outputs_{RUN}")
KIT = Path("/kaggle/working/kit")
ARGO_DIR = Path("/kaggle/working/argo")
CFG = "configs/experiment.yaml"


def run(cmd, cwd=ML):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if os.path.isdir(REPO):  # pull the branch the notebook cloned, never merge another one into it
    run("git pull --ff-only origin $(git rev-parse --abbrev-ref HEAD)", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")
sys.path.insert(0, f"{ML}/src")
if PUBLISH or UPLOAD:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

from oceanscope_ml.config import load_config  # noqa: E402
from oceanscope_ml.data.argo import fetch_month  # noqa: E402

config = load_config(f"{ML}/{CFG}")

# 1) Inputs.
stores, stats, preds, baselines, ckpts, histories = [], None, [], {}, {}, {}
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
        if "best_ft.pt" in files and (name == RUN or name.startswith(f"{RUN}_l")):
            ckpts[name] = Path(root) / "best_ft.pt"
        for f in files:
            if f.startswith("baseline_") and f.endswith("_val.csv"):
                baselines.setdefault(f, Path(root) / f)
            if f.startswith(f"finetune_{RUN}_l") and f.endswith(".csv"):
                histories[f[len("finetune_") : -len(".csv")]] = Path(root) / f
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats, "\nPredictions:", preds, "\nCheckpoints:", ckpts)
print("Baselines:", list(baselines))
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the stats notebook (stats.zarr).")
if len(preds) != 1:
    raise SystemExit(f"Need exactly one predictions_{RUN}.zarr (found {len(preds)}).")
if not ckpts:
    raise SystemExit(f"No checkpoints/{RUN}_l*/best_ft.pt: attach the fine-tune notebook.")
if len(ckpts) > 1:  # the lambda with the lowest validation ARGO RMSE, as kaggle_finetune.py
    scores = {r: pd.read_csv(histories[r])["val_argo_rmse_c"].min() for r in ckpts}
    print("Validation ARGO RMSE per run:", scores)
    ckpt = ckpts[min(scores, key=scores.get)]
else:
    ckpt = next(iter(ckpts.values()))
print("Checkpoint:", ckpt)
S = " ".join(stores)

# 2) ARGO profiles of the validation years (2021 stays sealed).
val_files = [fetch_month(m, config, ARGO_DIR) for m in pd.period_range("2018-01", "2020-12")]
holdout = Path(ML) / "configs" / "argo_eval_floats.json"

# 3) Fresh outputs folder: the prediction store, metrics on all held-out floats, manifest.
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "predictions").mkdir(parents=True)
(OUT / "metrics").mkdir()
PRED = OUT / "predictions" / f"predictions_{RUN}.zarr"
shutil.copytree(preds[0], PRED)
for name, src in baselines.items():
    shutil.copy(src, OUT / "metrics" / name)
shutil.copy(holdout, OUT / "metrics" / "argo_eval_floats.json")
run(
    f"python -m oceanscope_ml.evaluation.argo_validation --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --argo {' '.join(str(p) for p in val_files)} "
    f"--holdout {holdout} --out-dir {OUT} --half all"
)
run(
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} --predictions {PRED} "
    f"--store {S} --stats {stats} --out-dir {OUT}"
)
for name in (f"{RUN}_val.csv", f"argo_{RUN}.csv", f"{RUN}_headline.csv"):
    run(f"cat {OUT}/metrics/{name}")
run(f"cat {OUT}/manifest.json")

# 4) Publish (dry run always).
run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT} --dry-run")
if PUBLISH:
    run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT}")
    print(f"\nPublished run {RUN}")

# 5) Live kit for the same run name (uploaded only with OCEANSCOPE_UPLOAD=1).
kit_cmd = (
    f"python -m oceanscope_ml.live kit --checkpoint {ckpt} --run-id {RUN} --stats {stats} "
    f"--static-store {stores[0]} --out {KIT}"
)
run(kit_cmd)
run(f"cat {KIT}/kit.json")
if UPLOAD:
    if not PUBLISH:
        print("Note: uploading a kit for a run that this notebook did not publish.")
    run(kit_cmd + " --upload")
print(
    "\nPUBLISH SINGLE DONE"
    + (" (published)" if PUBLISH else " (publish dry run)")
    + (" (kit uploaded)" if UPLOAD else " (kit not uploaded)")
)
