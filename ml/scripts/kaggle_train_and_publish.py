"""Kaggle: stats, train, predict, evaluate, publish and baselines in one run (PRD_AI_ML §4)."""

# Run with Save Version -> "Save & Run All (Commit)". Settings: env OCEANSCOPE_RUN (default main2),
# OCEANSCOPE_PUBLISH=1 to upload. Run as: %run ml/scripts/kaggle_train_and_publish.py
# Everything it writes under /kaggle/working is kept with the version.
# Needs: the part stores attached as inputs, GPU on, Internet on, secret HF_TOKEN attached.
import os
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
OUT = "/kaggle/working/outputs"
STATS = "/kaggle/working/stats.zarr"
CACHE = f"/tmp/cache_{os.environ.get('OCEANSCOPE_RUN', 'main2')}"
RUN = os.environ.get("OCEANSCOPE_RUN", "main2")
PUBLISH = os.environ.get("OCEANSCOPE_PUBLISH", "0") == "1"


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

# Show what an earlier run of this RUN left behind (it resumes from last.pt).
_last = f"{ML}/checkpoints/{RUN}/last.pt"
if os.path.exists(_last):
    import torch

    print(
        "Found earlier checkpoint:",
        _last,
        torch.load(_last, map_location="cpu", weights_only=False)["state"],
    )

# The real part stores (skip the small sample fixtures inside the repo copies).
stores = []
for root, dirs, _ in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    if root.endswith("oceanscope_v1.zarr"):
        stores.append(root)
        dirs[:] = []
stores.sort()
print("Stores:", *stores, sep="\n  ")
S = " ".join(stores)
CFG = "configs/experiment.yaml"

# 1) Stats across all stores, then the OK/FAIL check.
run(f"python -m oceanscope_ml.data.parts stats --config {CFG} --parts {S} --out {STATS}")
run(f"python -m oceanscope_ml.data.parts check --parts {S} --stats {STATS}")

# 2) Train (stops early on its own; 25 is only a ceiling).
run(
    f"python -m oceanscope_ml.train --config {CFG} --store {S} --stats {STATS} "
    f"--cache-dir {CACHE} --device cuda --epochs 25 --run-id {RUN}"
)

# 3) Predict the validation years and evaluate.
run(
    f"python -m oceanscope_ml.inference.predict --config {CFG} --store {S} --stats {STATS} "
    f"--cache-dir {CACHE} --checkpoint checkpoints/{RUN}/best.pt --out-dir {OUT}"
)
EVAL = (
    f"python -m oceanscope_ml.evaluation.evaluate --config {CFG} "
    f"--predictions {OUT}/predictions/predictions_{RUN}.zarr --store {S} --stats {STATS} "
    f"--out-dir {OUT}"
)
run(EVAL)
run(f"cat {OUT}/metrics/{RUN}_val.csv")

# 4) Publish now, so the app has the model even if the baselines below are slow.
run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT} --dry-run")
if PUBLISH:
    run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT}")

# 5) Baselines (for the comparison table), then evaluate and publish again so the
#    manifest lists them.
run(
    f"python -m oceanscope_ml.models.baselines --config {CFG} --store {S} --stats {STATS} "
    f"--out-dir {OUT}/metrics --models-dir checkpoints"
)
run(EVAL)
if PUBLISH:
    run(f"python -m oceanscope_ml.publish --config {CFG} --out-dir {OUT}")
print("\nALL DONE")
