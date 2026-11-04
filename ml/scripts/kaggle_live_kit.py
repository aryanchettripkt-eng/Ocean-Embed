"""Kaggle: build the live kit for an ensemble run, run the P9-1 check, and upload the kit.

Run with Save Version -> "Save & Run All (Commit)", or cell by cell. No GPU. Internet on.
Run as: %run /kaggle/working/ocean-embed/ml/scripts/kaggle_live_kit.py

Inputs (Add Input -> Your Work -> Notebooks, finished versions):
- each member's training notebook: it holds ``checkpoints/<run>/best.pt`` or ``best_ft.pt``
  (and, for a fine-tune, ``metrics/finetune_<run>_l<lambda>.csv`` for every lambda tried);
- the main2 training notebook, for ``stats.zarr``;
- one part store (``oceanscope_v1.zarr``), only for the static fields.

Secrets: HF_TOKEN (only to upload); for the check also COPERNICUSMARINE_SERVICE_USERNAME,
COPERNICUSMARINE_SERVICE_PASSWORD, EARTHDATA_USERNAME, EARTHDATA_PASSWORD. ARGO profiles come
from the Argo ERDDAP, which needs no account.

Settings (env):
- OCEANSCOPE_RUN: the published name (default ens3)
- OCEANSCOPE_MEMBERS: comma-separated members (default main2_ai,main2_ft,main3). A member is
  matched to the checkpoint named exactly like it, or to its fine-tunes ``<member>_l<lambda>``;
  among fine-tunes the one with the lowest validation ARGO RMSE is taken, as in
  kaggle_finetune.py. ``name=checkpoint_run`` picks one by hand (e.g. main2_ai=main2_ai_l0.1).
- OCEANSCOPE_CHECK_MONTH: P9-1 month (default 2024-10; empty to skip the check)
- OCEANSCOPE_UPLOAD=1: upload the kit to live.kit_repo at the end (default: no upload)
- OCEANSCOPE_BRANCH: repo branch (default main)

The daily live run refuses a kit whose run is not the published one, so upload the kit on the
day the ensemble is published (kaggle_publish_ensemble.py), not before.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
RUN = os.environ.get("OCEANSCOPE_RUN", "ens3")
MEMBERS = [
    m.strip()
    for m in os.environ.get("OCEANSCOPE_MEMBERS", "main2_ai,main2_ft,main3").split(",")
    if m.strip()
]
MONTH = os.environ.get("OCEANSCOPE_CHECK_MONTH", "2024-10").strip()
UPLOAD = os.environ.get("OCEANSCOPE_UPLOAD", "0") == "1"
BRANCH = os.environ.get("OCEANSCOPE_BRANCH", "main")
KIT = Path("/kaggle/working/kit")
OUT = Path("/kaggle/working/outputs_live")
CFG = "configs/experiment.yaml"
SECRETS = (
    "HF_TOKEN",
    "COPERNICUSMARINE_SERVICE_USERNAME",
    "COPERNICUSMARINE_SERVICE_PASSWORD",
    "EARTHDATA_USERNAME",
    "EARTHDATA_PASSWORD",
)


def run(cmd, cwd=ML):
    print("\n$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not os.path.isdir(REPO):
    run(
        f"git clone -b {BRANCH} https://github.com/aryanchettripkt-eng/ocean-embed.git {REPO}",
        cwd="/kaggle/working",
    )
else:
    run(f"git pull origin {BRANCH}", cwd=REPO)
run(f"{sys.executable} -m pip install -q -e .")
os.environ["PYTHONPATH"] = f"{ML}/src"

from kaggle_secrets import UserSecretsClient  # noqa: E402

sec = UserSecretsClient()
for key in SECRETS:
    try:
        os.environ[key] = sec.get_secret(key)
    except Exception:  # noqa: BLE001 - a secret that is not attached
        print(f"Secret {key} not attached")

import pandas as pd  # noqa: E402

# 1) Inputs: checkpoints by the run name of their folder, fine-tune histories, stats, a part.
ckpts, histories, stats, part = {}, {}, None, None
for root, dirs, files in os.walk("/kaggle/input"):
    if "fixtures" in root:
        dirs[:] = []
        continue
    name = os.path.basename(root)
    if name == "stats.zarr":
        stats = stats or root
        dirs[:] = []
        continue
    if name == "oceanscope_v1.zarr":
        part = part or root
        dirs[:] = []
        continue
    if name.endswith(".zarr"):
        dirs[:] = []
        continue
    for f in ("best_ft.pt", "best.pt"):
        if f in files:
            ckpts.setdefault(name, []).append(Path(root) / f)
            break
    for f in files:
        if f.startswith("finetune_") and f.endswith(".csv"):
            histories.setdefault(f[len("finetune_") : -len(".csv")], Path(root) / f)
print("Stats:", stats, "\nPart:", part)
print("Checkpoints found:", *(f"{k}: {v}" for k, v in sorted(ckpts.items())), sep="\n  ")
if stats is None or part is None:
    raise SystemExit("Attach the main2 training version (stats.zarr) and one part store.")


def pick(member):
    """(member name, checkpoint path) for one OCEANSCOPE_MEMBERS entry."""
    name, _, chosen = member.partition("=")
    if chosen:
        cands = [chosen]
    else:
        cands = [r for r in ckpts if r == name or r.startswith(f"{name}_l")]
    if not cands:
        raise SystemExit(f"No checkpoint for {name}: attach its training notebook")
    if len(cands) > 1:
        scores = {}
        for r in cands:
            if r not in histories:
                raise SystemExit(f"{name}: several runs {cands} and no finetune_{r}.csv")
            scores[r] = pd.read_csv(histories[r])["val_argo_rmse_c"].min()
        print(f"{name}: validation ARGO RMSE per run:", scores)
        cands = [min(scores, key=scores.get)]
    rid = cands[0]
    if rid not in ckpts:
        raise SystemExit(f"{name}: no checkpoint folder named {rid}")
    if len(ckpts[rid]) > 1:
        raise SystemExit(f"{name}: {rid} found in several inputs: {ckpts[rid]}. Attach one.")
    print(f"Member {name}: {ckpts[rid][0]}")
    return name, ckpts[rid][0]


chosen = [pick(m) for m in MEMBERS]
members = " ".join(f"--member {n}={p}" for n, p in chosen)

# 2) Kit (uploaded only at the end, after the check).
run(
    f"python -m oceanscope_ml.live kit {members} --run-id {RUN} --stats {stats} "
    f"--static-store {part} --out {KIT}"
)
run(f"cat {KIT}/kit.json | head -40")

# 3) P9-1: NRT vs reprocessed inputs on one past month, for the whole ensemble.
if MONTH:
    run(
        f"python -m oceanscope_ml.evaluation.live_check --config {CFG} --kit-dir {KIT} "
        f"--month {MONTH} --out-dir {OUT}"
    )
    run(f"cat {OUT}/reports/live_mode_check.md")

# 4) Upload.
if UPLOAD:
    run(
        f"python -m oceanscope_ml.live kit {members} --run-id {RUN} --stats {stats} "
        f"--static-store {part} --out {KIT} --upload"
    )
print("\nLIVE KIT DONE" + (" (uploaded)" if UPLOAD else " (not uploaded)"))
