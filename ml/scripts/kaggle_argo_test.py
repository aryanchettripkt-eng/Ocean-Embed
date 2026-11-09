"""Kaggle: the 2021 ARGO test, scored ONCE (DECISIONS 2026-11-06, Aryan's choice on the card).

Run with Save Version -> "Save & Run All (Commit)". GPU recommended (three members predict 365
days each). Internet on (ARGO months 2020-12 to 2021-12 come from the Argo ERDDAP).
Run as: %env OCEANSCOPE_OWNER_GO=1, then
%run /kaggle/working/ocean-embed/ml/scripts/kaggle_argo_test.py

Inputs (Add Input -> Your Work -> Notebooks, finished versions):
- the part stores (``oceanscope_v1.zarr``), including the one that holds 2021;
- the main2 training notebook (``stats.zarr``);
- each member's training notebook (``checkpoints/<run>/best.pt`` or ``best_ft.pt``, picked the
  way kaggle_live_kit.py picks them);
- the run that saved the picked corrections (run F): it holds
  ``predictions_<PICK>.zarr`` (whose attributes give the members and weights),
  ``models/residual_model_<tree run>.txt`` + ``.json`` and ``models/argo_correct_<PICK>.json``.

Single model (DECISIONS 2026-11-09): with OCEANSCOPE_SINGLE=main3_aix30a the inputs are the part
stores, the stats version and that model's fine-tune notebook (``checkpoints/main3_aix30a_l3``);
no ensemble, tree or correction, and OCEANSCOPE_PICK / OCEANSCOPE_BASE are ignored.

Settings (env):
- OCEANSCOPE_OWNER_GO=1: required. Aryan types it; without it the script stops before reading 2021.
- OCEANSCOPE_PICK: the corrected run picked on the validation years (default ens3w_rm2_oi)
- OCEANSCOPE_BASE: the plain ensemble's name (default ens3; scored on the same floats for reference)
- OCEANSCOPE_MEMBER_PICKS: optional ``name=checkpoint_run`` overrides, comma-separated

Nothing is published. The result is final: don't rerun it to try again.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = "/kaggle/working/ocean-embed"
ML = f"{REPO}/ml"
PICK = os.environ.get("OCEANSCOPE_PICK", "ens3w_rm2_oi").strip()
BASE = os.environ.get("OCEANSCOPE_BASE", "ens3").strip()
GO = os.environ.get("OCEANSCOPE_OWNER_GO", "") == "1"
SINGLE = os.environ.get("OCEANSCOPE_SINGLE", "").strip()
if SINGLE:
    PICK = SINGLE
OVERRIDES = dict(
    o.split("=", 1) for o in os.environ.get("OCEANSCOPE_MEMBER_PICKS", "").split(",") if "=" in o
)
OUT = Path("/kaggle/working/outputs_test")
ARGO_DIR = Path("/kaggle/working/argo")
CFG = "configs/experiment.yaml"

if not GO:
    raise SystemExit(
        "The 2021 ARGO test runs once, on Aryan's go. Add a cell line %env OCEANSCOPE_OWNER_GO=1."
    )


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

# 1) Inputs.
stores, stats, picked, ckpts, histories, models = [], None, [], {}, {}, {}
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
    if name == f"predictions_{PICK}.zarr":
        picked.append(Path(root))
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
        if f.startswith(("residual_model_", "argo_correct_")) and f.endswith((".txt", ".json")):
            models.setdefault(f, []).append(Path(root) / f)
stores.sort()
print("Stores:", *stores, sep="\n  ")
print("Stats:", stats)
if not stores or stats is None:
    raise SystemExit("Attach the part stores and the main2 training version (stats.zarr).")
if not SINGLE:
    if len(picked) != 1:
        raise SystemExit(f"Need exactly one predictions_{PICK}.zarr (the run F version); {picked}")
    attrs = json.loads((picked[0] / ".zattrs").read_text())
    member_ids = json.loads(attrs["ensemble_members"])
    weights = json.loads(attrs["ensemble_weights"])
    tree_run = str(attrs["base_run_id"])
    print(f"{PICK}: members {member_ids}, weights {weights}, tree run {tree_run}")


def one(fname):
    paths = models.get(fname, [])
    if len(paths) != 1:
        raise SystemExit(f"Need exactly one {fname} (attach the run F version once): {paths}")
    return paths[0]


if not SINGLE:
    tree_model = one(f"residual_model_{tree_run}.txt")
    one(f"residual_model_{tree_run}.json")
    oi_settings = one(f"argo_correct_{PICK}.json")


def pick(name):
    chosen = OVERRIDES.get(name)
    cands = [chosen] if chosen else [r for r in ckpts if r == name or r.startswith(f"{name}_l")]
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
    if rid not in ckpts or len(ckpts[rid]) != 1:
        raise SystemExit(
            f"{name}: need exactly one checkpoint folder named {rid}: {ckpts.get(rid)}"
        )
    print(f"Member {name}: {ckpts[rid][0]}")
    return ckpts[rid][0]


if SINGLE:
    model_args = f"--single {SINGLE}={pick(SINGLE)}"
    labels = [f"{SINGLE}_test"]
else:
    members = " ".join(f"--member {m}={pick(m)}" for m in member_ids)
    model_args = (
        f"{members} --weights {','.join(f'{w:g}' for w in weights)} --base-id {BASE} "
        f"--residual-model {tree_model} --oi-settings {oi_settings}"
    )
    labels = [f"{PICK}_test", f"{BASE}_test"]

# 2) ARGO months: December 2020 (the correction's window) to December 2021.
months = pd.period_range("2020-12", "2021-12", freq="M")
argo_files = [fetch_month(m, config, ARGO_DIR, allow_test=True) for m in months]
print(f"ARGO months ready: {len(argo_files)}", flush=True)
holdout = Path(ML) / "configs" / "argo_eval_floats.json"

# 3) The test itself (refuses a second time in the same output folder).
run(
    f"python -m oceanscope_ml.evaluation.argo_test --config {CFG} {model_args} "
    f"--store {' '.join(stores)} "
    f"--stats {stats} --argo {' '.join(str(p) for p in argo_files)} --holdout {holdout} "
    f"--out-dir {OUT} --owner-go"
)

M = OUT / "metrics"
for label in labels:
    run(f"cat {M / f'argo_{label}.csv'}")
print("\nRESULT: Bay of Bengal 50-300 m, all held-out floats, 2021 (scored once)")
for label in labels:
    h = pd.read_csv(M / f"{label}_headline.csv")
    print(f"\n{label}")
    print(h.drop(columns=["period"]).round(3).to_string(index=False))
print(json.dumps(json.loads((M / f"argo_test_{PICK}.json").read_text())["results"], indent=2))
print(f"\n2021 ARGO TEST DONE for {PICK} (scored once, nothing published)")
