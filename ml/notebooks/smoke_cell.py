# Smoke run: two weeks of each risky year, built as separate parts, then the stats
# step, the OK/FAIL check and the baselines, exactly as the full build will run them.
# Run after cells 1-3 (clone + install, secrets, preflight, which sets STATIC).
# Keeps going after a failed year so one run shows every error. Target: 30-45 min.
import os
import subprocess
import sys
import time
from pathlib import Path

SMOKE_YEARS = [1993, 2000, 2007, 2014, 2020, 2021]  # 2021 is built only, never evaluated
SMOKE_DAYS = 14
OUT = Path("/kaggle/working/smoke")
OUT.mkdir(parents=True, exist_ok=True)
# `%run` gets a fresh namespace, so look for the attached oceanscope-static dataset.
# Kaggle mounts inputs at different depths, so walk /kaggle/input (skipping big stores).
static = globals().get("STATIC")
if not static:
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if not d.endswith(".zarr") and d not in (".git", "node_modules")]
        if "static_fields.nc" in files:
            static = Path(root) / "static_fields.nc"
            break
print("static file:", static)
if static is None:
    # Without it every year fails: the GEBCO download URL returns 404.
    raise SystemExit(
        "static_fields.nc not found under /kaggle/input. Attach the oceanscope-static "
        "dataset (Add Input -> Your Work -> Datasets) and run again."
    )


def run(name, args, show_tail=True):
    """Run one step, log it to OUT/<name>.log, and print its time and outcome."""
    log = OUT / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(
            [sys.executable, "-u", "-m", *args], stdout=f, stderr=subprocess.STDOUT
        ).returncode
    mins = (time.time() - t0) / 60
    print(f"{'OK  ' if rc == 0 else 'FAIL'} {name:<12} {mins:5.1f} min   log: {log}")
    if rc != 0 and show_tail:
        print("  " + "\n  ".join(log.read_text().splitlines()[-25:]))
    return rc == 0


parts = []
for yr in SMOKE_YEARS:
    part = OUT / f"part_{yr}"
    args = [
        "oceanscope_ml.data.build_store",
        "--config", "configs/experiment.yaml",
        "--years", str(yr),
        "--out-dir", str(part),
        "--scratch-dir", f"/tmp/smoke_scratch/{yr}",
        "--stream", "--part",
        "--smoke-days", str(SMOKE_DAYS),
    ]  # fmt: skip
    if static:
        args += ["--static-file", str(static)]
    if run(f"build {yr}", args):
        parts.append(str(part / "oceanscope_v1.zarr"))

stats = str(OUT / "stats.zarr")
ok_stats = bool(parts) and run(
    "stats",
    ["oceanscope_ml.data.parts", "stats", "--config", "configs/experiment.yaml",
     "--parts", *parts, "--out", stats],
)  # fmt: skip

if parts:
    check = ["oceanscope_ml.data.parts", "check", "--parts", *parts,
             "--years", ",".join(map(str, SMOKE_YEARS))]  # fmt: skip
    if ok_stats:
        check += ["--stats", stats]
    run("check", check, show_tail=False)
    print((OUT / "check.log").read_text())

if ok_stats:
    run(
        "baselines",
        ["oceanscope_ml.models.baselines", "--config", "configs/experiment.yaml",
         "--store", *parts, "--stats", stats, "--out-dir", str(OUT / "metrics"),
         "--models-dir", str(OUT / "models")],
    )  # fmt: skip
    print("\n".join((OUT / "baselines.log").read_text().splitlines()[-22:]))

print("\nSmoke parts built:", len(parts), "of", len(SMOKE_YEARS))
