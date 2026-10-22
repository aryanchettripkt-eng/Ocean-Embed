"""Tests verifying .gitignore rules for data directories and source packages (Review R2, R5)."""

import subprocess
from pathlib import Path


def test_gitignore_rules() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Must be IGNORED: pipeline data and outputs
    ignored_paths = [
        "ml/data/raw/x.nc",
        "ml/outputs/x",
        "ml/checkpoints/a.pt",
        "data/raw/x.nc",
        "outputs/x",
        "checkpoints/a.pt",
    ]
    for path in ignored_paths:
        res = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root)
        assert res.returncode == 0, f"{path} should be ignored by .gitignore"

    # Must NOT be ignored: fixtures, source modules in data/ folders, frontend lib files
    tracked_paths = [
        "fixtures/sample_outputs/predictions/predictions_synthetic_v1.zarr/.zattrs",
        "ml/src/oceanscope_ml/data/download.py",
        "backend/app/data/store.py",
        "backend/app/data/loader.py",
        "frontend/src/lib/api/types.ts",
    ]
    for path in tracked_paths:
        res = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root)
        assert res.returncode != 0, f"{path} should NOT be ignored by .gitignore"
