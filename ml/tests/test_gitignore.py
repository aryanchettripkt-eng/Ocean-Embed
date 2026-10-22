"""Tests verifying .gitignore rules for ML data directories and committed fixtures (Review R2)."""

import subprocess
from pathlib import Path


def test_gitignore_rules() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent

    # 1. ml/data/raw/x.nc must be ignored
    res1 = subprocess.run(
        ["git", "check-ignore", "-q", "ml/data/raw/x.nc"],
        cwd=repo_root,
    )
    assert res1.returncode == 0, "ml/data/raw/x.nc should be ignored by .gitignore"

    # 2. ml/outputs/x must be ignored
    res2 = subprocess.run(
        ["git", "check-ignore", "-q", "ml/outputs/x"],
        cwd=repo_root,
    )
    assert res2.returncode == 0, "ml/outputs/x should be ignored by .gitignore"

    # 3. fixtures/.../.zattrs must NOT be ignored
    res3 = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "fixtures/sample_outputs/predictions/predictions_synthetic_v1.zarr/.zattrs",
        ],
        cwd=repo_root,
    )
    assert res3.returncode != 0, "fixtures sample outputs must NOT be ignored by .gitignore"
