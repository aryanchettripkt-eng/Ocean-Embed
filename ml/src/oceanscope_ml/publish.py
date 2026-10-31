"""Upload one run's outputs to the Hugging Face dataset repo (PRD_AI_ML §4 stage 9, §6.4).

Usage (from ml/, with HF_TOKEN set to a write token):

    python -m oceanscope_ml.publish --config configs/experiment.yaml [--out-dir outputs] \
        [--repo <org>/<repo>] [--dry-run]

Layout in the repo (the backend downloads exactly this tree):

    manifest.json
    predictions/predictions_<run_id>.zarr/...
    metrics/*.csv, metrics/*.parquet
    reports/evaluation_<run_id>.md   (when it exists)

Before anything is uploaded, the manifest is validated against the schema, every file it
names must exist, and the prediction store must open the way the backend opens it. The data
files go first and ``manifest.json`` goes last, in its own commit, so the backend never sees a
manifest that points at files not uploaded yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config

logger = logging.getLogger("oceanscope_ml.publish")

DATA_DIRS = ("predictions", "metrics", "reports")


def manifest_files(manifest: dict[str, Any]) -> list[str]:
    """Relative paths of every file the manifest names (nulls skipped)."""
    paths: list[str] = []
    for value in manifest["files"].values():
        if isinstance(value, list):
            paths.extend(str(v) for v in value)
        elif value:
            paths.append(str(value))
    return paths


def check_outputs(out_dir: Path) -> dict[str, Any]:
    """Validate the manifest and the files it names; return the manifest."""
    import jsonschema

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {out_dir}; run evaluation.evaluate first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parent / "schemas" / "manifest.schema.json"
    jsonschema.validate(
        manifest,
        json.loads(schema_path.read_text(encoding="utf-8")),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    if manifest["synthetic"]:
        raise ValueError("This manifest is marked synthetic; publish only real runs")
    missing = [p for p in manifest_files(manifest) if not (out_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"Files named in manifest.json are missing: {missing}")

    # Open the predictions exactly as the backend does (app/data/store.py).
    pred = xr.open_zarr(str(out_dir / manifest["files"]["predictions"]), consolidated=True)
    if str(pred.attrs.get("run_id")) != manifest["active_run_id"]:
        raise ValueError("Prediction store run_id does not match manifest active_run_id")
    for var in ("temp_pred", "temp_glorys", "coverage"):
        if var not in pred:
            raise KeyError(f"Prediction store lacks {var}")
    return manifest


def _allow_patterns(manifest: dict[str, Any], out_dir: Path) -> list[str]:
    """Upload only this run's files, so stale local outputs never reach the repo."""
    patterns: list[str] = []
    for rel in manifest_files(manifest):
        patterns.append(f"{rel}/**" if (out_dir / rel).is_dir() else rel)
    report = f"reports/evaluation_{manifest['active_run_id']}.md"
    if (out_dir / report).exists():
        patterns.append(report)
    return patterns


def publish(
    config: ExperimentConfig,
    out_dir: Path | None = None,
    repo_id: str | None = None,
    dry_run: bool = False,
    api: Any = None,
) -> dict[str, Any]:
    """Check the outputs, then upload data files and finally manifest.json."""
    out = Path(out_dir or config.paths.outputs)
    manifest = check_outputs(out)
    repo = repo_id or str(config.publish.get("hf_dataset_repo", ""))
    if not repo:
        raise ValueError("No Hugging Face repo: set publish.hf_dataset_repo or pass --repo")
    patterns = _allow_patterns(manifest, out)
    run_id = manifest["active_run_id"]
    logger.info("Run %s: %d files/folders to publish to %s", run_id, len(patterns), repo)
    for p in patterns:
        logger.info("  %s", p)
    if dry_run:
        logger.info("Dry run: nothing uploaded.")
        return manifest

    if api is None:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is not set (use a Kaggle/Colab secret, never git)")
        api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=patterns,
        commit_message=f"Publish outputs for run {run_id}",
    )
    # manifest.json last, in its own commit (PRD §6.4).
    api.upload_file(
        path_or_fileobj=str(out / "manifest.json"),
        path_in_repo="manifest.json",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Activate run {run_id}",
    )
    logger.info("Published run %s: https://huggingface.co/datasets/%s", run_id, repo)
    return manifest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a run's outputs to Hugging Face.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    parser.add_argument("--repo", default=None, help="Default: publish.hf_dataset_repo")
    parser.add_argument("--dry-run", action="store_true", help="Check and list, upload nothing")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    publish(
        load_config(args.config),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        repo_id=args.repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
