"""Deploy script for OceanScope AI Hugging Face dataset and backend Space (Phase 2).

Uploads sample outputs to an HF dataset repository and deploys the backend service
to an HF Space running Docker SDK.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_hf_token() -> str:
    """Retrieve Hugging Face write token from environment or .env file."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    # Look in root .env or backend .env
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if token:
                        return token
    return ""


def deploy_dataset(
    api: Any,
    repo_id: str,
    source_dir: Path,
    dry_run: bool = False,
) -> None:
    """Create dataset repo on HF and upload outputs."""
    logger.info("Target Dataset Repo: %s", repo_id)
    if dry_run:
        logger.info(
            "[DRY-RUN] Would create dataset repo %s and upload %s",
            repo_id,
            source_dir,
        )
        return

    logger.info("Creating / checking dataset repository %s...", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    logger.info("Uploading %s to dataset repo %s...", source_dir, repo_id)
    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish outputs (OceanScope AI)",
    )
    logger.info("Dataset upload complete: https://huggingface.co/datasets/%s", repo_id)


def deploy_space(
    api: Any,
    space_id: str,
    dataset_repo_id: str,
    backend_dir: Path,
    dry_run: bool = False,
) -> None:
    """Create Docker Space on HF, upload backend files, and configure environment variables."""
    logger.info("Target Backend Space: %s", space_id)
    if dry_run:
        logger.info(
            "[DRY-RUN] Would create Space %s (sdk=docker) and upload backend files",
            space_id,
        )
        return

    logger.info("Creating / checking Space %s (sdk=docker)...", space_id)
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
    except Exception as e:
        logger.warning(
            "create_repo returned '%s'. Checking if space already exists...",
            e,
        )
        try:
            api.space_info(repo_id=space_id)
            logger.info("Space %s exists. Proceeding with deployment...", space_id)
        except Exception:
            raise e

    # Prepare deployment bundle in a clean temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir)

        # Copy Dockerfile
        shutil.copy2(backend_dir / "Dockerfile", staging_dir / "Dockerfile")

        # Copy pyproject.toml
        shutil.copy2(backend_dir / "pyproject.toml", staging_dir / "pyproject.toml")

        # Copy README.md (with HF YAML frontmatter)
        shutil.copy2(backend_dir / "README.md", staging_dir / "README.md")

        # Copy openapi.json
        if (backend_dir / "openapi.json").exists():
            shutil.copy2(backend_dir / "openapi.json", staging_dir / "openapi.json")

        # Copy app package
        shutil.copytree(
            backend_dir / "app",
            staging_dir / "app",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        logger.info("Uploading backend code to Space %s...", space_id)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy OceanScope AI backend (FastAPI Docker SDK)",
        )

    # Configure Space variables for dataset loading and CORS
    logger.info("Setting Space variables on %s...", space_id)
    variables = {
        "HF_DATASET_REPO": dataset_repo_id,
        "DATA_DIR": "/app/data",
        "CORS_ORIGINS": "*",
        "RELOAD_INTERVAL_MIN": "60",
    }
    for key, val in variables.items():
        try:
            api.add_space_variable(repo_id=space_id, key=key, value=val)
            logger.info("  Set %s = %s", key, val)
        except Exception as e:
            logger.debug("Could not set space variable %s (may already exist): %s", key, e)

    space_host = space_id.replace("/", "-")
    public_url = f"https://{space_host}.hf.space"
    logger.info("Space deployment submitted!")
    logger.info("Space Dashboard: https://huggingface.co/spaces/%s", space_id)
    logger.info("Public API URL:  %s/api/v1/health", public_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy OceanScope AI dataset & backend to Hugging Face"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying remote HF Hub",
    )
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset upload")
    parser.add_argument("--skip-space", action="store_true", help="Skip backend space deployment")
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="",
        help="HF dataset repo ID (<org>/<repo>)",
    )
    parser.add_argument(
        "--space-repo",
        type=str,
        default="",
        help="HF space repo ID (<org>/<repo>)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "fixtures" / "sample_outputs",
        help="Local directory with outputs and manifest.json",
    )
    args = parser.parse_args()

    token = get_hf_token()
    if not token and not args.dry_run:
        logger.error("No HF_TOKEN found in environment or .env file.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub is required. Install via `pip install huggingface_hub`.")
        sys.exit(1)

    api = HfApi(token=token if token else None)
    if not args.dry_run:
        user_info = api.whoami()
        username = str(user_info["name"])
        logger.info("Authenticated as Hugging Face user: %s", username)
    else:
        username = "user"

    dataset_repo = args.dataset_repo or f"{username}/oceanscope-outputs"
    space_repo = args.space_repo or f"{username}/oceanscope-api"
    backend_dir = Path(__file__).resolve().parent

    if not args.skip_dataset:
        deploy_dataset(api, dataset_repo, args.data_dir, dry_run=args.dry_run)

    if not args.skip_space:
        deploy_space(api, space_repo, dataset_repo, backend_dir, dry_run=args.dry_run)

    logger.info("P2-1 Hugging Face actions completed successfully.")


if __name__ == "__main__":
    main()
