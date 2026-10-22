"""Data loader for OceanScope backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema

from app.settings import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Manages loading and validation of output manifest and datasets."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.manifest: dict[str, Any] | None = None
        self._load_error: str | None = None
        self.load()

    def _get_schema_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "schemas" / "manifest.schema.json"

    def _maybe_download_from_hf(self) -> None:
        """Download dataset from HF Hub if configured and manifest.json is absent."""
        manifest_file = self.data_dir / "manifest.json"
        if not manifest_file.exists() and settings.HF_DATASET_REPO:
            try:
                from huggingface_hub import snapshot_download

                logger.info(
                    "Downloading dataset from HF: %s (revision: %s) to %s",
                    settings.HF_DATASET_REPO,
                    settings.HF_DATASET_REVISION,
                    self.data_dir,
                )
                snapshot_download(
                    repo_id=settings.HF_DATASET_REPO,
                    repo_type="dataset",
                    revision=settings.HF_DATASET_REVISION,
                    local_dir=str(self.data_dir),
                    token=settings.HF_TOKEN,
                )
            except Exception as e:
                logger.error("Failed to download dataset from HF: %s", e)
                self._load_error = f"Failed to download dataset from HF: {e}"

    def load(self, data_dir: str | Path | None = None) -> None:
        """Attempt to load and validate manifest.json."""
        if data_dir is not None:
            self.data_dir = Path(data_dir)

        self._maybe_download_from_hf()

        manifest_file = self.data_dir / "manifest.json"
        if not manifest_file.exists():
            self._load_error = f"Manifest not found in {self.data_dir}"
            self.manifest = None
            return

        try:
            with open(manifest_file, encoding="utf-8") as f:
                raw_manifest = json.load(f)

            schema_path = self._get_schema_path()
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)

            jsonschema.validate(
                instance=raw_manifest,
                schema=schema,
                format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
            )
            self.manifest = raw_manifest
            self._load_error = None
        except Exception as e:
            # If we already had a valid manifest, keep serving it and log error (PRD §9 test 12)
            if self.manifest is not None:
                logger.error("Failed to reload new manifest, keeping current valid manifest: %s", e)
            else:
                self._load_error = f"Manifest validation error: {e}"
                self.manifest = None

    @property
    def is_loaded(self) -> bool:
        """Return True if manifest is loaded and valid."""
        return self.manifest is not None

    @property
    def active_run_id(self) -> str | None:
        """Return active_run_id from manifest if loaded."""
        if self.manifest:
            run_id = self.manifest.get("active_run_id")
            return str(run_id) if run_id is not None else None
        return None

    @property
    def load_error(self) -> str | None:
        """Return load error message if failed."""
        return self._load_error


loader = DataLoader()
