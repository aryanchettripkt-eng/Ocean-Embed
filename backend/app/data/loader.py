"""Data loader for OceanScope backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from app.settings import settings


class DataLoader:
    """Manages loading and validation of output manifest and datasets."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.is_custom_dir = data_dir is not None
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.manifest: dict[str, Any] | None = None
        self._load_error: str | None = None
        self.load(allow_fallback=not self.is_custom_dir)

    def _get_schema_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "schemas" / "manifest.schema.json"

    def load(self, allow_fallback: bool = True) -> None:
        """Attempt to load and validate manifest.json."""
        manifest_file = self.data_dir / "manifest.json"
        if allow_fallback and not manifest_file.exists():
            fallback_dir = (
                Path(__file__).resolve().parent.parent.parent.parent / "fixtures" / "sample_outputs"
            )
            if (fallback_dir / "manifest.json").exists():
                manifest_file = fallback_dir / "manifest.json"
                self.data_dir = fallback_dir

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
