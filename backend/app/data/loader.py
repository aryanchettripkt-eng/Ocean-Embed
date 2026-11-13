"""Data loader for OceanScope backend."""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jsonschema

from app.settings import settings

logger = logging.getLogger(__name__)

# Remembers the blob id of every file fetched, so unchanged files are not fetched again.
_INDEX_FILE = ".hf-sync-index.json"
# Parallel downloads. Each holds a small buffer; few workers keep memory and the HF rate low.
_DOWNLOAD_WORKERS = 4


class DataLoader:
    """Manages loading and validation of output manifest and datasets."""

    def __init__(self, data_dir: str | Path | None = None, autoload: bool = True) -> None:
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.manifest: dict[str, Any] | None = None
        self._load_error: str | None = "Data store is still loading"
        self._current_hf_revision: str | None = None
        self._current_hf_commit_sha: str | None = None
        self._sync_error: str | None = None
        if autoload:
            self.load()

    def _get_schema_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "schemas" / "manifest.schema.json"

    @staticmethod
    def _manifest_paths(manifest: dict[str, Any]) -> list[str]:
        """Repo paths (files or folders) that a manifest's `files` block points at."""
        paths: list[str] = []
        for value in manifest.get("files", {}).values():
            for rel in value if isinstance(value, list) else [value]:
                if isinstance(rel, str) and rel.strip("/"):
                    paths.append(rel.strip("/"))
        return paths

    def _read_index(self) -> dict[str, str]:
        try:
            with open(self.data_dir / _INDEX_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_index(self, index: dict[str, str]) -> None:
        tmp = self.data_dir / f"{_INDEX_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f)
        os.replace(tmp, self.data_dir / _INDEX_FILE)

    def _sync_from_hf(self) -> None:
        """Bring the active run from the HF dataset to DATA_DIR, manifest last.

        Only the files the remote manifest names are listed and fetched (the dataset holds
        every run ever published, and listing all of it took a lot of memory). Files whose
        blob id matches the last sync are skipped, so a new live day fetches a few chunks.
        manifest.json is replaced only after every file arrived, so a failed or partial
        download never points the server at a half-copied run; the next reload retries.
        A remote manifest that can't be read means "try again later", never "download all".
        """
        self._sync_error = None
        if not settings.HF_DATASET_REPO:
            return
        from huggingface_hub import HfApi, hf_hub_download

        repo = settings.HF_DATASET_REPO
        revision = settings.HF_DATASET_REVISION or "main"
        token = (
            settings.HF_TOKEN.strip() if (settings.HF_TOKEN and settings.HF_TOKEN.strip()) else None
        )
        manifest_file = self.data_dir / "manifest.json"
        api = HfApi(token=token)

        try:
            remote_sha = api.dataset_info(repo_id=repo, revision=revision).sha
        except Exception as e:
            logger.warning("Could not check the HF dataset, keeping local data: %s", e)
            if not manifest_file.exists():
                self._sync_error = f"Could not reach the HF dataset: {e}"
            return
        if (
            manifest_file.exists()
            and remote_sha
            and remote_sha == self._current_hf_commit_sha
            and self._current_hf_revision == revision
        ):
            return

        try:
            remote_manifest_path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename="manifest.json",
                revision=remote_sha or revision,
                token=token,
            )
            with open(remote_manifest_path, encoding="utf-8") as f:
                remote_manifest = json.load(f)

            wanted: dict[str, str] = {}
            paths = self._manifest_paths(remote_manifest)
            infos = api.get_paths_info(
                repo_id=repo, paths=paths, repo_type="dataset", revision=remote_sha or revision
            )
            for info in infos:
                blob = getattr(info, "blob_id", None)
                if blob is not None:
                    wanted[info.path] = str(blob)
                    continue
                for entry in api.list_repo_tree(
                    repo_id=repo,
                    path_in_repo=info.path,
                    recursive=True,
                    repo_type="dataset",
                    revision=remote_sha or revision,
                ):
                    entry_blob = getattr(entry, "blob_id", None)
                    if entry_blob is not None:
                        wanted[entry.path] = str(entry_blob)

            self.data_dir.mkdir(parents=True, exist_ok=True)
            index = self._read_index()
            todo = [
                path
                for path, blob in wanted.items()
                if index.get(path) != blob or not (self.data_dir / path).exists()
            ]
            logger.info(
                "Syncing HF dataset %s at %s: %d files in the run, %d to fetch",
                repo,
                remote_sha,
                len(wanted),
                len(todo),
            )

            def fetch(path: str) -> str:
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=path,
                    revision=remote_sha or revision,
                    local_dir=str(self.data_dir),
                    token=token,
                )
                return path

            failed: Exception | None = None
            with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
                for future in as_completed([pool.submit(fetch, p) for p in todo]):
                    try:
                        done = future.result()
                    except Exception as e:  # keep the files that did arrive for the retry
                        failed = failed or e
                        continue
                    index[done] = wanted[done]
            self._write_index(index)
            if failed is not None:
                raise failed

            tmp = self.data_dir / "manifest.json.tmp"
            shutil.copyfile(remote_manifest_path, tmp)
            os.replace(tmp, manifest_file)
            self._current_hf_revision = revision
            self._current_hf_commit_sha = remote_sha
        except Exception as e:
            logger.error("Failed to sync the dataset from HF, keeping local data: %s", e)
            if not manifest_file.exists():
                self._sync_error = f"Failed to download dataset from HF: {e}"

    def load(self, data_dir: str | Path | None = None) -> None:
        """Attempt to load and validate manifest.json."""
        if data_dir is not None:
            self.data_dir = Path(data_dir)

        self._sync_from_hf()

        manifest_file = self.data_dir / "manifest.json"
        if not manifest_file.exists():
            if self.manifest is not None:
                logger.error("Manifest disappeared on reload, keeping current valid manifest")
            else:
                self._load_error = self._sync_error or f"Manifest not found in {self.data_dir}"
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
            old_manifest = self.manifest
            if old_manifest is not None and raw_manifest == old_manifest:
                # Nothing published changed (or a sync stopped part-way): keep the open store
                # and its caches, so an hourly check costs nothing and never shows a half-synced
                # store. A live day always changes the manifest (its available end date).
                self._load_error = None
                return
            self.manifest = raw_manifest
            self._load_error = None

            if old_manifest is not None:
                try:
                    from app.data.store import store

                    store.clear_caches()
                except Exception:
                    pass
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


# Loaded by the app's lifespan in a background thread, so the port binds before any download.
loader = DataLoader(autoload=False)
