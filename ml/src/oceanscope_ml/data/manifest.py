"""Manager for data/raw/MANIFEST.json tracking downloaded products, checksums, and metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute sha256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def load_manifest(raw_dir: Path | str) -> dict[str, Any]:
    """Load existing MANIFEST.json or return a fresh schema."""
    manifest_path = Path(raw_dir) / "MANIFEST.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    return {
        "manifest_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "products": {},
    }


def save_manifest(raw_dir: Path | str, manifest: dict[str, Any]) -> Path:
    """Save manifest dict to data/raw/MANIFEST.json."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()

    manifest_file = raw_path / "MANIFEST.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_file


def record_planned_product(
    manifest: dict[str, Any],
    product_id: str,
    source_id: str,
    source_url: str,
    variables: list[str],
    date_range: dict[str, str],
    estimated_size_mb: float,
    notes: str = "",
) -> None:
    """Record a planned product in the manifest."""
    manifest["products"][product_id] = {
        "source_id": source_id,
        "source_url": source_url,
        "variables": variables,
        "date_range": date_range,
        "estimated_size_mb": estimated_size_mb,
        "notes": notes,
        "status": "planned",
        "files": manifest["products"].get(product_id, {}).get("files", []),
    }


def record_downloaded_file(
    manifest: dict[str, Any],
    product_id: str,
    file_path: Path,
    raw_dir: Path,
    source_id: str = "",
) -> None:
    """Record a newly downloaded or verified file in the manifest."""
    if product_id not in manifest["products"]:
        manifest["products"][product_id] = {
            "source_id": source_id,
            "source_url": "",
            "variables": [],
            "date_range": {},
            "estimated_size_mb": 0.0,
            "status": "in_progress",
            "files": [],
        }

    rel_path = str(file_path.relative_to(raw_dir).as_posix())
    size_bytes = file_path.stat().st_size
    sha256_hash = compute_sha256(file_path)

    # Update or append file entry
    files_list = manifest["products"][product_id]["files"]
    existing_idx = next(
        (i for i, f in enumerate(files_list) if f.get("rel_path") == rel_path), None
    )

    file_entry = {
        "rel_path": rel_path,
        "size_bytes": size_bytes,
        "sha256": sha256_hash,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }

    if existing_idx is not None:
        files_list[existing_idx] = file_entry
    else:
        files_list.append(file_entry)

    manifest["products"][product_id]["status"] = "downloaded"
