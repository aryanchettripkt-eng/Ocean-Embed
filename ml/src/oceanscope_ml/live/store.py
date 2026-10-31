"""Append live days to the published prediction store (PRD_AI_ML §6.2, §6.4, §12).

The published store is not downloaded whole. A live run fetches only the store's metadata
and coordinate arrays (a "skeleton"; chunks that are not on disk read as missing), appends the
new day locally, and uploads just the files that changed in one commit. ``manifest.json``
follows in its own commit, so the backend never sees an end date whose data is not there.

Live days are labelled ``live`` in ``split_of_each_date`` and have all-NaN ``temp_glorys``.
The live part is kept to a rolling window: when it grows past ``window_days + trim_slack_days``
the oldest live days are dropped down to ``window_days`` (so the rewrite happens about once a
month, not every day).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.inference.predict import split_runs

logger = logging.getLogger("oceanscope_ml.live.store")

LIVE_LABEL = "live"
STORE_VARS = ("temp_pred", "temp_glorys", "coverage")
COORD_VARS = ("time", "depth", "lat", "lon")


class PublishedRepo(Protocol):
    """Where the published outputs live (the Hugging Face dataset repo, or a folder)."""

    def download(self, patterns: list[str], local_dir: Path) -> None: ...

    def commit(self, local_dir: Path, add: list[str], delete: list[str], message: str) -> None: ...


class LocalRepo:
    """A folder standing in for the dataset repo (tests and local dry runs)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _files(self) -> list[str]:
        return sorted(
            p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()
        )

    def download(self, patterns: list[str], local_dir: Path) -> None:
        for rel in self._files():
            if any(fnmatch.fnmatch(rel, pat) for pat in patterns):
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / rel, dest)

    def commit(self, local_dir: Path, add: list[str], delete: list[str], message: str) -> None:
        for rel in delete:
            (self.root / rel).unlink(missing_ok=True)
        for rel in add:
            dest = self.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_dir / rel, dest)
        logger.info("Committed to %s: %s (+%d, -%d)", self.root, message, len(add), len(delete))


class HfRepo:
    """The Hugging Face dataset repo of PRD §6.4."""

    def __init__(self, repo_id: str, api: Any = None) -> None:
        self.repo_id = repo_id
        self._api = api

    def _token(self) -> str:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is not set (a GitHub secret in the live workflow)")
        return token

    @property
    def api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self._token())
        return self._api

    def download(self, patterns: list[str], local_dir: Path) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(local_dir),
            token=os.environ.get("HF_TOKEN", "").strip() or None,
        )

    def commit(self, local_dir: Path, add: list[str], delete: list[str], message: str) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        ops: list[Any] = [CommitOperationDelete(path_in_repo=rel) for rel in delete]
        ops += [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(local_dir / rel))
            for rel in add
        ]
        if not ops:
            return
        self.api.create_commit(
            repo_id=self.repo_id, repo_type="dataset", operations=ops, commit_message=message
        )
        logger.info("Committed to %s: %s (+%d, -%d)", self.repo_id, message, len(add), len(delete))


# ------------------------------------------------------------------ skeleton


def skeleton_patterns(store_rel: str) -> list[str]:
    """Metadata files and coordinate arrays of a zarr v2 store (no data chunks)."""
    return [
        f"{store_rel}/.z*",
        f"{store_rel}/*/.z*",
        *[f"{store_rel}/{c}/*" for c in COORD_VARS],
    ]


def chunk_patterns(store_rel: str, positions: range) -> list[str]:
    """Data chunk files of the given time positions (one day per chunk, PRD §6.2)."""
    return [f"{store_rel}/{v}/{i}.*" for i in positions for v in STORE_VARS]


def file_hashes(root: Path) -> dict[str, str]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def diff_files(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str]]:
    """(added or changed, deleted) relative paths."""
    add = sorted(k for k, h in after.items() if before.get(k) != h)
    delete = sorted(k for k in before if k not in after)
    return add, delete


# ------------------------------------------------------------------ store edits


def store_times(store: Path) -> pd.DatetimeIndex:
    with xr.open_zarr(store, consolidated=True) as ds:
        return pd.DatetimeIndex(ds["time"].values)


def live_positions(times: pd.DatetimeIndex, config: ExperimentConfig) -> np.ndarray:
    """Positions of live days: every date after the test period."""
    return np.flatnonzero(times > pd.Timestamp(config.splits.test[1]))


def _relabel(store: Path, config: ExperimentConfig) -> None:
    """Rewrite ``split_of_each_date`` live entries from the store's live dates."""
    times = store_times(store)
    group = zarr.open_group(str(store), mode="r+", zarr_format=2)
    raw = group.attrs.get("split_of_each_date", "{}")
    if not isinstance(raw, str):
        raise ValueError("split_of_each_date must be a JSON string attribute (PRD §6.2)")
    splits: dict[str, Any] = json.loads(raw)
    splits = {k: v for k, v in splits.items() if v != LIVE_LABEL}
    splits.update(split_runs(times[live_positions(times, config)], LIVE_LABEL))
    group.attrs["split_of_each_date"] = json.dumps(splits)
    zarr.consolidate_metadata(str(store))


def append_day(store: Path, day: xr.Dataset, config: ExperimentConfig) -> None:
    """Append one predicted day (temp_pred, coverage) with NaN temp_glorys, labelled live."""
    if day.sizes["time"] != 1:
        raise ValueError("append_day appends exactly one day")
    date = pd.Timestamp(day["time"].values[0])
    test_end = pd.Timestamp(config.splits.test[1])
    if date <= test_end:
        raise ValueError(f"Live date {date.date()} is not after the test period")
    with xr.open_zarr(store, consolidated=True) as ds:
        last = pd.Timestamp(ds["time"].values[-1])
        if date <= last:
            raise ValueError(f"Store already reaches {last.date()}; cannot append {date.date()}")
        for c in ("depth", "lat", "lon"):
            if not np.allclose(ds[c].values, day[c].values):
                raise ValueError(f"Live day {c} does not match the published store")
        dtypes = {v: ds[v].dtype for v in STORE_VARS}
        attrs = dict(ds.attrs)
    out = xr.Dataset(
        {
            "temp_pred": day["temp_pred"].astype(dtypes["temp_pred"]),
            "temp_glorys": xr.full_like(day["temp_pred"], np.nan).astype(dtypes["temp_glorys"]),
            "coverage": day["coverage"].astype(dtypes["coverage"]),
        }
    )
    out = out.drop_vars(["depth", "lat", "lon"])
    out.attrs = attrs  # to_zarr rewrites the group attrs; keep run_id and the rest
    out.to_zarr(store, append_dim="time", consolidated=True, zarr_format=2)
    _relabel(store, config)


def trim_live(
    store: Path,
    config: ExperimentConfig,
    window_days: int,
    slack_days: int,
    fetch_chunks: Any = None,
) -> int:
    """Drop the oldest live days once there are more than window + slack; returns days dropped.

    ``fetch_chunks(positions)`` must bring the data chunks of those positions to local disk
    first (the skeleton holds none).
    """
    times = store_times(store)
    live = live_positions(times, config)
    if len(live) <= window_days + slack_days:
        return 0
    first, n = int(live[0]), len(times)
    if not np.array_equal(live, np.arange(first, n)):
        raise ValueError("Live days must be the last days of the store")
    drop = len(live) - window_days
    if fetch_chunks is not None:
        fetch_chunks(range(first, n))
    new_n = n - drop
    for name in (*STORE_VARS, "time"):
        arr = zarr.open_array(str(store / name), mode="r+", zarr_format=2)
        arr[first:new_n] = arr[first + drop : n]
        arr.resize((new_n, *arr.shape[1:]))
    # Chunk files past the new end are not part of the store any more.
    for name in STORE_VARS:
        for f in (store / name).iterdir():
            head = f.name.split(".")[0]
            if head.isdigit() and int(head) >= new_n:
                f.unlink()
    _relabel(store, config)
    logger.info("Trimmed %d oldest live days (kept %d)", drop, window_days)
    return drop


def update_manifest(manifest_path: Path, end: pd.Timestamp) -> dict[str, Any]:
    """Set ``available_dates.end`` and validate against the schema."""
    import jsonschema

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["available_dates"]["end"] = str(end.date())
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "manifest.schema.json"
    jsonschema.validate(
        manifest,
        json.loads(schema_path.read_text(encoding="utf-8")),
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
