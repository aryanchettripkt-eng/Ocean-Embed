"""One live-mode run: predict the newest complete NRT day and publish it (PRD_AI_ML §12).

Steps, in order. Any error stops the run with a non-zero exit (a red workflow run) before
anything is uploaded, except that a manifest commit can fail after its data commit; the next
run repairs that first (see ``repair_manifest``).

1. Open the live kit (checkpoint, stats, static fields); check it is the active run.
2. Fetch the store skeleton and ``manifest.json`` from the dataset repo.
3. Fetch every NRT input for the last ``max_lag_days`` plus the history window; pick the
   newest day whose inputs and history are all present. Newer days are skipped with the
   reason in the log (no partial maps). No complete day at all fails the run.
4. If that day is already published, stop: nothing new today.
   When a kit member has ARGO input channels, fetch the ARGO profiles of the window before
   that day (held-out floats left out); a failed fetch, or none at all, fails the run.
5. Predict the day (the mean of the members for an ensemble kit), append it (``live``
   label, NaN ``temp_glorys``), trim the rolling window.
6. Upload the changed store files in one commit, then ``manifest.json`` in its own commit.
7. Write the run log (per-input newest day and latency, chosen day, skipped days).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.live import inputs as live_inputs
from oceanscope_ml.live.argo import ArgoFetcher
from oceanscope_ml.live.kit import Kit, argo_window_days, download_kit, open_kit, predict_days
from oceanscope_ml.live.store import (
    PublishedRepo,
    append_day,
    chunk_patterns,
    diff_files,
    file_hashes,
    live_positions,
    skeleton_patterns,
    store_times,
    trim_live,
    update_manifest,
)

logger = logging.getLogger("oceanscope_ml.live.run")


DEFAULT_LIVE_CONFIG = "configs/live.yaml"


def load_live_config(path: str | Path = DEFAULT_LIVE_CONFIG) -> dict[str, Any]:
    """``configs/live.yaml``: separate from experiment.yaml, whose hash training runs check."""
    import yaml

    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


class LiveRunError(RuntimeError):
    """A live run that must show up as failed."""


@dataclass
class LiveSettings:
    sources: dict[str, Any]
    kit_repo: str = ""
    max_lag_days: int = 10
    window_days: int = 365
    trim_slack_days: int = 30
    era5_min_lag_days: int = 5

    @classmethod
    def from_dict(cls, live_cfg: dict[str, Any]) -> LiveSettings:
        raw = dict(live_cfg.get("live") or {})
        if not raw.get("sources"):
            raise ValueError("The live config has no live.sources")
        return cls(
            sources=dict(raw["sources"]),
            kit_repo=str(raw.get("kit_repo", "")),
            max_lag_days=int(raw.get("max_lag_days", 10)),
            window_days=int(raw.get("window_days", 365)),
            trim_slack_days=int(raw.get("trim_slack_days", 30)),
            era5_min_lag_days=int(raw.get("era5_min_lag_days", 5)),
        )


@dataclass
class RunLog:
    today: str
    status: str = "started"
    run_id: str | None = None
    members: list[str] = field(default_factory=list)
    target_day: str | None = None
    published_end_before: str | None = None
    published_end_after: str | None = None
    input_last_day: dict[str, str | None] = field(default_factory=dict)
    input_latency_days: dict[str, int | None] = field(default_factory=dict)
    skipped_days: dict[str, list[str]] = field(default_factory=dict)
    argo_profiles: int | None = None
    argo_floats: int | None = None
    trimmed_days: int = 0
    uploaded_files: int = 0
    deleted_files: int = 0
    notes: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    def markdown(self) -> str:
        lines = [
            f"## Live run {self.today}: {self.status}",
            "",
            f"- Run: `{self.run_id}`",
            f"- Members: {', '.join(self.members) or self.run_id}",
            f"- Target day: {self.target_day or 'none'}",
            f"- ARGO input profiles / floats: {self.argo_profiles} / {self.argo_floats}",
            f"- Published end: {self.published_end_before} → {self.published_end_after}",
            f"- Trimmed live days: {self.trimmed_days}",
            f"- Files uploaded / deleted: {self.uploaded_files} / {self.deleted_files}",
            "",
            "| Input | Newest day | Latency (days) |",
            "|---|---|---|",
        ]
        for var, day in self.input_last_day.items():
            lines.append(f"| {var} | {day or 'none'} | {self.input_latency_days.get(var)} |")
        if self.skipped_days:
            lines += ["", "Skipped days:"]
            lines += [f"- {d}: {'; '.join(why)}" for d, why in self.skipped_days.items()]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def repair_manifest(
    work: Path, manifest: dict[str, Any], store_rel: str, config: ExperimentConfig
) -> pd.Timestamp | None:
    """If an earlier run's data commit landed but its manifest commit did not, fix the end."""
    times = store_times(work / store_rel)
    live = live_positions(times, config)
    end = pd.Timestamp(manifest["available_dates"]["end"])
    if len(live) and times[live[-1]] > end:
        return pd.Timestamp(times[live[-1]])
    return None


def run_live(
    config: ExperimentConfig,
    repo: PublishedRepo,
    kit: Kit,
    work: Path,
    today: pd.Timestamp,
    fetch: live_inputs.Fetcher,
    settings: LiveSettings,
    dry_run: bool = False,
    log: RunLog | None = None,
    argo_fetch: ArgoFetcher | None = None,
) -> RunLog:
    """One live run; ``log`` is filled in as the run goes, so a failure keeps what was found."""
    log = log if log is not None else RunLog(today=str(today.date()))
    log.run_id = kit.run_id
    log.members = [m.name for m in kit.members]
    published = work / "published"
    published.mkdir(parents=True, exist_ok=True)

    # --- 2. skeleton and manifest
    repo.download(["manifest.json"], published)
    manifest_path = published / "manifest.json"
    if not manifest_path.exists():
        raise LiveRunError("The dataset repo has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("synthetic"):
        raise LiveRunError("The published manifest is synthetic; live mode needs a real run")
    if manifest["active_run_id"] != kit.run_id:
        raise LiveRunError(
            f"Active run is {manifest['active_run_id']} but the live kit is for {kit.run_id}; "
            "rebuild the kit for the active run"
        )
    store_rel = str(manifest["files"]["predictions"])
    repo.download(skeleton_patterns(store_rel), published)
    store = published / store_rel
    with xr.open_zarr(store, consolidated=True) as pred:
        if str(pred.attrs.get("run_id")) != kit.run_id:
            raise LiveRunError(
                f"Prediction store run_id {pred.attrs.get('run_id')} != kit {kit.run_id}"
            )
    log.published_end_before = manifest["available_dates"]["end"]

    repaired = repair_manifest(published, manifest, store_rel, config)
    if repaired is not None:
        log.notes.append(
            f"Manifest end repaired to {repaired.date()} (earlier manifest commit lost)"
        )
        if not dry_run:
            update_manifest(manifest_path, repaired)
            repo.commit(published, ["manifest.json"], [], f"Live: repair end {repaired.date()}")
        manifest["available_dates"]["end"] = str(repaired.date())

    # --- 3. inputs
    static = kit.static()
    ocean = static["ocean_mask"].values.astype(bool)
    lookbacks = live_inputs.var_lookbacks(config)
    max_lb = max(lookbacks.values())
    newest = today - pd.Timedelta(days=1)
    oldest = today - pd.Timedelta(days=settings.max_lag_days)
    days = pd.date_range(oldest - pd.Timedelta(days=max_lb), newest, freq="D")
    product_end = {"era5": today - pd.Timedelta(days=settings.era5_min_lag_days)}
    inputs = live_inputs.load_inputs(config, settings.sources, days, ocean, fetch, product_end)
    check = live_inputs.check_inputs(
        live_inputs.daily_coverage(inputs, ocean), lookbacks, newest, oldest
    )
    log.input_last_day = check.last_day
    log.input_latency_days = {
        v: (None if d is None else int((today - pd.Timestamp(d)).days))
        for v, d in check.last_day.items()
    }
    log.skipped_days = check.skipped
    if check.target is None:
        raise LiveRunError(
            f"No day in {oldest.date()}..{newest.date()} has every input; see skipped_days"
        )
    target = check.target
    log.target_day = str(target.date())
    times = store_times(store)
    if target <= times[-1]:
        log.status = "up_to_date"
        log.published_end_after = manifest["available_dates"]["end"]
        log.notes.append(f"Newest complete day {target.date()} is already published")
        return log

    profiles = None
    if kit.needs_argo:
        if argo_fetch is None:
            raise LiveRunError("The kit has ARGO input members but no ARGO fetcher was given")
        argo_start = target - pd.Timedelta(days=argo_window_days(config) - 1)
        profiles = argo_fetch(argo_start, target)
        log.argo_profiles = int(len(profiles))
        log.argo_floats = int(profiles["platform_number"].nunique()) if len(profiles) else 0
        if profiles.empty:
            raise LiveRunError(
                f"No ARGO profiles in {argo_start.date()}..{target.date()}; the ARGO input "
                "channels would be empty"
            )
        held = set(profiles["platform_number"].astype(str)) & kit.holdout()
        if held:
            raise LiveRunError(f"Held-out floats reached the ARGO inputs: {sorted(held)[:5]}")

    # --- 5. predict, append, trim
    before = file_hashes(published)
    window = live_inputs.write_window_store(
        work / "window.zarr", inputs, static, target - pd.Timedelta(days=max_lb), target
    )
    day = predict_days(config, kit, window, [target], argo_profiles=profiles)
    if not (day["coverage"].values == 1).any():
        raise LiveRunError(f"Prediction for {target.date()} has no valid cell")
    append_day(store, day, config)

    def fetch_chunks(positions: range) -> None:
        repo.download(chunk_patterns(store_rel, positions), published)
        before.update(
            {k: v for k, v in file_hashes(published).items() if k not in before}
        )  # fetched chunks are not changes by themselves

    log.trimmed_days = trim_live(
        store, config, settings.window_days, settings.trim_slack_days, fetch_chunks
    )

    # --- 6. upload: data first, manifest last
    add, delete = diff_files(before, file_hashes(published))
    add = [a for a in add if a != "manifest.json"]
    log.uploaded_files, log.deleted_files = len(add), len(delete)
    update_manifest(manifest_path, target)
    log.published_end_after = str(target.date())
    if dry_run:
        log.status = "dry_run"
        log.notes.append("Dry run: nothing uploaded")
        return log
    repo.commit(published, add, delete, f"Live: add {target.date()} (run {kit.run_id})")
    repo.commit(published, ["manifest.json"], [], f"Live: available to {target.date()}")
    log.status = "published"
    return log


def resolve_kit(config: ExperimentConfig, settings: LiveSettings, kit_dir: Path | None) -> Kit:
    if kit_dir is None:
        if not settings.kit_repo:
            raise ValueError("No --kit-dir and no live.kit_repo in the config")
        kit_dir = download_kit(settings.kit_repo, Path(tempfile.mkdtemp(prefix="oceanscope-kit-")))
    return open_kit(kit_dir, config)


def default_argo_fetch(config: ExperimentConfig, kit: Kit, scratch: Path) -> ArgoFetcher | None:
    """ERDDAP ARGO profiles for kits with ARGO input members; None otherwise."""
    if not kit.needs_argo:
        return None
    from oceanscope_ml.live import argo as live_argo

    return live_argo.default_fetch(config, scratch, kit.holdout())


def default_fetch(config: ExperimentConfig, scratch: Path) -> live_inputs.Fetcher:
    r = config.region
    bbox = (r.lon_min, r.lon_max, r.lat_min, r.lat_max)
    return partial(live_inputs.fetch_remote, bbox=bbox, scratch=scratch)


def write_step_summary(log: RunLog) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(log.markdown())


def utc_today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())
