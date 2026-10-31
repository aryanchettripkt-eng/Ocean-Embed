"""Live mode CLI (PRD_AI_ML §12, PHASES P9-3).

Usage (from ml/):

    # Once per published run: pack the kit, and upload it to live.kit_repo (needs HF_TOKEN)
    python -m oceanscope_ml.live kit --checkpoint checkpoints/<run_id>/best.pt \
        --stats <stats.zarr> --static-store <part.zarr> --out kit/ [--upload]

    # Every day (the scheduled workflow): one new day, or nothing if it is already published
    python -m oceanscope_ml.live run [--kit-dir kit/] [--dry-run] [--log outputs/live/log.json]

``run`` needs COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD, EARTHDATA_USERNAME/PASSWORD, a CDS key
(CDSAPI_URL + CDSAPI_KEY) and HF_TOKEN in the environment. Any error exits non-zero.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

from oceanscope_ml.config import load_config
from oceanscope_ml.live.kit import build_kit, upload_kit
from oceanscope_ml.live.run import (
    DEFAULT_LIVE_CONFIG,
    LiveSettings,
    RunLog,
    default_fetch,
    load_live_config,
    resolve_kit,
    run_live,
    utc_today,
    write_step_summary,
)
from oceanscope_ml.live.store import HfRepo, LocalRepo

logger = logging.getLogger("oceanscope_ml.live")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OceanScope live mode")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--live-config", default=DEFAULT_LIVE_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    kit = sub.add_parser("kit", help="Pack (and upload) the live kit for a trained run")
    kit.add_argument("--checkpoint", required=True)
    kit.add_argument("--stats", required=True, help="Stats store (clim + norm_stats)")
    kit.add_argument("--static-store", required=True, help="Any part store with static fields")
    kit.add_argument("--out", required=True)
    kit.add_argument("--upload", action="store_true", help="Upload to live.kit_repo")

    run = sub.add_parser("run", help="Publish the newest complete NRT day")
    run.add_argument("--kit-dir", default=None, help="Default: download live.kit_repo")
    run.add_argument("--repo", default=None, help="Default: publish.hf_dataset_repo")
    run.add_argument("--local-repo", default=None, help="A folder instead of the HF repo")
    run.add_argument("--work-dir", default=None, help="Scratch folder (default: a temp dir)")
    run.add_argument("--today", default=None, help="YYYY-MM-DD; default: today in UTC")
    run.add_argument("--dry-run", action="store_true", help="Do everything but upload")
    run.add_argument("--log", default="outputs/live/live_run.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args(argv)
    config = load_config(args.config)

    if args.command == "kit":
        kit = build_kit(
            config, Path(args.checkpoint), Path(args.stats), Path(args.static_store), Path(args.out)
        )
        if args.upload:
            repo = LiveSettings.from_dict(load_live_config(args.live_config)).kit_repo
            if not repo:
                raise SystemExit("live.kit_repo is empty in the config")
            upload_kit(kit, repo)
        return 0

    settings = LiveSettings.from_dict(load_live_config(args.live_config))
    today = pd.Timestamp(args.today) if args.today else utc_today()
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="oceanscope-"))
    if args.local_repo:
        repo_obj: LocalRepo | HfRepo = LocalRepo(Path(args.local_repo))
    else:
        repo_id = args.repo or str(config.publish.get("hf_dataset_repo", ""))
        if not repo_id:
            raise SystemExit("No dataset repo: set publish.hf_dataset_repo or pass --repo")
        repo_obj = HfRepo(repo_id)
    log_path = Path(args.log)
    log = RunLog(today=str(today.date()))
    try:
        kit_obj = resolve_kit(config, settings, Path(args.kit_dir) if args.kit_dir else None)
        run_live(
            config,
            repo_obj,
            kit_obj,
            work,
            today,
            default_fetch(config, work / "scratch"),
            settings,
            dry_run=args.dry_run,
            log=log,
        )
    except Exception as e:
        log.status = "failed"
        log.notes.append(f"{type(e).__name__}: {e}")
        log.write(log_path)
        write_step_summary(log)
        logger.exception("Live run failed")
        return 1
    log.write(log_path)
    write_step_summary(log)
    logger.info("Live run %s: %s (target %s)", log.today, log.status, log.target_day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
