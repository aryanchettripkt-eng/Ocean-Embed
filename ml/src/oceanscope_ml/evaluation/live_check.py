"""P9-1: NRT vs reprocessed inputs on one past month (PRD_AI_ML §12).

The published checkpoint is run twice on the same month: once with the near-real-time inputs
live mode will use (``live.sources``) and once with the reprocessed inputs the training store
used (``live_check.reprocessed``, both in ``configs/live.yaml``). Both are scored against
GLORYS on the same cells and days.
The owner reads the report and decides whether live mode is enabled; this script does not
set a pass mark.

Usage (from ml/, with the Copernicus, Earthdata and CDS credentials in the environment):

    python -m oceanscope_ml.evaluation.live_check --kit-dir kit/ --month 2020-06 \
        [--glorys-dataset cmems_mod_glo_phy_myint_0.083deg_P1D-m] [--out-dir outputs]

Writes ``<out>/reports/live_mode_check.md`` and ``<out>/metrics/live_mode_check.csv``.
The month may not overlap the sealed test period (PRD §0 rule 3).
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.transform import TestLeakageError
from oceanscope_ml.live import inputs as live_inputs
from oceanscope_ml.live.kit import Kit, open_kit, predict_days
from oceanscope_ml.live.run import DEFAULT_LIVE_CONFIG, default_fetch, load_live_config

logger = logging.getLogger("oceanscope_ml.evaluation.live_check")

SETS = ("nrt", "reprocessed")
GlorysFetcher = Callable[[pd.DatetimeIndex], np.ndarray]


def month_days(month: str, config: ExperimentConfig) -> pd.DatetimeIndex:
    """Days of ``YYYY-MM``; refuses a month that touches the sealed test period."""
    start = pd.Timestamp(f"{month}-01")
    days = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")
    test_start, test_end = (pd.Timestamp(d) for d in config.splits.test)
    if days[-1] >= test_start and days[0] <= test_end:
        raise TestLeakageError(f"{month} overlaps the sealed test period (PRD_AI_ML §0 rule 3)")
    train_start, train_end = (pd.Timestamp(d) for d in config.splits.train)
    if days[-1] >= train_start and days[0] <= train_end:
        logger.warning("%s is in the training period; errors there are optimistic", month)
    return days


def depth_table(
    config: ExperimentConfig,
    preds: dict[str, xr.Dataset],
    glorys: np.ndarray,
    days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Per-depth RMSE vs GLORYS for each input set, on cells valid in both sets."""
    index = pd.DatetimeIndex(days)
    common = sorted(set(preds["nrt"]["time"].values) & set(preds["reprocessed"]["time"].values))
    if not common:
        raise ValueError("No day has complete inputs in both the NRT and reprocessed sets")
    sel = {k: v.sel(time=common) for k, v in preds.items()}
    ref = glorys[index.get_indexer(pd.DatetimeIndex(common))]
    valid = (sel["nrt"]["coverage"].values == 1) & (sel["reprocessed"]["coverage"].values == 1)
    rows: list[dict[str, Any]] = []
    for d, depth in enumerate(config.depths_m):
        nrt = sel["nrt"]["temp_pred"].values[:, d].astype(np.float64)
        rep = sel["reprocessed"]["temp_pred"].values[:, d].astype(np.float64)
        r = ref[:, d].astype(np.float64)
        m = valid & np.isfinite(r) & np.isfinite(nrt) & np.isfinite(rep)
        n = int(m.sum())
        rmse_nrt = float(np.sqrt(np.mean((nrt[m] - r[m]) ** 2))) if n else np.nan
        rmse_rep = float(np.sqrt(np.mean((rep[m] - r[m]) ** 2))) if n else np.nan
        rows.append(
            {
                "depth_m": depth,
                "n": n,
                "rmse_nrt_c": rmse_nrt,
                "rmse_reprocessed_c": rmse_rep,
                "rmse_diff_c": rmse_nrt - rmse_rep,
                "rmse_nrt_vs_reprocessed_c": float(np.sqrt(np.mean((nrt[m] - rep[m]) ** 2)))
                if n
                else np.nan,
                "bias_nrt_c": float(np.mean(nrt[m] - r[m])) if n else np.nan,
                "bias_reprocessed_c": float(np.mean(rep[m] - r[m])) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    month: str,
    kit: Kit,
    table: pd.DataFrame,
    used: dict[str, dict[str, Any]],
    complete: dict[str, list[str]],
    n_days: int,
    glorys_id: str,
) -> Path:
    def f(x: float) -> str:
        return "n/a" if not np.isfinite(x) else f"{x:.3f}"

    lines = [
        f"# Live mode check: NRT vs reprocessed inputs, {month}",
        "",
        f"Run `{kit.run_id}`. Created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
        "`oceanscope_ml.evaluation.live_check` (PRD_AI_ML §12, PHASES P9-1).",
        f"Reference: GLORYS `{glorys_id}`. Scored on days complete in both sets and cells "
        "with a valid prediction in both.",
        "",
        f"Days with complete inputs: NRT {len(complete['nrt'])}/{n_days}, "
        f"reprocessed {len(complete['reprocessed'])}/{n_days}.",
        "",
        "| Depth (m) | n | RMSE NRT (°C) | RMSE reprocessed (°C) | Difference (°C) | "
        "RMSE NRT vs reprocessed (°C) | Bias NRT | Bias reprocessed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['depth_m']:g} | {int(r['n'])} | {f(r['rmse_nrt_c'])} | "
            f"{f(r['rmse_reprocessed_c'])} | {f(r['rmse_diff_c'])} | "
            f"{f(r['rmse_nrt_vs_reprocessed_c'])} | {f(r['bias_nrt_c'])} | "
            f"{f(r['bias_reprocessed_c'])} |"
        )
    lines += ["", "Inputs used:", "", "| Product | NRT | Reprocessed |", "|---|---|---|"]
    for product in used["nrt"]:
        lines.append(
            f"| {product} | `{used['nrt'][product]['dataset_id']}` | "
            f"`{used['reprocessed'].get(product, {}).get('dataset_id', '-')}` |"
        )
    lines += [
        "",
        "Difference = RMSE NRT − RMSE reprocessed; positive means the NRT inputs are worse.",
        "Decision (owner): accept or reject enabling live mode. Not decided by this script.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def default_glorys(config: ExperimentConfig, spec: dict[str, Any]) -> GlorysFetcher:
    def fetch(days: pd.DatetimeIndex) -> np.ndarray:
        from oceanscope_ml.data.build_store import _load_with_retries, stream_copernicus
        from oceanscope_ml.data.reduce import reduce_glorys12

        r = config.region
        raw = stream_copernicus(
            dataset_id=spec["dataset_id"],
            bbox=(r.lon_min, r.lon_max, r.lat_min, r.lat_max),
            dates=(str(days[0].date()), str(days[-1].date())),
            variables=list(spec.get("variables", ["thetao"])),
            depths=(float(min(config.depths_m)), float(max(config.depths_m))),
            depth_levels=list(config.depths_m),
        )
        raw = _load_with_retries(raw, spec["dataset_id"])
        grid = live_inputs.target_grid(config)
        thetao = reduce_glorys12(raw, grid, list(config.depths_m))["thetao"]
        thetao = thetao.assign_coords(time=pd.to_datetime(thetao["time"].values).normalize())
        thetao = thetao.reindex(time=days)
        return np.asarray(thetao.values, dtype=np.float32)

    return fetch


def run_check(
    config: ExperimentConfig,
    kit: Kit,
    month: str,
    fetch: live_inputs.Fetcher,
    glorys_fetch: GlorysFetcher,
    out_dir: Path,
    work: Path,
    glorys_id: str,
    live_cfg: dict[str, Any],
) -> dict[str, Path]:
    days = month_days(month, config)
    raw = live_cfg
    sources = {
        "nrt": dict((raw.get("live") or {}).get("sources") or {}),
        "reprocessed": dict((raw.get("live_check") or {}).get("reprocessed") or {}),
    }
    for name, src in sources.items():
        if not src:
            raise ValueError(f"No {name} sources in the config")
    lookbacks = live_inputs.var_lookbacks(config)
    max_lb = max(lookbacks.values())
    fetch_days = pd.date_range(days[0] - pd.Timedelta(days=max_lb), days[-1], freq="D")
    static = kit.static()
    ocean = static["ocean_mask"].values.astype(bool)

    preds: dict[str, xr.Dataset] = {}
    complete: dict[str, list[str]] = {}
    for name in SETS:
        inputs = live_inputs.load_inputs(config, sources[name], fetch_days, ocean, fetch)
        cov = live_inputs.daily_coverage(inputs, ocean)
        ok = [d for d in days if live_inputs.check_inputs(cov, lookbacks, d, d).target is not None]
        complete[name] = [str(d.date()) for d in ok]
        logger.info("%s: %d of %d days have complete inputs", name, len(ok), len(days))
        if not ok:
            raise ValueError(f"No day of {month} has complete {name} inputs")
        window = live_inputs.write_window_store(
            work / f"{name}.zarr", inputs, static, fetch_days[0], days[-1]
        )
        preds[name] = predict_days(config, kit, window, ok)

    glorys = glorys_fetch(days)
    table = depth_table(config, preds, glorys, days)
    csv = out_dir / "metrics" / "live_mode_check.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.insert(0, "month", month)
    table.insert(1, "run_id", kit.run_id)
    table.to_csv(csv, index=False)
    report = write_report(
        out_dir / "reports" / "live_mode_check.md",
        month,
        kit,
        table,
        sources,
        complete,
        len(days),
        glorys_id,
    )
    return {"csv": csv, "report": report}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="P9-1 NRT vs reprocessed check")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--live-config", default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--kit-dir", required=True)
    parser.add_argument("--month", required=True, help="YYYY-MM, outside the test period")
    parser.add_argument("--glorys-dataset", default=None, help="Override live_check.glorys")
    parser.add_argument("--out-dir", default=None, help="Default: paths.outputs")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    kit = open_kit(Path(args.kit_dir), config)
    live_cfg = load_live_config(args.live_config)
    spec = dict((live_cfg.get("live_check") or {}).get("glorys") or {})
    if args.glorys_dataset:
        spec["dataset_id"] = args.glorys_dataset
    if not spec.get("dataset_id"):
        raise SystemExit("No GLORYS dataset: set live_check.glorys or pass --glorys-dataset")
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="livecheck-"))
    paths = run_check(
        config,
        kit,
        args.month,
        default_fetch(config, work / "scratch"),
        default_glorys(config, spec),
        Path(args.out_dir or config.paths.outputs),
        work,
        spec["dataset_id"],
        live_cfg,
    )
    print(f"[OK] {paths['report']}\n[OK] {paths['csv']}")


if __name__ == "__main__":
    main()
