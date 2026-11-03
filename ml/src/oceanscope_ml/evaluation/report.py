"""Evaluation report ``reports/evaluation_<run_id>.md`` written from the metric files (P5-3, M3).

Usage (from ml/):

    python -m oceanscope_ml.evaluation.report --config configs/experiment.yaml \
        --run-id <run_id> [--out-dir outputs]

Every number in the report is read from a file under ``<out>/metrics/`` that code produced
(PRD_AI_ML §0 rule 2); nothing is typed in. A file that does not exist yet is listed under
"Not produced yet" instead of being skipped silently. The leakage caveats (PRD_AI_ML §3, §7;
PRD_BACKEND §7 wording) are always written, whatever files exist.

Test-year numbers appear only if ``final_test`` has written ``<run_id>_test.csv``; the report
never reads test data itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.evaluation.argo_validation import (
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    headline_verdict,
)
from oceanscope_ml.evaluation.temporal import (
    CLIMATOLOGY,
    FLICKER_FLAG_MIN_DEPTHS,
    FLICKER_RATIO_FLAG,
    HEAT_RMS_RATIO_FLAG,
    INVERSION_FLAG_FRAC,
)

logger = logging.getLogger("oceanscope_ml.evaluation.report")

# Same words as the backend's fixed strings (PRD_BACKEND §7, backend/app/schemas/caveats.py),
# so the report and the website say the same thing.
GLORYS_CAVEAT = (
    "Predictions are trained to reproduce the GLORYS reanalysis. GLORYS assimilates satellite "
    "altimetry and ARGO profiles, so agreement with GLORYS is not the same as agreement with "
    "the real ocean."
)
ARGO_CAVEAT = (
    "ARGO profiles are also assimilated into GLORYS, so this comparison is not fully independent."
)
SYNTHETIC_BANNER = (
    "**SYNTHETIC: these numbers come from synthetic test data, not from a model run. "
    "Do not quote them as results.**"
)
M2_GATE_DEPTHS = 10  # PRD_AI_ML §9 M2: beat climatology at >= 10 of 15 depths
BASELINES = ("climatology", "ridge")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or isinstance(value, bool):
        return "n/a" if value is None else str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        x = float(value)
        if math.isnan(x):
            return "n/a"
        if digits == 0:
            return str(int(x))
        # Large values (heat content in J m-2) read better in scientific notation.
        return f"{x:.3e}" if abs(x) >= 1e5 else f"{x:.{digits}f}"
    text = str(value)
    return "n/a" if text.lower() == "nan" else text


def md_table(df: pd.DataFrame, digits: dict[str, int] | None = None) -> str:
    """Markdown table; floats to 3 decimals unless ``digits`` says otherwise."""
    digits = digits or {}
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], digits.get(c, 3)) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _depth_key(value: Any) -> str:
    """Depth label as written in the CSVs ('0.494', 'all'), so files can be joined."""
    text = str(value)
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


class _Report:
    def __init__(self, config: ExperimentConfig, run_id: str, out: Path) -> None:
        self.config = config
        self.run_id = run_id
        self.out = out
        self.metrics = out / "metrics"
        self.lines: list[str] = []
        self.missing: list[str] = []
        manifest_path = out / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )

    def file(self, name: str) -> pd.DataFrame | None:
        df = _read(self.metrics / name)
        if df is None:
            self.missing.append(f"metrics/{name}")
        return df

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def synthetic(self, frames: list[pd.DataFrame | None]) -> bool:
        if self.manifest.get("synthetic") is True:
            return True
        for df in frames:
            if df is not None and "model" in df.columns:
                if df["model"].astype(str).str.contains("synthetic", case=False).any():
                    return True
        return False


def _runs_row(metrics: Path, run_id: str) -> dict[str, Any]:
    runs = _read(metrics / "runs.csv")
    if runs is None:
        return {}
    rows = runs[runs["run_id"].astype(str) == run_id]
    return {} if rows.empty else dict(rows.iloc[-1])


def _caveats(r: _Report, test_scored: bool) -> None:
    train = r.config.splits.train
    val = r.config.splits.val
    test = r.config.splits.test
    r.add(
        "## Caveats (read first)",
        "",
        f"- **GLORYS is not the real ocean.** {GLORYS_CAVEAT}",
        f"- **ARGO is not fully independent.** {ARGO_CAVEAT} ARGO metrics use only the "
        "held-out floats in `argo_eval_floats.json`, which are never model inputs or "
        "fine-tuning targets (PRD_AI_ML §7).",
        f"- **Validation scores are optimistic.** Validation dates ({val[0]} to {val[1]}) were "
        "used for early stopping and model selection, so they are not an untouched test.",
    )
    if test_scored:
        r.add(
            f"- **Test year.** The sealed test year ({test[0]} to {test[1]}) was scored once by "
            "`final_test`; its numbers are labelled below and nothing was tuned on them."
        )
    else:
        r.add(
            f"- **Test year not evaluated.** The test year ({test[0]} to {test[1]}) is sealed "
            "(PRD_AI_ML §0 rule 3). Every number here is from validation dates."
        )
    r.add(
        f"- **Training-only statistics.** The climatology and normalization statistics come from "
        f"the training period ({train[0]} to {train[1]}) only (PRD_AI_ML §0 rule 4).",
        "- **Competitors without data are n/a.** A product with no data for a date (for example "
        "GREP or ARMOR3D outside their coverage) is reported as n/a, never extrapolated.",
        "",
    )


def _headline(r: _Report, test_scored: bool) -> None:
    r.add("## Primary criterion (PRD_AI_ML §1.1, pre-registered)", "")
    r.add(
        f"Bay of Bengal (lon ≥ 78°E, lat ≥ 5°N), GLORYS levels from {HEADLINE_DEPTH_MIN_M:g} to "
        f"{HEADLINE_DEPTH_MAX_M:g} m: is the model's RMSE against held-out ARGO floats lower "
        'than GLORYS12, the GREP ensemble mean and ARMOR3D at the same matchups? "Better" '
        "means the upper bound of the 95% float-bootstrap interval of (model RMSE − competitor "
        "RMSE) is below 0. The test-year result is the headline; the validation result is a "
        "development check.",
        "",
    )
    head = r.file(f"{r.run_id}_headline.csv")
    if head is None:
        r.add("Not produced yet.", "")
        return
    for period, label in (("val", "Validation dates"), ("test", "Test year (sealed)")):
        rows = head[head["period"] == period]
        if rows.empty or (rows["n_obs"].fillna(0) == 0).all():
            note = "not evaluated" if period == "test" and not test_scored else "no matchups"
            r.add(f"**{label}:** {note}.", "")
            continue
        verdict = headline_verdict(head, period)
        text = {True: "Met", False: "Not met", None: "Not decided (a competitor has no data)"}
        r.add(f"**{label}: {text[verdict]}.**", "")
        cols = ["product", "n_floats", "n_obs", "rmse_c", "ci_low_c", "ci_high_c"]
        cols += ["diff_vs_model_c", "diff_ci_low_c", "diff_ci_high_c"]
        r.add(md_table(rows[cols], {"n_floats": 0, "n_obs": 0}), "")


def _validation(r: _Report) -> pd.DataFrame | None:
    r.add("## Validation vs GLORYS12 (secondary)", "")
    val = r.file(f"{r.run_id}_val.csv")
    if val is None:
        r.add("Not produced yet.", "")
        return None
    table = val[["depth_m", "n", "rmse_c", "mae_c", "bias_c", "skill_vs_clim_pct"]].copy()
    table["depth_m"] = table["depth_m"].map(_depth_key)
    for name in BASELINES:
        base = r.file(f"baseline_{name}_val.csv")
        if base is not None:
            b = base[["depth_m", "rmse_c"]].copy()
            b["depth_m"] = b["depth_m"].map(_depth_key)
            table = table.merge(b.rename(columns={"rmse_c": f"{name}_rmse_c"}), "left", "depth_m")
    r.add(
        "Errors are model minus GLORYS on cells with a valid prediction (coverage 1) and a valid "
        "GLORYS target. `skill_vs_clim_pct` = 100 × (1 − RMSE_model / RMSE_climatology) on "
        "exactly those cells. Baseline RMSE columns come from the baseline files, which score "
        "every valid ocean target cell, so they are close to but not on the identical cells.",
        "",
        md_table(table, {"n": 0, "skill_vs_clim_pct": 1}),
        "",
    )
    per_depth = val[val["depth_m"].astype(str) != "all"]
    beats = int((per_depth["skill_vs_clim_pct"] > 0).sum())
    total = len(per_depth)
    passed = beats >= M2_GATE_DEPTHS
    r.add(
        f"**M2 gate (PRD_AI_ML §9):** the model beats climatology (skill > 0) at {beats} of "
        f"{total} depths; the gate needs {M2_GATE_DEPTHS}. "
        + ("Passed." if passed else "Not passed: report and stop, do not tune on test."),
        "",
    )
    return val


def _pivot_section(r: _Report, title: str, name: str, column: str) -> None:
    r.add(f"### {title}", "")
    df = r.file(name)
    if df is None:
        r.add("Not produced yet.", "")
        return
    df = df.copy()
    df["depth_m"] = df["depth_m"].map(_depth_key)
    order = list(dict.fromkeys(df["depth_m"]))
    wide = df.pivot_table(
        index="depth_m", columns=column, values="rmse_c", sort=False, dropna=False
    )
    wide = wide.reindex(order).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    r.add("RMSE (°C) vs GLORYS12.", "", md_table(wide), "")


def _ablation(r: _Report) -> None:
    r.add("## Ablations (PRD_AI_ML §5.7)", "")
    df = r.file(f"ablation_{r.run_id}.csv")
    if df is None:
        r.add("Not produced yet.", "")
        return
    df = df.copy()
    df["depth_m"] = df["depth_m"].map(_depth_key)
    order = list(dict.fromkeys(df["depth_m"]))
    wide = df.pivot_table(
        index="depth_m", columns="removed_group", values="delta_rmse_c", dropna=False
    )
    wide = wide.reindex(order).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    r.add(
        "Change in validation RMSE (°C) when each input group is removed; positive means the "
        "group helps.",
        "",
        md_table(wide),
        "",
    )


def _truthy(series: pd.Series) -> pd.Series:
    """The ``flagged``/``flag`` columns read back from CSV as bool, or as text with blanks."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def _temporal(r: _Report, model: str | None) -> None:
    """§5.8 items 1 and 2, read from the files ``evaluation.temporal`` writes."""
    r.add("## Temporal consistency (PRD_AI_ML §5.8)", "", "### Day-to-day flicker", "")
    df = r.file(f"{r.run_id}_val_temporal.csv")
    if df is None:
        r.add("Not produced yet.", "")
    else:
        model = model or str(df["model"].iloc[0])
        table = df.copy()
        table["depth_m"] = table["depth_m"].map(_depth_key)
        table["flagged"] = _truthy(table["flagged"])
        rows = table[table["model"] == model].copy()
        per_depth = rows[rows["depth_m"] != "all"]
        flagged = [str(d) for d in per_depth.loc[per_depth["flagged"], "depth_m"]]
        rows["flagged"] = rows["flagged"].map(lambda f: "jumps" if f else "")
        cols = ["depth_m", "n", "tend_rmse_pred_c", "tend_rmse_glorys_c", "tend_err_rmse_c"]
        cols += ["lag1_autocorr_pred", "lag1_autocorr_glorys", "flagged"]
        r.add(
            f"RMS of the day-to-day change for `{model}` and GLORYS on the same cells, from "
            f"consecutive valid days. A depth is flagged when the model's change is more than "
            f"{FLICKER_RATIO_FLAG:g} × GLORYS's.",
            "",
            md_table(rows[cols], {"n": 0}),
            "",
        )
        if flagged:
            r.add(f"**Flagged at {len(flagged)} depth(s):** {', '.join(flagged)} m.")
            if len(flagged) >= FLICKER_FLAG_MIN_DEPTHS:
                r.add(
                    f"This is {FLICKER_FLAG_MIN_DEPTHS} or more depths, the trigger for stretch "
                    "item §10.2 (temporal model)."
                )
        else:
            r.add("**No depth flagged.**")
        others = table[(table["model"] != model) & (table["depth_m"] == "all")]
        if not others.empty:
            r.add(
                "",
                "Other models, all depths together:",
                "",
                md_table(
                    others[
                        ["model", "n", "tend_rmse_pred_c", "tend_rmse_glorys_c", "tend_err_rmse_c"]
                    ],  # fmt: skip
                    {"n": 0},
                ),
            )
        r.add("")

    r.add("### Per-year bias (warming check)", "")
    bias = r.file(f"{r.run_id}_bias_by_year.csv")
    if bias is not None:
        model = model or str(bias["model"].iloc[0])
        b = bias.copy()
        b["depth_m"] = b["depth_m"].map(_depth_key)
        mine = b[b["model"] == model]
        order = list(dict.fromkeys(mine["depth_m"]))
        mine = mine.assign(year_split=mine["year"].astype(str) + " (" + mine["split"] + ")")
        wide = mine.pivot_table(
            index="depth_m", columns="year_split", values="bias_c", sort=False, dropna=False
        )
        wide = wide.reindex(order).reset_index()
        wide.columns = [str(c) for c in wide.columns]
        r.add(
            f"Mean (`{model}` − GLORYS) per year, °C. The target is an anomaly from a fixed "
            "training-period climatology, so a model that damps a warming trend shows a bias "
            "that falls year by year. Train years, when present, come from a held-in sample.",
            "",
            md_table(wide),
            "",
        )
    else:
        r.add("Not produced yet.", "")

    trend = r.file(f"{r.run_id}_bias_trend.csv")
    if trend is None:
        return
    t = trend.copy()
    t["depth_m"] = t["depth_m"].map(_depth_key)
    order = list(dict.fromkeys(t["depth_m"]))
    # One row per model and depth, so pivot (not pivot_table) keeps NaN slopes visible.
    wide_t = t.pivot(index="depth_m", columns="model", values="slope_c_per_yr")
    wide_t = wide_t.reindex(order).reset_index()
    wide_t.columns = [str(c) if c == "depth_m" else f"{c} slope (°C/yr)" for c in wide_t.columns]
    n_years = int(t["n_years"].max()) if not t.empty else 0
    r.add(
        f"Least-squares slope of the yearly bias, up to {n_years} year(s); each year counts once. "
        f"The `{CLIMATOLOGY}` slope shows how far the ocean itself drifted from the fixed "
        "climatology over these years; a model slope well below it means the model damps the "
        "trend.",
        "",
        md_table(wide_t, {}),
        "",
    )


def _physics(r: _Report) -> None:
    """§5.8 item 3 from ``<run_id>_physics.csv``; the flags are the ones the checks wrote."""
    r.add("### Physics sanity checks", "")
    df = r.file(f"{r.run_id}_physics.csv")
    if df is None:
        r.add("Not produced yet.", "")
        return
    df = df.copy()
    df["flag"] = _truthy(df["flag"])
    r.add(
        "(a) Monthly change in 0–300 m heat content vs ERA5 net heat flux, for the model and for "
        "GLORYS (advection makes both differ; the model should not be much worse than GLORYS). "
        "(b) Share of predicted profiles with an inversion > 0.5 °C where GLORYS has none. "
        "(c) Share of cells more than 3 °C from GLORYS.",
        "",
        md_table(df.assign(flag=df["flag"].map(lambda f: "FLAG" if f else "")), {"n": 0}),
        "",
    )
    flagged = df[df["flag"]]
    if flagged.empty:
        r.add(
            f"**No physics flag** (heat-content RMS > {HEAT_RMS_RATIO_FLAG:g} × GLORYS, or "
            f"inversions not in GLORYS > {100 * INVERSION_FLAG_FRAC:g}% of profiles).",
            "",
        )
    else:
        items = [f"{row.check} / {row.region} / {row.metric}" for row in flagged.itertuples()]
        r.add(f"**Flagged:** {'; '.join(items)}.", "")


def _figure(r: _Report) -> None:
    r.add("### 90-day time series", "")
    fig = r.out / "figures" / f"timeseries_{r.run_id}.png"
    if fig.exists():
        r.add(f"![Model vs GLORYS at three points](../figures/{fig.name})", "")
    else:
        r.missing.append(f"figures/{fig.name}")
        r.add("Not produced yet.", "")


def _argo(r: _Report) -> None:
    r.add("## ARGO held-out floats (PRD_AI_ML §7)", "", ARGO_CAVEAT, "")
    df = r.file(f"argo_{r.run_id}.csv")
    if df is None:
        r.add("Not produced yet.", "")
        return
    cols = ["depth_m", "n_profiles", "n_obs", "rmse_c", "glorys_rmse_c", "grep_rmse_c"]
    cols += ["armor3d_rmse_c", "bias_c"]
    table = df[[c for c in cols if c in df.columns]].copy()
    table["depth_m"] = table["depth_m"].map(_depth_key)
    r.add(
        "RMSE (°C) against held-out floats at the same points; n/a where a product has no data.",
        "",
        md_table(table, {"n_profiles": 0, "n_obs": 0}),
        "",
    )


def _write_test(r: _Report) -> None:
    r.add("## Final test, sealed year (scored once)", "")
    df = _read(r.metrics / f"{r.run_id}_test.csv")
    if df is None:
        r.add("Not evaluated: the test year is sealed until Aryan says run the final test.", "")
        return
    flag = r.manifest.get("test_evaluated")
    r.add(
        "Scored once by `final_test` on the owner's go. Nothing was tuned on these numbers. "
        f"Manifest `test_evaluated`: {flag}.",
        "",
    )
    table = df[["depth_m", "n", "rmse_c", "mae_c", "bias_c", "skill_vs_clim_pct"]].copy()
    table["depth_m"] = table["depth_m"].map(_depth_key)
    r.add(md_table(table, {"n": 0, "skill_vs_clim_pct": 1}), "")


def generate_report(config: ExperimentConfig, run_id: str, out_dir: Path | None = None) -> Path:
    """Write ``<out>/reports/evaluation_<run_id>.md`` and return its path."""
    out = Path(out_dir or config.paths.outputs)
    r = _Report(config, run_id, out)
    test_scored = (r.metrics / f"{run_id}_test.csv").exists()
    run = _runs_row(r.metrics, run_id)

    _caveats(r, test_scored)
    _headline(r, test_scored)
    val = _validation(r)
    model = str(val["model"].iloc[0]) if val is not None and len(val) else None
    _pivot_section(r, "By season", f"{run_id}_val_by_season.csv", "season")
    _pivot_section(r, "By region", f"{run_id}_val_by_region.csv", "region")
    _ablation(r)
    _temporal(r, model)
    _physics(r)
    _figure(r)
    _argo(r)
    _write_test(r)

    frames = [_read(r.metrics / f"{run_id}_val.csv"), _read(r.metrics / f"{run_id}_headline.csv")]
    header = [f"# Evaluation report: {run_id}", ""]
    if r.synthetic(frames):
        header += [SYNTHETIC_BANNER, ""]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header += [
        f"- Generated: {generated} by `oceanscope_ml.evaluation.report` from the files in "
        "`metrics/`. No number in this report was typed in.",
        f"- Model: {model or 'n/a'}; dataset version: "
        f"{run.get('dataset_version', config.dataset_version)}",
        f"- Git commit: {run.get('git_commit', 'n/a')}; config hash: "
        f"{run.get('config_hash', 'n/a')}",
        f"- Training: {_fmt(run.get('epochs'), 0)} epochs, best epoch "
        f"{_fmt(run.get('best_epoch'), 0)}, {_fmt(run.get('wall_minutes'), 1)} min on "
        f"{run.get('gpu_name', 'n/a')}",
        "",
    ]
    footer = ["## Not produced yet", ""]
    footer += [f"- `{m}`" for m in r.missing] if r.missing else ["Nothing missing."]

    dest = out / "reports" / f"evaluation_{run_id}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join([*header, *r.lines, *footer]) + "\n", encoding="utf-8")
    logger.info("Report written: %s (%d file(s) not produced yet)", dest, len(r.missing))
    return dest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the evaluation report for one run.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default=None, help="Outputs dir (default paths.outputs)")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    config = load_config(args.config)
    generate_report(config, args.run_id, Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
