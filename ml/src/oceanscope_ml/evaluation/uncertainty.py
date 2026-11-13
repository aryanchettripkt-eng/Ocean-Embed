"""Honest error bars? A Bayesian last layer on the frozen U-Net, fitted with Batch-and-Match.

Usage (from ml/):

    # 1) Matches + head features at the held-out floats (one predict pass per split)
    python -m oceanscope_ml.evaluation.uncertainty collect --config configs/experiment.yaml \
        --checkpoint <best_ft.pt> --store <parts...> --stats <stats.zarr> \
        --argo <argo_YYYY-MM.nc ...> --holdout configs/argo_eval_floats.json --split val
    # 2) Fit on the validation years, check coverage
    python -m oceanscope_ml.evaluation.uncertainty analyse --fit <uncert_val.parquet> \
        [--check <uncert_test.parquet>] --out-dir outputs

The model's mean is the trained U-Net (frozen). Per depth level ``d`` the error of that mean at
an ARGO observation, ``r = obs - pred`` (°C), gets a heteroscedastic Bayesian linear model on
the 32 features ``x`` that feed the U-Net's 1x1 head (standardized, plus a constant):

    r / s_d ~ Normal(beta_d . x, exp(gamma_d . x)^2),   beta_d ~ N(0, 0.5^2), gamma_d ~ N(0, 0.5^2)

``s_d`` is the RMS error at that level on the fitting floats. ``beta`` is a small mean
correction in the head's own feature space and ``gamma`` sets an input-dependent error size. The
posterior over the 66 parameters per level is not Gaussian; it is fitted with Batch-and-Match
(Cai, Modi et al., ICML 2024; ``bam_fit``), a score-based VI method, started from the Laplace
approximation at the MAP. The features are whitened first (they are strongly correlated), and
the fit uses one observation per profile and level (``thin``), so a profile counts once per
level however finely the float samples it. Floats are still correlated in time, so the parameter
posterior is somewhat too narrow; the noise term dominates the error bars anyway.

Error bars compared (``METHODS``):

- ``const``: Normal(pred, s_d^2), one fixed error size per level (the cheap baseline);
- ``bll_plugin``: the last layer at the posterior mean only (input-dependent, no parameter
  uncertainty);
- ``bll_bam``: the full BaM posterior predictive (a mixture over 200 parameter draws).

Each gives a PIT value per observation (the predictive CDF at the observation). The central
``p`` interval holds the observation when ``|PIT - 0.5| <= p / 2``, so coverage at 68% and 95%
and the reliability curve need no quantile search. Coverage is reported per level, per depth
band (0-50, 50-300, 300-1100 m), for the Bay of Bengal 50-300 m band and overall, with 95%
float-bootstrap intervals.

What is fitted on what: only held-out floats are used (they are never model inputs). ``fit``
is the validation years (2018-2020). The ``val_halves`` check fits on the dev half of those
floats and scores the final half. The 2021 check (``--check`` with the test-year table) scores
a second look at the sealed test year, on Aryan's request (DECISIONS 2026-11-13); nothing is
fitted on 2021.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtr

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.parts import open_parts
from oceanscope_ml.evaluation.argo_validation import (
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    _in_bay_of_bengal,
    _load_holdout,
    _nearest_cell,
    holdout_halves,
    load_profiles,
    match_profiles,
)
from oceanscope_ml.inference.predict import predict_store

logger = logging.getLogger("oceanscope_ml.evaluation.uncertainty")

METHODS = ("const", "bll_plugin", "bll_bam")
NOMINAL = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95, 0.99)
BANDS = {"0-50 m": (0.0, 50.0), "50-300 m": (50.0, 300.0), "300-1100 m": (300.0, 1100.0)}
PRIOR_SD = 0.5
LOG_SD_CLIP = 8.0
POSTERIOR_DRAWS = 200
BOOTSTRAP_DRAWS = 200
SEED = 13
KEY = ["platform_number", "cycle"]


# --- Batch-and-Match ---------------------------------------------------------------------------


def _sym_sqrt(m: np.ndarray) -> np.ndarray:
    w, q = np.linalg.eigh((m + m.T) / 2)
    return np.asarray((q * np.sqrt(np.clip(w, 0, None))) @ q.T)


def bam_update(
    x: np.ndarray, g: np.ndarray, mu0: np.ndarray, s0: np.ndarray, reg: float
) -> tuple[np.ndarray, np.ndarray]:
    """One Batch-and-Match step (Cai et al. 2024, eq. 11-12; gsmvi.bam.bam_update).

    ``x`` (B, D) are draws from N(mu0, s0), ``g`` the target's scores at them, ``reg`` the
    step size lambda. Uses the symmetric form
    S = 2 V^1/2 (I + (I + 4 V^1/2 U V^1/2)^1/2)^-1 V^1/2.
    """
    xbar, gbar = x.mean(0), g.mean(0)
    c = (x - xbar).T @ (x - xbar) / len(x)
    gg = (g - gbar).T @ (g - gbar) / len(g)
    u = reg * gg + reg / (1 + reg) * np.outer(gbar, gbar)
    v = s0 + reg * c + reg / (1 + reg) * np.outer(mu0 - xbar, mu0 - xbar)
    vh = _sym_sqrt(v)
    eye = np.eye(len(mu0))
    s = 2 * vh @ np.linalg.solve(eye + _sym_sqrt(eye + 4 * vh @ u @ vh), vh)
    s = (s + s.T) / 2
    mu = mu0 / (1 + reg) + reg / (1 + reg) * (s @ gbar + xbar)
    return mu, s


def bam_fit(
    score: Any,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    niter: int = 300,
    batch: int = 64,
    reg0: float = 1e4,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit N(mu, cov) to the target with score ``score((B, D)) -> (B, D)``.
    Step sizes lambda_t = reg0 / (1 + t), the paper's schedule. The posterior here is very
    narrow (thousands of profiles, 66 parameters), so reg0 is large: with the examples' 100 the
    steps shrink before the mean gets there (``fit_heads`` logs the distance to the MAP)."""
    rng = np.random.default_rng(seed)
    d = len(mu)
    for t in range(niter):
        chol = np.linalg.cholesky(cov + 1e-10 * np.eye(d))
        x = mu + rng.standard_normal((batch, d)) @ chol.T
        mu_new, cov_new = bam_update(x, score(x), mu, cov, reg0 / (1 + t))
        try:
            np.linalg.cholesky(cov_new)
        except np.linalg.LinAlgError:
            continue  # a step broken by rounding: keep the previous one (as gsmvi does)
        if np.all(np.isfinite(mu_new)):
            mu, cov = mu_new, cov_new
    return mu, cov


# --- The last layer ----------------------------------------------------------------------------


def head_score(x: np.ndarray, r: np.ndarray, w: np.ndarray) -> Any:
    """Score of the per-level log posterior over theta = (beta, gamma), each of size x.shape[1].

    x (N, P) features with a constant column, r (N,) scaled errors, w (N,) row weights.
    """
    p = x.shape[1]

    def score(theta: np.ndarray) -> np.ndarray:
        beta, gamma = theta[:, :p], theta[:, p:]
        resid = r[:, None] - x @ beta.T  # (N, B)
        inv_var = np.exp(-2 * np.clip(x @ gamma.T, -LOG_SD_CLIP, LOG_SD_CLIP))
        g_beta = (x * w[:, None]).T @ (resid * inv_var)  # (P, B)
        g_gamma = (x * w[:, None]).T @ (resid**2 * inv_var - 1)
        return np.concatenate([g_beta.T, g_gamma.T], axis=1) - theta / PRIOR_SD**2

    return score


def _design(feats: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """Standardized, then whitened (PCA) features, plus a constant column."""
    z = (feats - fit["mean"]) / fit["sd"] @ fit["proj"]
    return np.column_stack([z, np.ones(len(feats))])


def _whitening(feats: np.ndarray) -> dict[str, Any]:
    """The head features are strongly correlated (and some nearly constant), which makes the
    posterior badly conditioned; whitening them is what lets BaM converge on the real data."""
    mean, sd = feats.mean(0), feats.std(0) + 1e-6
    w, v = np.linalg.eigh(np.cov(((feats - mean) / sd).T))
    keep = w > 1e-6 * w.max()
    return {"mean": mean, "sd": sd, "proj": v[:, keep] / np.sqrt(w[keep])}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if not cols:
        raise ValueError("The table has no head-feature columns f0, f1, ...")
    return sorted(cols, key=lambda c: int(c[1:]))


def row_weights(df: pd.DataFrame) -> np.ndarray:
    """1 / (rows of the same profile at the same level)."""
    n = df.groupby([*KEY, "depth_m"])["temp_obs_c"].transform("size").to_numpy(float)
    return 1.0 / n


def thin(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One observation per profile and level (picked at random, fixed seed), so a profile counts
    once per level however finely the float samples it, and the fit stays fast."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.drop_duplicates([*KEY, "depth_m"]).sort_index()


def _neg_log_post(x: np.ndarray, r: np.ndarray) -> Any:
    p = x.shape[1]

    def f(theta: np.ndarray) -> float:
        log_sd = np.clip(x @ theta[p:], -LOG_SD_CLIP, LOG_SD_CLIP)
        ll = np.sum(-log_sd - 0.5 * ((r - x @ theta[:p]) * np.exp(-log_sd)) ** 2)
        return float(-(ll - 0.5 * np.sum(theta**2) / PRIOR_SD**2))

    return f


def laplace(x: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MAP and the inverse Fisher information there (beta and gamma blocks are uncorrelated
    in expectation). BaM starts from this and corrects it towards the non-Gaussian posterior."""
    from scipy.optimize import minimize

    p = x.shape[1]
    score = head_score(x, r, np.ones(len(r)))
    theta0 = np.zeros(2 * p)
    res = minimize(_neg_log_post(x, r), theta0, jac=lambda th: -score(th[None])[0],
                   method="L-BFGS-B", options={"maxiter": 2000})  # fmt: skip
    inv_var = np.exp(-2 * np.clip(x @ res.x[p:], -LOG_SD_CLIP, LOG_SD_CLIP))
    info_b = (x * inv_var[:, None]).T @ x + np.eye(p) / PRIOR_SD**2
    info_g = 2 * x.T @ x + np.eye(p) / PRIOR_SD**2
    cov = np.zeros((2 * p, 2 * p))
    cov[:p, :p], cov[p:, p:] = np.linalg.inv(info_b), np.linalg.inv(info_g)
    return res.x, (cov + cov.T) / 2


def fit_heads(df: pd.DataFrame, *, niter: int = 100, seed: int = SEED) -> dict[str, Any]:
    """Per level: whitening, s_d, the Laplace start and the BaM posterior (mu, cov)."""
    df = thin(df, seed)
    cols = feature_columns(df)
    fit: dict[str, Any] = {"columns": cols, **_whitening(df[cols].to_numpy(float))}
    levels: dict[float, dict[str, Any]] = {}
    for depth, g in df.groupby("depth_m"):
        err = (g["temp_obs_c"] - g["temp_pred_c"]).to_numpy(float)
        s_d = float(np.sqrt(np.mean(err**2)))
        x = _design(g[cols].to_numpy(float), fit)
        r = err / s_d
        p = x.shape[1]
        theta_map, cov0 = laplace(x, r)
        mu, cov = bam_fit(head_score(x, r, np.ones(len(r))), theta_map, cov0, niter=niter,
                          reg0=10.0, seed=seed)  # fmt: skip
        gap = float(np.max(np.abs(mu - theta_map) / np.sqrt(np.diag(cov0))))
        sd_ratio = np.sqrt(np.diag(cov) / np.diag(cov0))
        levels[float(depth)] = {"s": s_d, "mu": mu, "cov": cov, "n_profiles": len(g),
                                "map_gap_sd": gap}  # fmt: skip
        logger.info("Level %.1f m: %d profile-levels, s = %.3f °C, log-sd intercept %.2f, "
                    "BaM mean vs MAP: max %.2f Laplace sd, BaM sd / Laplace sd %.2f-%.2f",
                    depth, len(g), s_d, mu[2 * p - 1], gap, sd_ratio.min(),
                    sd_ratio.max())  # fmt: skip
    fit["levels"] = levels
    return fit


def predictive(df: pd.DataFrame, fit: dict[str, Any], seed: int = SEED) -> pd.DataFrame:
    """Mean, sd and PIT of each method at each row (levels the fit lacks are dropped)."""
    rng = np.random.default_rng(seed)
    out = []
    for depth, g in df.groupby("depth_m"):
        lev = fit["levels"].get(float(depth))
        if lev is None:
            continue
        obs, pred = g["temp_obs_c"].to_numpy(float), g["temp_pred_c"].to_numpy(float)
        s_d, mu, cov = lev["s"], lev["mu"], lev["cov"]
        x = _design(g[fit["columns"]].to_numpy(float), fit)
        p = x.shape[1]
        res = pd.DataFrame(index=g.index)
        res["const_mean"], res["const_sd"] = pred, s_d
        res["const_pit"] = ndtr((obs - pred) / s_d)
        m = pred + s_d * (x @ mu[:p])
        sd = s_d * np.exp(np.clip(x @ mu[p:], -LOG_SD_CLIP, LOG_SD_CLIP))
        res["bll_plugin_mean"], res["bll_plugin_sd"] = m, sd
        res["bll_plugin_pit"] = ndtr((obs - m) / sd)
        th = rng.multivariate_normal(mu, cov, POSTERIOR_DRAWS)
        ms = pred[:, None] + s_d * (x @ th[:, :p].T)
        ss = s_d * np.exp(np.clip(x @ th[:, p:].T, -LOG_SD_CLIP, LOG_SD_CLIP))
        mix_mean = ms.mean(1)
        res["bll_bam_mean"] = mix_mean
        res["bll_bam_sd"] = np.sqrt((ss**2 + ms**2).mean(1) - mix_mean**2)
        res["bll_bam_pit"] = ndtr((obs[:, None] - ms) / ss).mean(1)
        out.append(res)
    return df.join(pd.concat(out), how="inner")


# --- Scoring -----------------------------------------------------------------------------------


def _groups(df: pd.DataFrame) -> dict[str, np.ndarray]:
    z = df["depth_m"].to_numpy(float)
    bob = _in_bay_of_bengal(df["lat"].to_numpy(float), df["lon"].to_numpy(float))
    groups = {f"{d:g} m": z == d for d in sorted(df["depth_m"].unique())}
    for name, (lo, hi) in BANDS.items():
        groups[name] = (z >= lo) & (z <= hi) if lo == 0 else (z > lo) & (z <= hi)
    groups["Bay of Bengal 50-300 m"] = (
        bob & (z >= HEADLINE_DEPTH_MIN_M) & (z <= HEADLINE_DEPTH_MAX_M)
    )
    groups["all"] = np.ones(len(df), bool)
    return groups


def _cover(pit: np.ndarray, p: float) -> float:
    return float(np.mean(np.abs(pit - 0.5) <= p / 2)) if pit.size else float("nan")


def coverage_table(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Coverage at 68% and 95% (with float-bootstrap 95% intervals), mean sd and RMSE."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, sel in _groups(df).items():
        g = df[sel]
        floats = g["platform_number"].to_numpy(str)
        uniq = np.unique(floats)
        idx = {f: np.flatnonzero(floats == f) for f in uniq}
        draws = [np.concatenate([idx[f] for f in rng.choice(uniq, uniq.size)])
                 for _ in range(BOOTSTRAP_DRAWS)] if uniq.size else []  # fmt: skip
        obs = g["temp_obs_c"].to_numpy(float)
        for method in METHODS:
            pit = g[f"{method}_pit"].to_numpy(float)
            row: dict[str, Any] = {"group": name, "method": method, "n_obs": len(g),
                                   "n_floats": uniq.size}  # fmt: skip
            row["rmse_c"] = float(np.sqrt(np.mean((g[f"{method}_mean"] - obs) ** 2)))
            row["mean_sd_c"] = float(np.mean(g[f"{method}_sd"]))
            for p in (0.68, 0.95):
                tag = int(round(p * 100))
                row[f"cover{tag}"] = _cover(pit, p)
                boot = [_cover(pit[d], p) for d in draws]
                row[f"cover{tag}_lo"], row[f"cover{tag}_hi"] = (
                    np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    """Observed vs nominal coverage of central intervals, for the calibration plot."""
    rows = []
    for name, sel in _groups(df).items():
        if name not in ("all", "Bay of Bengal 50-300 m", *BANDS) or not sel.any():
            continue
        for method in METHODS:
            pit = df.loc[sel, f"{method}_pit"].to_numpy(float)
            for p in NOMINAL:
                rows.append({"group": name, "method": method, "nominal": p,
                             "observed": _cover(pit, p)})  # fmt: skip
    return pd.DataFrame(rows)


def plot_reliability(table: pd.DataFrame, path: Path, title: str) -> Path | None:
    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:  # matplotlib is not a dependency; Kaggle has it
        logger.info("matplotlib not installed: no plot for %s", path.name)
        return None
    groups = ["all", "0-50 m", "50-300 m", "300-1100 m", "Bay of Bengal 50-300 m"]
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4), sharey=True)
    for ax, grp in zip(axes, groups, strict=True):
        ax.plot([0, 1], [0, 1], color="0.7", lw=1)
        for method in METHODS:
            t = table[(table["group"] == grp) & (table["method"] == method)]
            ax.plot(t["nominal"], t["observed"], marker="o", ms=3, label=method)
        ax.set_title(grp)
        ax.set_xlabel("nominal coverage")
    axes[0].set_ylabel("observed coverage")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def analyse(
    fit_df: pd.DataFrame, check: dict[str, pd.DataFrame], out_dir: Path, *, niter: int = 100
) -> dict[str, pd.DataFrame]:
    """Fit on ``fit_df``; score each ``check`` table. Also the dev -> final half check."""
    metrics = out_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    halves = holdout_halves(fit_df)
    half = fit_df["platform_number"].map(dict(zip(halves["platform_number"], halves["half"])))
    tables: dict[str, pd.DataFrame] = {}
    jobs = [("val_halves", fit_df[half == "dev"], fit_df[half == "final"])]
    jobs += [(name, fit_df, df) for name, df in check.items()]
    for name, fit_on, score_on in jobs:
        logger.info("Fitting the last layer for %s on %d rows", name, len(fit_on))
        fit = fit_heads(fit_on, niter=niter)
        scored = predictive(score_on, fit)
        cov = coverage_table(scored)
        rel = reliability_table(scored)
        cov.to_csv(metrics / f"uncertainty_coverage_{name}.csv", index=False)
        rel.to_csv(metrics / f"uncertainty_reliability_{name}.csv", index=False)
        plot_reliability(rel, metrics / f"uncertainty_reliability_{name}.png", name)
        tables[name] = cov
    return tables


# --- Collecting matches with features ----------------------------------------------------------


class FeatureSampler:
    """``predict_store`` feature sink: keeps the head features at each held-out profile."""

    def __init__(self, profiles: pd.DataFrame, lat: np.ndarray, lon: np.ndarray) -> None:
        prof = profiles.groupby(KEY, as_index=False)[["time", "lat", "lon"]].first()
        li, lat_ok = _nearest_cell(lat, prof["lat"].to_numpy(float))
        lo, lon_ok = _nearest_cell(lon, prof["lon"].to_numpy(float))
        prof = prof.assign(i_lat=li, i_lon=lo, day=pd.DatetimeIndex(prof["time"]).normalize())
        self.profiles = prof[np.asarray(lat_ok & lon_ok)]
        self.by_day = {d: g for d, g in self.profiles.groupby("day")}
        self.rows: list[pd.DataFrame] = []

    def __call__(self, dates: pd.DatetimeIndex, feats: np.ndarray) -> None:
        for k, day in enumerate(pd.DatetimeIndex(dates).normalize()):
            g = self.by_day.get(day)
            if g is None:
                continue
            f = feats[k][:, g["i_lat"].to_numpy(), g["i_lon"].to_numpy()].T
            cols = {f"f{c}": f[:, c] for c in range(f.shape[1])}
            self.rows.append(pd.DataFrame({k_: g[k_].to_numpy() for k_ in KEY} | cols))

    def table(self) -> pd.DataFrame:
        if not self.rows:
            raise ValueError("No held-out profile fell on a predicted day")
        out = pd.concat(self.rows, ignore_index=True)
        out["platform_number"] = out["platform_number"].astype(str)
        return out


def collect(
    config: ExperimentConfig,
    checkpoint: Path,
    store_paths: list[Path],
    argo_paths: list[Path],
    holdout_path: Path,
    split: str,
    *,
    stats_path: Path | None = None,
    out_dir: Path,
    allow_test: bool = False,
    device_choice: str = "auto",
) -> Path:
    """Predict ``split`` once, keep the head features at the held-out floats, and write
    ``uncert_<split>.parquet``: the ``match_profiles`` rows plus f0..f{C-1}."""
    if split == "test" and not allow_test:
        raise PermissionError("The 2021 check reads the test year: needs the owner's go")
    holdout = _load_holdout(holdout_path)
    start, end = (pd.Timestamp(d) for d in getattr(config.splits, split))
    profiles = load_profiles(argo_paths, holdout, start, end)
    ds = open_parts(store_paths, stats_path)
    sampler = FeatureSampler(
        profiles, np.asarray(ds["lat"].values, float), np.asarray(ds["lon"].values, float)
    )
    logger.info("%s: %d held-out profiles from %d floats", split, len(sampler.profiles),
                sampler.profiles["platform_number"].nunique())  # fmt: skip
    pred_path = predict_store(
        config,
        store_paths,
        checkpoint,
        out_dir=out_dir / "uncertainty",
        stats_path=stats_path,
        device_choice=device_choice,
        run_id=f"uncert_{split}",
        split=split,
        allow_test=allow_test,
        argo_paths=argo_paths,
        holdout=holdout,
        feature_sink=sampler,
    )
    refs = xr.Dataset({v: ds[v] if v in ds else xr.full_like(ds["thetao"], np.nan)
                       for v in ("grep_mean", "armor3d")})  # fmt: skip
    matches = match_profiles(profiles, xr.open_zarr(pred_path), refs)
    matches["platform_number"] = matches["platform_number"].astype(str)
    table = matches.merge(sampler.table(), on=KEY, how="inner", validate="many_to_one")
    for col in ("temp_glorys_c", "temp_grep_c", "temp_armor3d_c"):
        table[col] = table[col].astype(float)
    dest = out_dir / f"uncert_{split}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(dest, index=False)
    logger.info("Wrote %s: %d rows", dest, len(table))
    return dest


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian last layer error bars (BaM).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--config", default="configs/experiment.yaml")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--store", nargs="+", required=True)
    c.add_argument("--stats", default=None)
    c.add_argument("--argo", nargs="+", required=True)
    c.add_argument("--holdout", required=True)
    c.add_argument("--split", choices=["val", "test"], default="val")
    c.add_argument("--out-dir", default="outputs")
    c.add_argument("--device", default="auto")
    c.add_argument("--owner-go", action="store_true", help="Needed for --split test")
    a = sub.add_parser("analyse")
    a.add_argument("--fit", required=True, help="uncert_val.parquet (validation years)")
    a.add_argument("--check", default=None, help="uncert_test.parquet (2021, second look)")
    a.add_argument("--out-dir", default="outputs")
    a.add_argument("--niter", type=int, default=100)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_cli().parse_args()
    if args.cmd == "collect":
        collect(
            load_config(args.config),
            Path(args.checkpoint),
            [Path(p) for p in args.store],
            [Path(p) for p in args.argo],
            Path(args.holdout),
            args.split,
            stats_path=Path(args.stats) if args.stats else None,
            out_dir=Path(args.out_dir),
            allow_test=args.owner_go,
            device_choice=args.device,
        )
        return
    check = {"test_2021": pd.read_parquet(args.check)} if args.check else {}
    tables = analyse(pd.read_parquet(args.fit), check, Path(args.out_dir), niter=args.niter)
    cols = ["group", "method", "n_floats", "rmse_c", "mean_sd_c", "cover68", "cover68_lo",
            "cover68_hi", "cover95", "cover95_lo", "cover95_hi"]  # fmt: skip
    for name, table in tables.items():
        keep = table[~table["group"].str.endswith(" m") | table["group"].str.contains("-")]
        print(f"\n{name}\n" + keep[cols].round(3).to_string(index=False))
    print(json.dumps({"nominal": [0.68, 0.95]}))


if __name__ == "__main__":
    main()
