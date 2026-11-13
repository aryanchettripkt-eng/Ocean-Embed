"""Hierarchical last layer: does a per-region error size sharpen the Bay of Bengal bars?

The rule is docs/uncertainty/HIERARCHY.md, written before any fit. Only the 2018-2020 table
(``uncert_val.parquet``) is used; 2021 is never read here.

Per depth level, with ``x`` the design of ``uncertainty`` (whitened head features + constant)
and ``g`` the row's region (Bay of Bengal, Arabian Sea, equatorial):

    r ~ N(beta . x, exp((gamma + delta_g) . x)^2)
    beta, gamma ~ N(0, 0.5^2),  delta_g ~ N(0, tau^2 I),  log tau ~ N(log 0.2, 1)

Parameters are laid out as ``[beta (P), gamma (P), delta or eta (R * P), log tau]``.

- centred: ``delta_g`` is a free parameter (the funnel geometry);
- non-centred: ``delta_g = tau * eta_g``, ``eta_g ~ N(0, I)``.

Usage (from ml/):

    python -m oceanscope_ml.evaluation.uncertainty_hier signal --fit uncert_val.parquet --out-dir o
    python -m oceanscope_ml.evaluation.uncertainty_hier cv --fit uncert_val.parquet --out-dir o
    python -m oceanscope_ml.evaluation.uncertainty_hier reference --fit uncert_val.parquet \
        --out-dir o [--nuts-draws o/nuts_draws.npz]

The NUTS reference itself is ``scripts/hier_nuts.py`` (numpyro, not a dependency of the
package; run it in a separate environment).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from oceanscope_ml.evaluation import uncertainty as un
from oceanscope_ml.evaluation.argo_validation import (
    HEADLINE_DEPTH_MAX_M,
    HEADLINE_DEPTH_MIN_M,
    _in_bay_of_bengal,
    float_region,
)
from oceanscope_ml.evaluation.evaluate import EQUATORIAL_MAX_LAT

logger = logging.getLogger("oceanscope_ml.evaluation.uncertainty_hier")

REGIONS = ("bay_of_bengal", "arabian_sea", "equatorial")
LOG_TAU_MEAN = float(np.log(0.2))
LOG_TAU_SD = 1.0
FOLD_SEED = 2026
N_FOLDS = 5
HIER_METHODS = ("hier_laplace", "hier_bam_c", "hier_bam_nc")
ALL_METHODS = ("const", "bll_bam", *HIER_METHODS)
REFERENCE_DEPTHS = (92.326, 186.126, 380.213)  # the table's levels nearest 100, 200, 500 m
MONSOON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "summer", 10: "autumn",
           11: "autumn"}  # fmt: skip


# --- Regions and folds -------------------------------------------------------------------------


def region_index(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """0 Bay of Bengal, 1 Arabian Sea, 2 equatorial (the §6.3 regions, made disjoint)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    out = np.where(lat < EQUATORIAL_MAX_LAT, 2, 1)
    return np.where(_in_bay_of_bengal(lat, lon), 0, out).astype(int)


def float_folds(df: pd.DataFrame, k: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    """Fold of each float, stratified by home region (the region of its median position).
    Within a region the floats are shuffled and dealt round-robin, continuing the count across
    regions so the folds stay balanced in size."""
    pos = df.groupby("platform_number")[["lat", "lon"]].median()
    home = pd.Series([float_region(float(a), float(o)) for a, o in zip(pos["lat"], pos["lon"],
                                                                         strict=True)],
                     index=pos.index.astype(str))  # fmt: skip
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    n = 0
    for region in REGIONS:
        floats = np.sort(home.index[home == region].to_numpy())
        for f in floats[rng.permutation(floats.size)]:
            folds[str(f)] = n % k
            n += 1
    return folds


# --- The model -----------------------------------------------------------------------------------


def _split(theta: np.ndarray, p: int, n_reg: int) -> tuple[np.ndarray, ...]:
    """(B, D) -> beta (B, P), gamma (B, P), z (B, R, P), log tau (B,)."""
    beta, gamma = theta[:, :p], theta[:, p : 2 * p]
    z = theta[:, 2 * p : 2 * p + n_reg * p].reshape(len(theta), n_reg, p)
    return beta, gamma, z, theta[:, -1]


def n_params(p: int, n_reg: int = len(REGIONS)) -> int:
    return 2 * p + n_reg * p + 1


def region_gamma(theta: np.ndarray, p: int, n_reg: int, centred: bool) -> np.ndarray:
    """gamma_g = gamma + delta_g for each draw: (B, R, P)."""
    _, gamma, z, s = _split(theta, p, n_reg)
    delta = z if centred else np.exp(s)[:, None, None] * z
    return np.asarray(gamma[:, None, :] + delta)


def hier_log_post(
    x: np.ndarray, r: np.ndarray, g: np.ndarray, n_reg: int, centred: bool
) -> tuple[Any, Any]:
    """Log posterior and its score, both vectorised over draws ``theta`` (B, D).

    x (N, P) design, r (N,) scaled errors, g (N,) region index."""
    p = x.shape[1]
    onehot = np.eye(n_reg)[g]  # (N, R)
    prior_var = un.PRIOR_SD**2

    def parts(theta: np.ndarray) -> tuple[np.ndarray, ...]:
        beta, gamma, z, s = _split(theta, p, n_reg)
        gam = region_gamma(theta, p, n_reg, centred)  # (B, R, P)
        log_sd = np.einsum("np,bnp->nb", x, gam[:, g, :])
        log_sd = np.clip(log_sd, -un.LOG_SD_CLIP, un.LOG_SD_CLIP)
        resid = r[:, None] - x @ beta.T
        inv_var = np.exp(-2 * log_sd)
        return beta, gamma, z, s, log_sd, resid, inv_var

    def log_post(theta: np.ndarray) -> np.ndarray:
        beta, gamma, z, s, log_sd, resid, inv_var = parts(theta)
        ll = np.sum(-log_sd - 0.5 * resid**2 * inv_var, axis=0)
        lp = -0.5 * (np.sum(beta**2, 1) + np.sum(gamma**2, 1)) / prior_var
        k = z.shape[1] * z.shape[2]
        zz = np.sum(z**2, axis=(1, 2))
        lp_z = -k * s - 0.5 * zz * np.exp(-2 * s) if centred else -0.5 * zz
        lp_s = -0.5 * ((s - LOG_TAU_MEAN) / LOG_TAU_SD) ** 2
        return np.asarray(ll + lp + lp_z + lp_s)

    def score(theta: np.ndarray) -> np.ndarray:
        beta, gamma, z, s, log_sd, resid, inv_var = parts(theta)
        a = resid**2 * inv_var - 1  # d ll / d log_sd, (N, B)
        g_beta = (x.T @ (resid * inv_var)).T - beta / prior_var
        per_reg = np.einsum("nr,np,nb->brp", onehot, x, a)  # (B, R, P)
        g_gamma = per_reg.sum(1) - gamma / prior_var
        k = z.shape[1] * z.shape[2]
        zz = np.sum(z**2, axis=(1, 2))
        if centred:
            g_z = per_reg - z * np.exp(-2 * s)[:, None, None]
            g_s = -k + zz * np.exp(-2 * s)
        else:
            tau = np.exp(s)
            g_z = tau[:, None, None] * per_reg - z
            g_s = tau * np.sum(per_reg * z, axis=(1, 2))
        g_s = g_s - (s - LOG_TAU_MEAN) / LOG_TAU_SD**2
        return np.concatenate([g_beta, g_gamma, g_z.reshape(len(theta), -1), g_s[:, None]], 1)

    return log_post, score


def _hessian(score: Any, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    d = theta.size
    h = np.empty((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        h[j] = (score((theta + e)[None])[0] - score((theta - e)[None])[0]) / (2 * eps)
    return (h + h.T) / 2


def _pd_inverse(neg_h: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(neg_h)
    w = np.clip(w, 1e-6 * max(w.max(), 1e-12), None)
    return np.asarray((v / w) @ v.T)


def hier_laplace(
    x: np.ndarray, r: np.ndarray, g: np.ndarray, n_reg: int = len(REGIONS)
) -> tuple[np.ndarray, np.ndarray]:
    """MAP and inverse negative Hessian of the non-centred posterior (the centred joint mode
    can sit at tau -> 0)."""
    p = x.shape[1]
    log_post, score = hier_log_post(x, r, g, n_reg, centred=False)
    start = np.zeros(n_params(p, n_reg))
    start[-1] = LOG_TAU_MEAN
    res = minimize(lambda th: -float(log_post(th[None])[0]), start,
                   jac=lambda th: -score(th[None])[0], method="L-BFGS-B",
                   options={"maxiter": 5000})  # fmt: skip
    return res.x, _pd_inverse(-_hessian(score, res.x))


def to_centred(mu: np.ndarray, cov: np.ndarray, p: int, n_reg: int) -> tuple[np.ndarray, ...]:
    """Map a non-centred Gaussian to centred coordinates (delta = tau * eta), to first order."""
    d = mu.size
    lo, hi = 2 * p, 2 * p + n_reg * p
    tau = float(np.exp(mu[-1]))
    jac = np.eye(d)
    jac[lo:hi, lo:hi] = tau * np.eye(hi - lo)
    jac[lo:hi, -1] = tau * mu[lo:hi]
    out = mu.copy()
    out[lo:hi] = tau * mu[lo:hi]
    c = jac @ cov @ jac.T
    return out, (c + c.T) / 2


def fit_level(
    x: np.ndarray, r: np.ndarray, g: np.ndarray, *, niter: int = 100, seed: int = un.SEED
) -> dict[str, dict[str, Any]]:
    """Laplace, BaM-centred and BaM-non-centred fits at one level."""
    p, n_reg = x.shape[1], len(REGIONS)
    mu_l, cov_l = hier_laplace(x, r, g, n_reg)
    out: dict[str, dict[str, Any]] = {"hier_laplace": {"mu": mu_l, "cov": cov_l, "centred": False}}
    _, score_nc = hier_log_post(x, r, g, n_reg, centred=False)
    mu, cov = un.bam_fit(score_nc, mu_l, cov_l, niter=niter, reg0=10.0, seed=seed)
    out["hier_bam_nc"] = {"mu": mu, "cov": cov, "centred": False}
    mu_c0, cov_c0 = to_centred(mu_l, cov_l, p, n_reg)
    _, score_c = hier_log_post(x, r, g, n_reg, centred=True)
    trace: list[float] = []
    mu, cov = _bam_traced(score_c, mu_c0, cov_c0, niter=niter, seed=seed, trace=trace)
    out["hier_bam_c"] = {"mu": mu, "cov": cov, "centred": True, "log_tau_trace": trace}
    for name, f in out.items():
        sd0 = np.sqrt(np.diag(cov_l if not f["centred"] else cov_c0))
        start = mu_l if not f["centred"] else mu_c0
        f["map_gap_sd"] = float(np.max(np.abs(f["mu"] - start) / sd0))
        f["log_tau_mean"], f["log_tau_sd"] = float(f["mu"][-1]), float(np.sqrt(f["cov"][-1, -1]))
    return out


def _bam_traced(
    score: Any, mu: np.ndarray, cov: np.ndarray, *, niter: int, seed: int, trace: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """``un.bam_fit`` one step at a time, recording the log tau mean after each step."""
    rng_seed = seed
    for t in range(niter):
        mu, cov = un.bam_fit(score, mu, cov, niter=1, reg0=10.0 / (1 + t), seed=rng_seed + t)
        trace.append(float(mu[-1]))
    return mu, cov


# --- Prediction and scoring ---------------------------------------------------------------------


def _mixture_scores(
    res: pd.DataFrame, name: str, obs: np.ndarray, ms: np.ndarray, ss: np.ndarray
) -> None:
    from scipy.special import logsumexp, ndtr

    mix_mean = ms.mean(1)
    res[f"{name}_mean"] = mix_mean
    res[f"{name}_sd"] = np.sqrt((ss**2 + ms**2).mean(1) - mix_mean**2)
    res[f"{name}_pit"] = ndtr((obs[:, None] - ms) / ss).mean(1)
    res[f"{name}_crps"] = un.mixture_crps(obs, ms, ss)
    log_dens = -0.5 * np.log(2 * np.pi * ss**2) - 0.5 * ((obs[:, None] - ms) / ss) ** 2
    res[f"{name}_nll"] = -(logsumexp(log_dens, axis=1) - np.log(ms.shape[1]))


def fit_hier(df: pd.DataFrame, *, niter: int = 100, seed: int = un.SEED) -> dict[str, Any]:
    """The global layer (``un.fit_heads`` with ``un.DEFAULT_FIT``) plus the three hierarchical
    fits at each level, on the same whitening and the same thinned rows."""
    fit = un.fit_heads(df, niter=niter, seed=seed, **un.DEFAULT_FIT)
    rows = un.thin(df, seed)
    cols = fit["columns"]
    hier: dict[float, dict[str, Any]] = {}
    for depth, grp in rows.groupby("depth_m"):
        lev = fit["levels"][float(depth)]
        x = un._design(grp[cols].to_numpy(float), fit)
        r = (grp["temp_obs_c"] - grp["temp_pred_c"]).to_numpy(float) / lev["s"]
        g = region_index(grp["lat"].to_numpy(float), grp["lon"].to_numpy(float))
        hier[float(depth)] = fit_level(x, r, g, niter=niter, seed=seed)
        logger.info("Level %.1f m: log tau Laplace %.2f±%.2f, BaM-c %.2f±%.2f, BaM-nc %.2f±%.2f",
                    depth, *[hier[float(depth)][m][k] for m in HIER_METHODS
                             for k in ("log_tau_mean", "log_tau_sd")])  # fmt: skip
    fit["hier"] = hier
    return fit


def predictive_hier(df: pd.DataFrame, fit: dict[str, Any], seed: int = un.SEED) -> pd.DataFrame:
    """``un.predictive`` (const, bll_plugin, bll_bam) plus each hierarchical posterior's
    predictive mixture, with the same cap as the global layer."""
    base = un.predictive(df, fit, seed)
    rng = np.random.default_rng(seed + 1)
    out = []
    for depth, grp in base.groupby("depth_m"):
        lev, hier = fit["levels"][float(depth)], fit["hier"][float(depth)]
        x = un._design(grp[fit["columns"]].to_numpy(float), fit)
        p = x.shape[1]
        g = region_index(grp["lat"].to_numpy(float), grp["lon"].to_numpy(float))
        obs, pred = grp["temp_obs_c"].to_numpy(float), grp["temp_pred_c"].to_numpy(float)
        res = pd.DataFrame(index=grp.index)
        for name in HIER_METHODS:
            f = hier[name]
            th = rng.multivariate_normal(f["mu"], f["cov"], un.POSTERIOR_DRAWS)
            gam = region_gamma(th, p, len(REGIONS), f["centred"])  # (B, R, P)
            log_sd = np.einsum("np,bnp->nb", x, gam[:, g, :])
            ss = lev["s"] * np.exp(np.clip(log_sd, lev["log_sd_lo"], lev["log_sd_hi"]))
            ms = pred[:, None] + lev["s"] * (x @ th[:, :p].T)
            _mixture_scores(res, name, obs, ms, ss)
        out.append(res)
    return base.join(pd.concat(out), how="inner")


def cross_validate(
    df: pd.DataFrame, *, niter: int = 100, k: int = N_FOLDS, with_hier: bool = True
) -> pd.DataFrame:
    """Out-of-fold predictions for every row, fitting on the other k-1 folds. Also keeps the
    out-of-fold whitened directions (z0..z7) for the signal check."""
    folds = float_folds(df, k)
    fold = df["platform_number"].astype(str).map(folds).to_numpy()
    scored = []
    for j in range(k):
        train, test = df[fold != j], df[fold == j]
        logger.info("Fold %d: fit on %d rows, score %d rows", j, len(train), len(test))
        if with_hier:
            fit = fit_hier(train, niter=niter)
            s = predictive_hier(test, fit)
        else:
            fit = un.fit_heads(train, niter=niter, **un.DEFAULT_FIT)
            s = un.predictive(test, fit)
        z = un._design(s[fit["columns"]].to_numpy(float), fit)[:, :-1]
        s = s.assign(fold=j, **{f"z{i}": z[:, i] for i in range(z.shape[1])})
        scored.append(s)
    return pd.concat(scored)


def paired_crps_diff(
    df: pd.DataFrame, a: str, b: str, seed: int = un.SEED
) -> tuple[float, float, float]:
    """Mean CRPS(a) - CRPS(b) with a float-bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    d = (df[f"{a}_crps"] - df[f"{b}_crps"]).to_numpy(float)
    floats = df["platform_number"].astype(str).to_numpy()
    uniq = np.unique(floats)
    idx = {f: np.flatnonzero(floats == f) for f in uniq}
    boot = [d[np.concatenate([idx[f] for f in rng.choice(uniq, uniq.size)])].mean()
            for _ in range(un.BOOTSTRAP_DRAWS)]  # fmt: skip
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# --- Step 2: signal in the Bay of Bengal ---------------------------------------------------------


def bob_rows(df: pd.DataFrame) -> pd.Series:
    z = df["depth_m"]
    return (region_index(df["lat"].to_numpy(float), df["lon"].to_numpy(float)) == 0) & (
        (z >= HEADLINE_DEPTH_MIN_M) & (z <= HEADLINE_DEPTH_MAX_M)
    )


def add_physical(df: pd.DataFrame, surface_m: float = 5.0) -> pd.DataFrame:
    """Monsoon phase and the U-Net's own stratification: predicted T at the level nearest
    ``surface_m`` minus predicted T at the row's level (same profile)."""
    out = df.copy()
    out["monsoon"] = pd.to_datetime(out["time"]).dt.month.map(MONSOON)
    levels = np.sort(out["depth_m"].unique())
    top = levels[np.argmin(np.abs(levels - surface_m))]
    sst = (out[out["depth_m"] == top].groupby(un.KEY)["temp_pred_c"].mean()
           .rename("pred_top_c"))  # fmt: skip
    out = out.join(sst, on=un.KEY)
    out["strat_pred_c"] = out["pred_top_c"] - out["temp_pred_c"]
    return out


def _cv_r2(
    x: np.ndarray, y: np.ndarray, floats: np.ndarray, k: int = N_FOLDS, seed: int = FOLD_SEED
) -> float:
    uniq = np.unique(floats)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(uniq.size) % k,
                       strict=True))  # fmt: skip
    fold = np.array([fold_of[f] for f in floats])
    pred = np.empty_like(y)
    design = np.column_stack([x, np.ones(len(y))])
    for j in range(k):
        tr = fold != j
        coef, *_ = np.linalg.lstsq(design[tr], y[tr], rcond=None)
        pred[~tr] = design[~tr] @ coef
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def signal_table(scored: pd.DataFrame, seed: int = un.SEED) -> pd.DataFrame:
    """Spearman of each candidate with |r| (float-bootstrap interval) and CV R^2 on log r^2,
    on Bay of Bengal 50-300 m out-of-fold rows."""
    df = add_physical(scored)
    df = df[bob_rows(df).to_numpy()].dropna(subset=["strat_pred_c"])
    err = (df["temp_obs_c"] - df["temp_pred_c"]).to_numpy(float)
    r = err / df["const_sd"].to_numpy(float)
    y = np.log(r**2 + 1e-12)
    floats = df["platform_number"].astype(str).to_numpy()
    uniq = np.unique(floats)
    idx = {f: np.flatnonzero(floats == f) for f in uniq}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([idx[f] for f in rng.choice(uniq, uniq.size)])
             for _ in range(un.BOOTSTRAP_DRAWS)]  # fmt: skip
    cands: dict[str, np.ndarray] = {
        "global layer log sd": np.log(df["bll_plugin_sd"] / df["const_sd"]).to_numpy()
    }
    for c in sorted([c for c in df.columns if c.startswith("z") and c[1:].isdigit()],
                    key=lambda c: int(c[1:])):  # fmt: skip
        cands[f"whitened direction {c[1:]}"] = df[c].to_numpy(float)
    cands["predicted stratification (T_5m - T)"] = df["strat_pred_c"].to_numpy(float)
    cands["depth"] = df["depth_m"].to_numpy(float)
    rows = []
    for name, v in cands.items():
        rho = spearmanr(v, np.abs(r)).statistic
        boot = [spearmanr(v[d], np.abs(r[d])).statistic for d in draws]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"candidate": name, "spearman": rho, "spearman_lo": lo, "spearman_hi": hi,
                     "cv_r2": _cv_r2(v[:, None], y, floats)})  # fmt: skip
    phase = pd.get_dummies(df["monsoon"]).to_numpy(float)[:, 1:]
    rows.append({"candidate": "monsoon phase (one-hot)", "spearman": np.nan,
                 "spearman_lo": np.nan, "spearman_hi": np.nan,
                 "cv_r2": _cv_r2(phase, y, floats)})  # fmt: skip
    by_phase = df.assign(abs_r=np.abs(r)).groupby("monsoon")["abs_r"].agg(["size", "mean"])
    logger.info("Bay of Bengal 50-300 m |r| by monsoon phase:\n%s", by_phase)
    allz = np.column_stack([cands[n] for n in cands if n.startswith("whitened")])
    rows.append({"candidate": "all 8 directions together", "spearman": np.nan,
                 "spearman_lo": np.nan, "spearman_hi": np.nan,
                 "cv_r2": _cv_r2(allz, y, floats)})  # fmt: skip
    out = pd.DataFrame(rows)
    out["n_rows"], out["n_floats"] = len(df), uniq.size
    return out


# --- Step 4: the reference posterior ------------------------------------------------------------


def reference_inputs(df: pd.DataFrame, path: Path, seed: int = un.SEED) -> Path:
    """Design, scaled errors and regions at ``REFERENCE_DEPTHS`` on all 2018-2020 rows (thinned,
    whitening fitted on all of them), for ``scripts/hier_nuts.py`` and ``compare_reference``."""
    rows = un.thin(df, seed)
    cols = un.feature_columns(rows)
    fit = un._whitening(rows[cols].to_numpy(float), un.DEFAULT_FIT["n_dirs"],
                        un.DEFAULT_FIT["clip"])  # fmt: skip
    arrays: dict[str, np.ndarray] = {"depths": np.array(REFERENCE_DEPTHS)}
    for k, depth in enumerate(REFERENCE_DEPTHS):
        grp = rows[np.isclose(rows["depth_m"], depth, atol=1e-3)]
        err = (grp["temp_obs_c"] - grp["temp_pred_c"]).to_numpy(float)
        arrays[f"x{k}"] = un._design(grp[cols].to_numpy(float), fit)
        arrays[f"r{k}"] = err / np.sqrt(np.mean(err**2))
        arrays[f"g{k}"] = region_index(grp["lat"].to_numpy(float), grp["lon"].to_numpy(float))
    np.savez(path, **arrays)  # type: ignore[arg-type]
    return path


def _nc_draws(f: dict[str, Any], p: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2): log tau and delta_BoB on the constant column, from a fitted Gaussian."""
    th = rng.multivariate_normal(f["mu"], f["cov"], n)
    gam = region_gamma(th, p, len(REGIONS), f["centred"])
    return np.column_stack([th[:, -1], gam[:, 0, -1] - th[:, 2 * p - 1]])


def compare_reference(inputs: Path, nuts: Path, out_dir: Path, *, niter: int = 100) -> pd.DataFrame:
    """Laplace, BaM-centred and BaM-non-centred against the NUTS draws at each reference depth:
    sd of log tau and of delta_BoB (constant column), each as a ratio to NUTS."""
    data, ref = np.load(inputs), np.load(nuts)
    rng = np.random.default_rng(un.SEED)
    rows, scatter = [], {}
    for k, depth in enumerate(data["depths"]):
        x, r, g = data[f"x{k}"], data[f"r{k}"], data[f"g{k}"]
        p = x.shape[1]
        fits = fit_level(x, r, g, niter=niter)
        nuts_s, nuts_d = ref[f"log_tau{k}"], ref[f"delta_bob{k}"]
        scatter[float(depth)] = (nuts_s, nuts_d)
        base = {"depth_m": float(depth), "n_rows": len(r),
                "nuts_divergences_nc": int(ref[f"div_nc{k}"]),
                "nuts_divergences_c": int(ref[f"div_c{k}"]),
                "nuts_max_rhat": float(ref[f"rhat{k}"])}  # fmt: skip
        rows.append({**base, "method": "nuts", "log_tau_mean": nuts_s.mean(),
                     "log_tau_sd": nuts_s.std(), "delta_bob_mean": nuts_d.mean(),
                     "delta_bob_sd": nuts_d.std(),
                     "sd_ratio_log_tau": 1.0, "sd_ratio_delta_bob": 1.0})  # fmt: skip
        for name in HIER_METHODS:
            f = fits[name]
            d = _nc_draws(f, p, 4000, rng)
            row = {**base, "method": name, "log_tau_mean": d[:, 0].mean(),
                   "log_tau_sd": d[:, 0].std(), "delta_bob_mean": d[:, 1].mean(),
                   "delta_bob_sd": d[:, 1].std(),
                   "map_gap_sd": f["map_gap_sd"]}  # fmt: skip
            row["sd_ratio_log_tau"] = row["log_tau_sd"] / nuts_s.std()
            row["sd_ratio_delta_bob"] = row["delta_bob_sd"] / nuts_d.std()
            if name == "hier_bam_c":
                tr = np.array(f["log_tau_trace"][-20:])
                row["log_tau_drift_last20"] = float(tr[-1] - tr[0])
            rows.append(row)
    table = pd.DataFrame(rows)
    table["funnel"] = (table["method"] != "nuts") & (table["sd_ratio_log_tau"] < 0.7)
    table.to_csv(out_dir / "reference_compare.csv", index=False)
    _plot_funnel(scatter, table, out_dir / "funnel_nuts.png")
    return table


def _plot_funnel(scatter: dict[float, Any], table: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(scatter), figsize=(4.5 * len(scatter), 4), sharey=True)
    for ax, (depth, (s, d)) in zip(np.atleast_1d(axes), scatter.items(), strict=True):
        ax.scatter(s, d, s=2, alpha=0.3, color="0.4", label="NUTS draws")
        t = table[(table["depth_m"] == depth) & (table["method"] != "nuts")]
        for _, row in t.iterrows():
            ax.errorbar(row["log_tau_mean"], row["delta_bob_mean"], xerr=2 * row["log_tau_sd"],
                        yerr=2 * row["delta_bob_sd"], fmt="o", ms=4, capsize=3,
                        label=f"{row['method']} (mean ± 2 sd)")  # fmt: skip
        ax.set_title(f"{depth:.0f} m")
        ax.set_xlabel("log tau")
    np.atleast_1d(axes)[0].set_ylabel("delta Bay of Bengal (constant column)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- CLI -----------------------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Hierarchical last layer (2018-2020 only).")
    ap.add_argument("command", choices=["signal", "cv", "reference-inputs", "reference"])
    ap.add_argument("--nuts-draws", help="output of scripts/hier_nuts.py (reference)")
    ap.add_argument("--fit", required=True, help="uncert_val.parquet (2018-2020)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--niter", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.fit)
    if pd.to_datetime(df["time"]).dt.year.max() > 2020:
        raise PermissionError("This analysis uses 2018-2020 only; the table has later rows.")
    if args.command == "reference-inputs":
        print(reference_inputs(df, out / "reference_inputs.npz"))
        return
    if args.command == "reference":
        table = compare_reference(out / "reference_inputs.npz", Path(args.nuts_draws), out,
                                  niter=args.niter)  # fmt: skip
        print(table.to_string(index=False))
        return
    scored = cross_validate(df, niter=args.niter, with_hier=args.command == "cv")
    tag = "global" if args.command == "signal" else "hier"
    scored.drop(columns=un.feature_columns(scored)).to_parquet(out / f"cv_{tag}.parquet")
    if args.command == "signal":
        sig = signal_table(scored)
        sig.to_csv(out / "signal_bob.csv", index=False)
        print(sig.to_string(index=False))
    table = un.coverage_table(scored, methods=un.METHODS if tag == "global" else ALL_METHODS)
    table.to_csv(out / f"cv_scores_{tag}.csv", index=False)
    print(table[table["group"].isin(["all", "Bay of Bengal 50-300 m", *un.BANDS])].to_string())
    if tag == "hier":
        bob = scored[bob_rows(scored).to_numpy()]
        for m in HIER_METHODS:
            print(m, "minus bll_bam CRPS in BoB 50-300 m:", paired_crps_diff(bob, m, "bll_bam"))


if __name__ == "__main__":
    main()
