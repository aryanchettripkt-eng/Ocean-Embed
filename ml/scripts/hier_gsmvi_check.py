"""Cross-check of the BaM collapse at 380 m with the official GSM-VI code, plus the
score-based divergence at each fitted Gaussian (docs/uncertainty/HIERARCHY.md, Step 4 check).

Separate environment (not dependencies of oceanscope_ml): jax, numpyro and GSM-VI
(github.com/modichirag/GSM-VI, ``pip install -e .``). Inputs from the package:
``reference_inputs.npz`` (``uncertainty_hier reference-inputs``) and ``mine_380.npz`` (this
repo's Laplace and BaM fits at 380 m, saved by the caller).

    python scripts/hier_gsmvi_check.py <reference_inputs.npz> <mine_380.npz> <out.npz>

What it does, at 380 m only (the third reference depth):
1. NUTS (numpyro, non-centred, 4 x 1000 draws), keeping every parameter, so the
   "NUTS-matched Gaussian" N(mean, cov of the draws) can be formed.
2. The official ``gsmvi.bam.BaM`` with the paper's ``linear`` schedule (reg0 / t), from three
   starts: this repo's Laplace fit, the NUTS-matched Gaussian, and N(0, I); batch 64; 100 and
   1000 iterations. Same for the centred parameterisation from the Laplace start.
3. The score-based divergence that BaM minimises (Cai et al. 2024, eq. 4),
   D(q; p) = E_q || Cov(q)^{1/2} (grad log q - grad log p) ||^2, by Monte Carlo (20,000
   draws in chunks of 500, the same draws for every q through a common normal base), at each
   Gaussian.
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from gsmvi.bam import BaM, Regularizers  # noqa: E402
from jax import random  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

numpyro.set_host_device_count(4)
PRIOR_SD, LOG_TAU_MEAN, LOG_TAU_SD, N_REG, CLIP = 0.5, float(np.log(0.2)), 1.0, 3, 8.0


def log_post_fn(x, r, g, centred):
    """Same posterior as oceanscope_ml.evaluation.uncertainty_hier.hier_log_post (one theta)."""
    x, r, g = jnp.asarray(x), jnp.asarray(r), jnp.asarray(g)
    p = x.shape[1]

    def lp(theta):
        beta, gamma = theta[:p], theta[p : 2 * p]
        z = theta[2 * p : 2 * p + N_REG * p].reshape(N_REG, p)
        s = theta[-1]
        delta = z if centred else jnp.exp(s) * z
        log_sd = jnp.clip(jnp.sum(x * (gamma[None, :] + delta[g]), axis=1), -CLIP, CLIP)
        resid = r - x @ beta
        ll = jnp.sum(-log_sd - 0.5 * resid**2 * jnp.exp(-2 * log_sd))
        prior = -0.5 * (jnp.sum(beta**2) + jnp.sum(gamma**2)) / PRIOR_SD**2
        k = N_REG * p
        prior_z = (
            -k * s - 0.5 * jnp.sum(z**2) * jnp.exp(-2 * s) if centred else -0.5 * jnp.sum(z**2)
        )
        prior_s = -0.5 * ((s - LOG_TAU_MEAN) / LOG_TAU_SD) ** 2
        return ll + prior + prior_z + prior_s

    return lp


def nuts_draws(x, g, r):
    p = x.shape[1]

    def model():
        beta = numpyro.sample("beta", dist.Normal(0, PRIOR_SD).expand([p]))
        gamma = numpyro.sample("gamma", dist.Normal(0, PRIOR_SD).expand([p]))
        eta = numpyro.sample("eta", dist.Normal(0, 1).expand([N_REG, p]))
        s = numpyro.sample("log_tau", dist.Normal(LOG_TAU_MEAN, LOG_TAU_SD))
        log_sd = jnp.sum(x * (gamma[None, :] + jnp.exp(s) * eta[g]), axis=1)
        numpyro.sample("r", dist.Normal(x @ beta, jnp.exp(log_sd)), obs=r)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=1000, num_samples=1000,
                num_chains=4, progress_bar=False)  # fmt: skip
    mcmc.run(random.PRNGKey(7), extra_fields=("diverging",))
    s = mcmc.get_samples()
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    theta = np.column_stack([np.asarray(s["beta"]), np.asarray(s["gamma"]),
                             np.asarray(s["eta"]).reshape(len(s["beta"]), -1),
                             np.asarray(s["log_tau"])[:, None]])  # fmt: skip
    return theta, div


def score_divergence(lp_g, mu, cov, base):
    """E_q || C^{1/2} (grad log q - grad log p) ||^2 with q = N(mu, C)."""
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mu)))
    xs = mu + base @ chol.T
    g_p = np.concatenate([np.asarray(lp_g(xs[a : a + 500])) for a in range(0, len(xs), 500)])
    g_q = -np.linalg.solve(cov, (xs - mu).T).T
    diff = g_q - g_p
    return float(np.mean(np.sum((diff @ chol) ** 2, axis=1)))


def main(inp, mine, out):
    data, my = np.load(inp), np.load(mine)
    x, r, g = data["x2"], data["r2"], data["g2"]
    print(f"depth {float(data['depths'][2]):.1f} m, n = {len(r)}", flush=True)
    lp_nc = log_post_fn(x, r, g, False)
    lp_c = log_post_fn(x, r, g, True)
    lpg_nc = jax.jit(jax.vmap(jax.grad(lp_nc)))
    lpg_c = jax.jit(jax.vmap(jax.grad(lp_c)))
    d = x.shape[1] * (2 + N_REG) + 1

    theta, div = nuts_draws(x, g, r)
    mu_n, cov_n = theta.mean(0), np.cov(theta.T)
    print(f"NUTS: log tau {mu_n[-1]:.2f} ± {np.sqrt(cov_n[-1, -1]):.3f}, divergences {div}",
          flush=True)  # fmt: skip

    fits = {"nuts_matched": (mu_n, cov_n), "mine_laplace": (my["mu_l"], my["cov_l"]),
            "mine_bam_nc_100": (my["mu_nc"], my["cov_nc"])}  # fmt: skip
    starts = {"laplace": (my["mu_l"], my["cov_l"]), "nuts": (mu_n, cov_n),
              "standard": (np.zeros(d), np.eye(d))}  # fmt: skip
    for start, (m0, c0) in starts.items():
        for niter in (100, 1000):
            bam = BaM(D=d, lp=jax.jit(lp_nc), lp_g=lpg_nc)
            mu, cov = bam.fit(random.PRNGKey(11), regf=Regularizers().linear(10.0), mean=m0,
                              cov=c0, batch_size=64, niter=niter, verbose=False)  # fmt: skip
            mu, cov = np.asarray(mu), np.asarray(cov)
            fits[f"gsmvi_nc_{start}_{niter}"] = (mu, cov)
            print(f"official BaM non-centred, start {start}, {niter} it: log tau {mu[-1]:.2f} ± "
                  f"{np.sqrt(cov[-1, -1]):.3f}", flush=True)  # fmt: skip
    # centred, from the Laplace fit mapped to centred coordinates (first order)
    p = x.shape[1]
    lo, hi = 2 * p, 2 * p + N_REG * p
    m0, c0 = my["mu_l"].copy(), my["cov_l"]
    tau = float(np.exp(m0[-1]))
    jac = np.eye(d)
    jac[lo:hi, lo:hi] *= tau
    jac[lo:hi, -1] = tau * m0[lo:hi]
    m0[lo:hi] *= tau
    c0 = jac @ c0 @ jac.T
    for niter in (100, 1000):
        bam = BaM(D=d, lp=jax.jit(lp_c), lp_g=lpg_c)
        c0s = (c0 + c0.T) / 2
        mu, cov = bam.fit(random.PRNGKey(11), regf=Regularizers().linear(10.0), mean=m0,
                          cov=c0s, batch_size=64, niter=niter, verbose=False)  # fmt: skip
        mu, cov = np.asarray(mu), np.asarray(cov)
        print(f"official BaM centred, start laplace, {niter} it: log tau {mu[-1]:.2f} ± "
              f"{np.sqrt(cov[-1, -1]):.3f}", flush=True)  # fmt: skip
    np.savez(out, nuts_theta=theta, **{f"mu_{k}": v[0] for k, v in fits.items()},
             **{f"cov_{k}": v[1] for k, v in fits.items()})  # fmt: skip

    base = np.random.default_rng(0).standard_normal((20000, d))
    print("\nScore-based divergence D(q; p) in the non-centred coordinates (lower = BaM prefers):")
    res = {}
    for name, (mu, cov) in fits.items():
        dv = score_divergence(lpg_nc, mu, (cov + cov.T) / 2, base)
        res[name] = dv
        print(f"  {name:28s} log tau {mu[-1]:6.2f} ± {np.sqrt(cov[-1, -1]):.3f}   D = {dv:.4g}",
              flush=True)  # fmt: skip
    np.savez(out, nuts_theta=theta, **{f"div_{k}": v for k, v in res.items()},
             **{f"mu_{k}": v[0] for k, v in fits.items()},
             **{f"cov_{k}": v[1] for k, v in fits.items()})  # fmt: skip


if __name__ == "__main__":
    main(*sys.argv[1:4])
