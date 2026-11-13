"""NUTS reference posterior for the hierarchical last layer (docs/uncertainty/HIERARCHY.md).

numpyro is not a dependency of oceanscope_ml: run this in a separate environment with
``pip install numpyro``. Input is ``reference_inputs.npz`` from
``python -m oceanscope_ml.evaluation.uncertainty_hier reference-inputs``; output is the draws
(log tau, delta_BoB on the constant column) plus divergences and R-hat, non-centred and centred.

    python scripts/hier_nuts.py <reference_inputs.npz> <nuts_draws.npz>
"""

import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from jax import random  # noqa: E402
from numpyro.diagnostics import split_gelman_rubin  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

numpyro.set_host_device_count(4)
PRIOR_SD, LOG_TAU_MEAN, LOG_TAU_SD, N_REG = 0.5, float(np.log(0.2)), 1.0, 3


def model(x, g, r, centred):
    p = x.shape[1]
    beta = numpyro.sample("beta", dist.Normal(0, PRIOR_SD).expand([p]))
    gamma = numpyro.sample("gamma", dist.Normal(0, PRIOR_SD).expand([p]))
    s = numpyro.sample("log_tau", dist.Normal(LOG_TAU_MEAN, LOG_TAU_SD))
    if centred:
        delta = numpyro.sample("delta", dist.Normal(0, jnp.exp(s)).expand([N_REG, p]))
    else:
        eta = numpyro.sample("eta", dist.Normal(0, 1).expand([N_REG, p]))
        delta = numpyro.deterministic("delta", jnp.exp(s) * eta)
    log_sd = jnp.sum(x * (gamma[None, :] + delta[g]), axis=1)
    numpyro.sample("r", dist.Normal(x @ beta, jnp.exp(log_sd)), obs=r)


def run(x, g, r, centred, seed):
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=1000, num_samples=1000,
                num_chains=4, progress_bar=False)  # fmt: skip
    mcmc.run(random.PRNGKey(seed), jnp.asarray(x), jnp.asarray(g), jnp.asarray(r), centred,
             extra_fields=("diverging",))  # fmt: skip
    by_chain = mcmc.get_samples(group_by_chain=True)
    rhat = max(float(np.max(np.asarray(split_gelman_rubin(np.asarray(v)))))
               for k, v in by_chain.items() if k != "delta" or centred)  # fmt: skip
    smp = mcmc.get_samples()
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    s = np.asarray(smp["log_tau"])
    d = np.asarray(smp["delta"][:, 0, -1])
    return s, d, div, rhat


def main(inp, out):
    data = np.load(inp)
    res = {}
    for k, depth in enumerate(data["depths"]):
        x, g, r = data[f"x{k}"], data[f"g{k}"], data[f"r{k}"]
        s, d, div_nc, rhat = run(x, g, r, False, k)
        _, _, div_c, rhat_c = run(x, g, r, True, 100 + k)
        print(f"{depth:.1f} m: n={len(r)}, log tau {s.mean():.2f} ± {s.std():.2f}, "
              f"delta_BoB sd {d.std():.3f}, divergences non-centred {div_nc}, centred {div_c}, "
              f"R-hat non-centred {rhat:.3f}, centred {rhat_c:.3f}", flush=True)  # fmt: skip
        res.update({f"log_tau{k}": s, f"delta_bob{k}": d, f"div_nc{k}": div_nc,
                    f"div_c{k}": div_c, f"rhat{k}": rhat, f"rhat_c{k}": rhat_c})  # fmt: skip
    np.savez(out, **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
