import numpy as np
from scipy.special import gammaln
import pandas as pd
import statsmodels.api as sm

# --- helper: draw one slope per (source, outcome) from the source's mixture ---
def sample_beta_so(rng, pi_s, beta_mean, beta_sds):
    """
    Draw a single outcome-level slope β_{s,o} from a mixture of Normals:
        z ~ Categorical(pi_s),  β_{s,o} ~ Normal(beta_mean[z], beta_sds[z])
    """
    k = len(beta_mean)
    z = rng.choice(k, p=pi_s)
    return float(rng.normal(beta_mean[z], beta_sds[z]))

# Simulate profile likelihoods for a Poisson GLM
def simulate_profile_likelihoods_poisson(
    K=5, S=8, O=8, n_obs=100,
    *, beta_mean=None, beta_sds=None, true_pis=None, grid=None, seed=0
):
    rng = np.random.default_rng(seed)
    if beta_mean is None:
        beta_mean = np.linspace(-1, 1, K)
    if beta_sds is None:
        beta_sds = np.full(K, 0.15)
    beta_mean = np.asarray(beta_mean, float)
    beta_sds  = np.asarray(beta_sds,  float)
    if grid is None:
        raise ValueError("Provide grid")
    grid = np.asarray(grid, float)
    if true_pis is None:
        true_pis = rng.dirichlet(np.ones(K) * 2.0, size=S)

    # Hold (x,y) per source so we can fit one GLM per source later
    data_by_source = {s: {"x": [], "y": []} for s in range(S)}

    recs, beta_recs = [], []
    for s in range(S):
        pi_s = true_pis[s]
        for o in range(O):
            # 1) one outcome-level slope from the mixture
            beta_so = sample_beta_so(rng, pi_s, beta_mean, beta_sds)

            # 2) simulate covariates and responses
            x_i  = rng.uniform(-1,  1, n_obs)
            lam  = np.exp(beta_so * x_i)
            y_i  = rng.poisson(lam)

            # record for later GLM
            data_by_source[s]["x"].append(x_i)
            data_by_source[s]["y"].append(y_i)

            # 3) profile log-likelihood over the grid
            for b in grid:
                logp = y_i * (b * x_i) - np.exp(b * x_i) - gammaln(y_i + 1)
                recs.append({
                    "source":  s + 1,
                    "outcome": o + 1,
                    "point":   b,
                    "value":   logp.sum()
                })

            # 4) store the true (per-outcome) β
            beta_recs.append({
                "source":  s + 1,
                "outcome": o + 1,
                "beta_true": beta_so
            })

    # assemble simulation outputs
    df_sim  = pd.DataFrame.from_records(recs)
    beta_df = pd.DataFrame.from_records(beta_recs)

    # 5) fit one Poisson GLM per source (pooling across outcomes)
    # 4) fit one Poisson GLM per source (pooling across outcomes) — NO INTERCEPT
    est_recs = []
    for s in range(S):
        x_all = np.concatenate(data_by_source[s]["x"])
        y_all = np.concatenate(data_by_source[s]["y"])
        X     = x_all[:, None]                    # <- no constant
        res   = sm.GLM(y_all, X, family=sm.families.Poisson()).fit()
        est_recs.append({"source": s + 1, "beta_est": res.params[0]})
    est_df = pd.DataFrame.from_records(est_recs)

    return df_sim, true_pis, beta_df, est_df


# Simulate profile likelihoods for a linear regression
def simulate_profile_likelihoods_linreg(
    K: int = 5,
    S: int = 8,
    O: int = 8,
    n_obs: int = 100,
    *,
    beta_mean: np.ndarray | None = None,
    beta_sds:  np.ndarray | None = None,
    true_pis:  np.ndarray | None = None,
    grid: np.ndarray | None = None,
    sigma_noise: float = 1.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # 1) component means / sds
    if beta_mean is None:
        beta_mean = np.linspace(-1, 1, K)
    if beta_sds is None:
        beta_sds = np.full(K, 0.15)
    beta_mean = np.asarray(beta_mean, dtype=float)
    beta_sds  = np.asarray(beta_sds,  dtype=float)
    if grid is None:
        raise ValueError("Supply a 1-D array of grid points via `grid`.")
    grid = np.asarray(grid, dtype=float)

    # 2) Dirichlet weights per source
    if true_pis is None:
        true_pis = rng.dirichlet(np.full(K, 2.0), size=S)

    # collect data per source for OLS
    data_by_source = {s: {"x": [], "y": []} for s in range(S)}

    const_term = -0.5 * n_obs * np.log(2 * np.pi * sigma_noise**2)
    recs, beta_recs = [], []

    for s in range(S):
        pi_s = true_pis[s]
        for o in range(O):
            # one outcome-level slope from the mixture
            beta_so = sample_beta_so(rng, pi_s, beta_mean, beta_sds)

            # simulate
            x_i = rng.uniform(-1, 1, size=n_obs)
            y_i = beta_so * x_i + rng.normal(0.0, sigma_noise, size=n_obs)

            data_by_source[s]["x"].append(x_i)
            data_by_source[s]["y"].append(y_i)

            for b in grid:
                resid_sq = (y_i - b * x_i) ** 2
                ll = const_term - 0.5 * resid_sq.sum() / sigma_noise**2
                recs.append({
                    "source":  s + 1,
                    "outcome": o + 1,
                    "point":   b,
                    "value":   ll
                })

            beta_recs.append({
                "source":  s + 1,
                "outcome": o + 1,
                "beta_true": beta_so
            })

    df_sim  = pd.DataFrame.from_records(recs)
    beta_df = pd.DataFrame.from_records(beta_recs)

    # OLS per source
    # OLS per source — NO INTERCEPT
    est_recs = []
    for s in range(S):
        x_all = np.concatenate(data_by_source[s]["x"])
        y_all = np.concatenate(data_by_source[s]["y"])
        X     = x_all[:, None]                    # <- no constant
        ols   = sm.OLS(y_all, X).fit()
        est_recs.append({"source": s + 1, "beta_est": ols.params[0]})
    est_df = pd.DataFrame.from_records(est_recs)


    return df_sim, true_pis, beta_df, est_df


# Simulate profile likelihoods for a logistic regression
def simulate_profile_likelihoods_logistic(
    K: int = 5,
    S: int = 8,
    O: int = 8,
    n_obs: int = 100,
    *,
    beta_mean: np.ndarray | None = None,
    beta_sds:  np.ndarray | None = None,
    true_pis:  np.ndarray | None = None,
    grid:      np.ndarray | None = None,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # 1) component means / sds
    if beta_mean is None:
        beta_mean = np.linspace(-1, 1, K)
    if beta_sds is None:
        beta_sds = np.full(K, 0.15)
    beta_mean = np.asarray(beta_mean, dtype=float)
    beta_sds  = np.asarray(beta_sds,  dtype=float)
    if grid is None:
        raise ValueError("Provide a 1-D array of grid points via `grid`")
    grid = np.asarray(grid, dtype=float)

    # 2) Dirichlet weights per source
    if true_pis is None:
        true_pis = rng.dirichlet(np.ones(K) * 2.0, size=S)

    # collect data per source for logistic fit
    data_by_source = {s: {"x": [], "y": []} for s in range(S)}

    recs, beta_recs = [], []

    for s in range(S):
        pi_s = true_pis[s]
        for o in range(O):
            # one outcome-level slope from the mixture
            beta_so = sample_beta_so(rng, pi_s, beta_mean, beta_sds)

            # simulate covariates & Bernoulli responses
            x_i     = rng.uniform(-1, 1, size=n_obs)
            logit_p = beta_so * x_i
            p_i     = 1.0 / (1.0 + np.exp(-logit_p))
            y_i     = rng.binomial(1, p_i)

            data_by_source[s]["x"].append(x_i)
            data_by_source[s]["y"].append(y_i)

            for b in grid:
                bx = b * x_i
                ll = (y_i * bx - np.log1p(np.exp(bx))).sum()
                recs.append({
                    "source":  s + 1,
                    "outcome": o + 1,
                    "point":   b,
                    "value":   ll
                })

            beta_recs.append({
                "source":  s + 1,
                "outcome": o + 1,
                "beta_true": beta_so
            })

    df_sim  = pd.DataFrame.from_records(recs)
    beta_df = pd.DataFrame.from_records(beta_recs)

    # logistic regression per source — NO INTERCEPT
    est_recs = []
    for s in range(S):
        x_all = np.concatenate(data_by_source[s]["x"])
        y_all = np.concatenate(data_by_source[s]["y"])
        X     = x_all[:, None]                    # <- no constant
        logit = sm.GLM(y_all, X, family=sm.families.Binomial()).fit()
        est_recs.append({"source": s + 1, "beta_est": logit.params[0]})
    est_df = pd.DataFrame.from_records(est_recs)


    return df_sim, true_pis, beta_df, est_df
