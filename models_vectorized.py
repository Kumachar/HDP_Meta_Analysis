import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def stick_breaking(beta):
    """
    Beta shape (..., k). Returns (..., k).
    """
    cumprod = jnp.cumprod(1.0 - beta, axis=-1)                 # (..., k)
    ones = jnp.ones_like(cumprod[..., :1])                     # (..., 1)
    portion_remaining = jnp.concatenate([ones, cumprod[..., :-1]], axis=-1)  # (..., k)
    return beta * portion_remaining

def _normalize_last_axis(x, eps=1e-12):
    return x / (x.sum(axis=-1, keepdims=True) + eps)

# --- keep this for backward-compat (scalar beta per source) ---
def custom_loglike_interp(beta_scalar, x_pack, ll_pack):
    """
    For one SOURCE: x_pack, ll_pack are (M, Lmax).
    Interpolate at a SINGLE beta across all M outcomes and sum.
    """
    def one_outcome(x_row, y_row):
        return jnp.interp(beta_scalar, x_row, y_row)
    return jax.vmap(one_outcome)(x_pack, ll_pack).sum()

# --- NEW: vector beta per outcome ---
def custom_loglike_interp_multi(beta_vec, x_pack, ll_pack, m_valid):
    """
    For one SOURCE:
      beta_vec: (M_max,)   per-outcome betas
      x_pack:   (M_max, Lmax)
      ll_pack:  (M_max, Lmax)
      m_valid:  scalar (# of real outcomes for this source; rest are padded)
    Returns scalar summed profile log-likelihood over valid outcomes.
    """
    def one_outcome(beta, x_row, y_row):
        return jnp.interp(beta, x_row, y_row)  # outside-range is clamped like np.interp
    vals = jax.vmap(one_outcome)(beta_vec, x_pack, ll_pack)  # (M_max,)
    mask = (jnp.arange(vals.shape[0]) < m_valid).astype(vals.dtype)
    return jnp.sum(vals * mask)

def HDP_model_vectorized(x_padded, ll_padded, lengths, N_sources: int, k: int):
    """
    x_padded:  (N_sources, M_max, Lmax)
    ll_padded: (N_sources, M_max, Lmax)
    lengths:   (N_sources,) number of valid outcomes per source (<= M_max)
    """
    # 1) Top-level
    gamma  = numpyro.sample("gamma",  dist.Gamma(1.0, 5.0))
    alpha0 = numpyro.sample("alpha0", dist.Gamma(1.0, 5.0))

    # 2) Global sticks (k,)
    beta_tilt = numpyro.sample("beta_tilt", dist.Beta(1.0, gamma).expand([k]))
    beta      = stick_breaking(beta_tilt)             # (k,)
    numpyro.deterministic("beta", beta)

    # 3) Source-specific sticks
    denom = jnp.clip(1.0 - jnp.cumsum(beta, axis=-1), 1e-12, None)   # (k,)
    pi_tilt = numpyro.sample(
        "pi_tilt",
        dist.Beta(alpha0 * beta, alpha0 * denom).expand([N_sources, k])
    )                                                                # (N_sources, k)
    pi      = stick_breaking(pi_tilt)                                # (N_sources, k)
    pi_norm = _normalize_last_axis(pi)                               # (N_sources, k)
    numpyro.deterministic("pi_norm", pi_norm)

    # 4) Shared component params
    mu    = numpyro.sample("mu",    dist.Normal(0.0, 10.0).expand([k]))   # (k,)
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0).expand([k]))    # (k,)

    comp = dist.Normal(
        loc=jnp.broadcast_to(mu,     (N_sources, k)),
        scale=jnp.broadcast_to(sigma, (N_sources, k)),
    )
    mix = dist.MixtureSameFamily(dist.Categorical(probs=pi_norm), comp)  # batch (N_sources,)

    # 5) Per-SOURCE beta_s (kept for compatibility; not used in likelihood)
    beta_s = numpyro.sample("beta_s", mix)  # (N_sources,)
    numpyro.deterministic("beta_by_source", beta_s)
    for s in range(int(N_sources)):
        numpyro.deterministic(f"beta_{s + 1}", beta_s[s])

    # 6) Per-OUTCOME betas beta_{s,o} used for the profile likelihood
    M_max = x_padded.shape[1]
    # sample_shape puts M_max in front; swap to (N_sources, M_max)
    beta_so = numpyro.sample("beta_s_o", mix, sample_shape=(M_max,))      # (M_max, N_sources)
    beta_so = jnp.swapaxes(beta_so, 0, 1)                                  # (N_sources, M_max)

    # (Optional but typically desired): name each beta_{s,o} for plotting
    for s in range(int(N_sources)):
        for o in range(int(M_max)):
            numpyro.deterministic(f"beta_{s + 1}_{o + 1}", beta_so[s, o])

    # 7) Profile likelihood with per-outcome betas, masking padded outcomes
    per_source_ll = jax.vmap(custom_loglike_interp_multi)(
        beta_so, x_padded, ll_padded, lengths
    )  # (N_sources,)
    numpyro.factor("profile_ll_sum", per_source_ll.sum())
