import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def stick_breaking(beta):
    portion_remaining = jnp.concatenate(
        [jnp.array([1.]), jnp.cumprod(1. - beta)[:-1]]
    )
    return beta * portion_remaining


def reparameterize(pi):
    return pi / pi.sum()

def custom_loglike(beta, outcome_data_list):
    """
    Custom log-likelihood function for the hierarchical Dirichlet process model.
    This function computes the log-likelihood of the observed data given the
    parameters of the model.

    :param beta: The parameter representing the stick-breaking proportions.
    :param outcome_data_list: A list of numpy arrays, where each array contains
    :return:   The total log-likelihood computed by interpolating the log-likelihood
    """

    total_ll = 0.
    for outcome_data in outcome_data_list:
        x_vals = outcome_data[:,0]
        loglike_vals = outcome_data[:,1]
        total_ll += jnp.interp(beta, x_vals, loglike_vals)
    return total_ll


def HDP_model(source_outcome_data,
              num_outcomes_dict,
              N_sources,
              k,
              data_point_mean):
    # 1) Top-level concentrations
    gamma  = numpyro.sample("gamma",  dist.Gamma(1.0, 5.0))
    alpha0 = numpyro.sample("alpha0", dist.Gamma(1.0, 5.0))

    # 2) Global stick-breaking
    beta_tilt = numpyro.sample("beta_tilt",
                               dist.Beta(1.0, gamma).expand([k]))
    beta      = stick_breaking(beta_tilt)
    numpyro.deterministic("beta", beta)

    # 3) Source-specific sticks & normalization
    pi_tilt = numpyro.sample(
        "pi_tilt",
        dist.Beta(alpha0 * beta, alpha0 * (1.0 - jnp.cumsum(beta)))
            .expand([N_sources, k])
    )
    pi_norm = jax.vmap(lambda row: reparameterize(stick_breaking(row)))(
        pi_tilt
    )
    numpyro.deterministic("pi_norm", pi_norm)

    # 4) Shared component parameters
    mu    = numpyro.sample("mu",    dist.Normal(0.0, 10.0).expand([k]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0).expand([k]))

    # 5) For each source s and each of its outcomes o, draw β_{s,o}
    #    and score it with the matching data slice only.
    for s in range(1, N_sources + 1):
        outcomes = source_outcome_data[s]          # a list of arrays, one per outcome
        for o, outcome_data in enumerate(outcomes, start=1):
            β_so = numpyro.sample(
                f"beta_{s}_{o}",
                dist.MixtureSameFamily(
                    dist.Categorical(probs=pi_norm[s-1]),
                    dist.Normal(mu, sigma),
                )
            )
            # wrap the single outcome in a list so custom_loglike still works
            numpyro.factor(
                f"loglike_{s}_{o}",
                custom_loglike(β_so, [outcome_data])
            )