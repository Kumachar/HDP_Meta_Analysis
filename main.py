from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from typing import Sequence

# --- Your local modules ---
from simulation import (
    simulate_profile_likelihoods_poisson,
    simulate_profile_likelihoods_linreg,
    simulate_profile_likelihoods_logistic,
)
# add at top with other imports
from utils import (
    save_beta_density_hist,
    store_beta_posteriors,
    compute_g0_components,
    plot_g0_density_from_file,
    kl_prior_posterior_beta,
    summarise_beta_prior_posterior,
    largest_mean_pi_norm,
    true_source_moments,
    true_source_median_mode,
    compute_mu_s_hat,            # NEW
    outcome_error_summary,       # NEW
    save_beta_trace_plots,       # NEW
)

from plot_results import (
    generate_summary_plots,
    plot_and_save_g0_by_rep,
    plot_g0_and_beta_single_axis,
    plot_true_vs_posterior_sources,  # 🔧 NEW
)
# fallback to non-vectorized model if user doesn't select the vectorized one
from models import HDP_model


# -----------------------
# Multi-CPU configuration
# -----------------------
def setup_multi_cpu(n_threads: int | None, prefer_parallel: bool, num_chains: int,
                    use_xla_flags: bool = False):
    """
    Configure JAX/NumPyro for multi-CPU use.
    - On GL, leave use_xla_flags=False (their XLA doesn't recognize legacy flags).
    """
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    if n_threads:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        if use_xla_flags:
            # Only enable on systems where these flags are supported
            os.environ["XLA_FLAGS"] = (
                f"--xla_cpu_multi_thread_eigen=true "
                f"--xla_cpu_multi_thread_eigen_thread_count={n_threads}"
            )

    import numpyro, jax
    chain_method = "vectorized"
    if prefer_parallel:
        numpyro.set_host_device_count(num_chains)
        if jax.local_device_count() >= num_chains:
            chain_method = "parallel"
    return chain_method

    import numpyro, jax
    chain_method = "vectorized"
    if prefer_parallel:
        numpyro.set_host_device_count(num_chains)
        if jax.local_device_count() >= num_chains:
            chain_method = "parallel"
    return chain_method


def _expand_vectorized_betas_to_legacy_vars(idata):
    import re, xarray as xr
    post = idata.posterior
    # If the model already wrote beta_{s,o}, leave them alone.
    if any(re.fullmatch(r"beta_\d+_\d+", v) for v in post.data_vars):
        return idata

    var = "beta_s" if "beta_s" in post.data_vars else ("beta_by_source" if "beta_by_source" in post.data_vars else None)
    if var is None:
        return idata

    da = post[var]
    src_dim = next((d for d in da.dims if d not in ("chain","draw")), None)
    if src_dim is None: return idata
    new_vars = {f"beta_{i+1}_s": da.isel({src_dim: i}) for i in range(da.sizes[src_dim])}
    idata.posterior = post.assign(**new_vars)
    return idata


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")

def _load_idata(path: Path) -> az.InferenceData:
    # You likely already have a loader; keep it if so.
    return az.from_netcdf(path)


def run_hdp_experiments(
    K: int,
    O: int,
    S: int,
    n_obs: int,
    N_sources: int,
    seed: int,
    output_base: str,
    *,
    num_warmup: int = 5000,
    num_samples: int = 20000,
    n_reps: int = 10,
    model_types: Sequence[str] | None = None,
    rep_override: Sequence[int] | None = None,
    use_vectorized_model: bool = True,
    num_chains: int = 4,
    n_threads: int | None = None,
    parallel_chains: bool = False,
    # NEW:
    outcome_cap: int | None = None,
    O_sim: int | None = None,
):
    """
    Simulate with O_sim (if given) then run the model using only the first
    `outcome_cap` outcomes per source (if given). Results are saved exactly
    for the outcomes present in df_sim after capping.
    """
    # 0) CPU + chain method
    chain_method = setup_multi_cpu(
        n_threads=n_threads, prefer_parallel=parallel_chains, num_chains=num_chains
    )

    import jax
    import jax.numpy as jnp
    from numpyro.infer import NUTS, MCMC

    if use_vectorized_model:
        try:
            from models_vectorized import HDP_model_vectorized as _model
        except Exception as e:
            raise ImportError(
                "use_vectorized_model=True but could not import HDP_model_vectorized "
                "from models_vectorized.py"
            ) from e
        # packer (define fallback if not exported)
        try:
            from models_vectorized import pack_sources
        except Exception:
            # local fallback
            def pack_sources(source_outcome_data_dict, M_outcomes=None):
                keys = sorted(source_outcome_data_dict.keys())
                lists = [source_outcome_data_dict[k] for k in keys]
                N = len(lists)
                Mi_list = [len(lst) for lst in lists]
                M = M_outcomes or max(Mi_list)
                Lmax = max(a.shape[0] for lst in lists for a in lst)

                def pad_row(a, L):
                    x = a[:, 0];
                    y = a[:, 1]
                    pad_len = L - a.shape[0]
                    if pad_len > 0:
                        tail_x = x[-1] + jnp.arange(1, pad_len + 1, dtype=x.dtype) * (1e-6)
                        x_pad = jnp.concatenate([x, tail_x], axis=0)
                        y_pad = jnp.pad(y, (0, pad_len), constant_values=y[-1])
                    else:
                        x_pad, y_pad = x, y
                    return x_pad, y_pad, a.shape[0]

                xs, ys, lens = [], [], []
                for lst in lists:
                    rows = list(lst[:M]) + [jnp.array([[0., 0.]], dtype=jnp.float32)] * max(0, M - len(lst))
                    x_rows, y_rows, l_rows = zip(*[pad_row(a, Lmax) for a in rows])
                    xs.append(jnp.stack(x_rows))
                    ys.append(jnp.stack(y_rows))
                    lens.append(jnp.array(l_rows))
                x_padded = jnp.stack(xs)
                ll_padded = jnp.stack(ys)
                lengths = jnp.stack(lens)
                return x_padded, ll_padded, lengths
    else:
        _model = HDP_model  # original non-vectorized model

    rng_master = np.random.default_rng(seed)

    # available simulators
    all_sims = dict(
        linear=simulate_profile_likelihoods_linreg,
        poisson=simulate_profile_likelihoods_poisson,
        logistic=simulate_profile_likelihoods_logistic,
    )
    sim_fns = all_sims if model_types is None else {
        m: all_sims[m] for m in model_types if m in all_sims
    }

    # --- NEW: decide how many outcomes to SIMULATE vs ANALYZE ---
    O_sim_eff = int(O_sim) if O_sim is not None else int(O)
    if outcome_cap is not None and outcome_cap > O_sim_eff:
        raise ValueError(f"--outcome-cap={outcome_cap} exceeds --O-sim={O_sim_eff}")

    for model_type, sim_fn in sim_fns.items():
        model_dir = Path(output_base) / model_type
        fig_dir   = model_dir / "figures" / "beta_plots"
        data_dir  = model_dir / "data"
        fig_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        stats_all = []
        kl_all = []
        reps = rep_override if rep_override is not None else list(range(1, n_reps + 1))

        for rep in reps:
            # 2) per-rep seeding — stable across O values
            rep_seed = int(np.random.SeedSequence([seed, rep]).generate_state(1, dtype=np.uint32)[0])
            rng_rep  = np.random.default_rng(rep_seed)

            # 3) simulate WITH O_sim_eff outcomes
            grid_pts  = np.linspace(-10, 10, 20_000)
            beta_mean = rng_rep.normal(loc=1.5, scale=0.7, size=K)
            beta_sds  = np.abs(rng_rep.normal(loc=0.0, scale=1.0, size=K))
            sim_kwargs = dict(
                K=K, S=S, O=O_sim_eff, n_obs=n_obs,
                beta_mean=beta_mean, beta_sds=beta_sds,
                grid=grid_pts, seed=rep_seed
            )
            if model_type == "linear":
                sim_kwargs["sigma_noise"] = 0.8

            df_sim, pis_true, df_betas, est_df = sim_fn(**sim_kwargs)

            # save "truth" once per rep (shared by 20 vs 50 runs)
            rep_dir = data_dir / f"rep{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            np.savez(rep_dir / "truth_source_mixtures.npz",
                     beta_mean=beta_mean, beta_sds=beta_sds, pis=pis_true)

            # 4) OPTIONAL CAP — use only the first outcome_cap outcomes downstream
            if outcome_cap is not None:
                df_sim   = df_sim[df_sim["outcome"] <= outcome_cap].copy()
                df_betas = df_betas[df_betas["outcome"] <= outcome_cap].copy()

            # 5) build per-source outcome lists from (possibly capped) df_sim
            source_outcome_data: dict[int, list] = {}
            for s in range(1, N_sources + 1):
                df_s = df_sim[df_sim["source"] == s]
                lst = []
                for o in sorted(df_s["outcome"].unique()):
                    arr = df_s[df_s["outcome"] == o][["point", "value"]].to_numpy()
                    lst.append(jnp.array(arr))
                source_outcome_data[s] = lst

            # 6) MCMC
            rng_key = jax.random.PRNGKey(rep_seed)
            kernel = NUTS(_model, dense_mass=True, target_accept_prob=0.90, max_tree_depth=12)
            mcmc = MCMC(kernel,
                        num_warmup=num_warmup,
                        num_samples=num_samples,
                        num_chains=num_chains,
                        chain_method=chain_method)

            if use_vectorized_model:
                # pack exactly up to the outcomes we're analyzing
                M_pack = int(outcome_cap) if outcome_cap is not None else int(O)
                x_padded, ll_padded, lengths = pack_sources(source_outcome_data, M_outcomes=M_pack)
                mcmc.run(
                    rng_key,
                    x_padded, ll_padded, lengths,
                    N_sources=N_sources,
                    k=K,
                )
            else:
                num_outcomes_dict = {s: len(source_outcome_data[s]) for s in source_outcome_data}
                data_point_mean   = df_sim["point"].mean()
                mcmc.run(
                    rng_key,
                    source_outcome_data=source_outcome_data,
                    num_outcomes_dict=num_outcomes_dict,
                    N_sources=N_sources,
                    k=K,
                    data_point_mean=data_point_mean,
                )

            idata = az.from_numpyro(mcmc)
            if use_vectorized_model:
                # ensure per-(s,o) deterministic names exist where available
                idata = _expand_vectorized_betas_to_legacy_vars(idata)

            # 7) store posteriors & plots (per-outcome saved based on df_sim content)
            store_beta_posteriors(
                idata=idata,
                df_sim=df_sim,           # << filtering here controls which (s,o) are saved
                beta_df=df_betas,
                output_folder=data_dir / f"rep{rep}",
                file_ext="npz",
                compress=True,
            )
            save_beta_density_hist(
                idata,
                df_sim=df_sim,
                output_folder=str(fig_dir),
                experiment=f"rep{rep}",
                per_outcome=True
            )

            # …(unchanged) KL and summary with true mixture stats…
            true_means, _true_vars = true_source_moments(pis_true, beta_mean, beta_sds)
            true_medians, true_modes = true_source_median_mode(pis_true, beta_mean, beta_sds)
            kl_vals = kl_prior_posterior_beta(idata, beta_df=df_betas)

            stats = summarise_beta_prior_posterior(
                idata, df_betas,
                kl_dict=kl_vals, glm_est_df=est_df,
                experiment_label=f"rep{rep}",
                true_source_means=true_means,
                true_source_medians=true_medians,
                true_source_modes=true_modes,
            )
            stats["rep"] = rep
            (stats).to_csv((data_dir / f"rep{rep}" / "beta_summary_stats.csv"), index=False)

            # G0 components & figure (unchanged)
            g0_path = data_dir / f"rep{rep}" / "g0_components.npz"
            compute_g0_components(idata, output_path=str(g0_path), xgrid_len=2000)
            g0_fig = fig_dir / f"rep{rep}_g0.png"
            plot_g0_density_from_file(npz_path=str(g0_path), density_cut=1e-2, save_to=str(g0_fig))

            print(f"[{model_type}] rep={rep} done.")

        # write family-level summary
        summary = pd.concat(stats_all or [pd.read_csv(p) for p in (data_dir).glob("rep*/beta_summary_stats.csv")],
                            ignore_index=True)
        summary.to_csv(model_dir / "beta_summary_stats.csv", index=False)
        print(f"Completed '{model_type}' models. Results in {model_dir}\n")



def main():
    parser = argparse.ArgumentParser(
        prog="simexp",
        description="Run HDP experiments or plot summaries"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run-hdp
    run_p = sub.add_parser("run-hdp", help="Run HDP experiments")
    run_p.add_argument("--K", type=int, required=True)
    run_p.add_argument("--O", type=int, required=True)
    run_p.add_argument("--S", type=int, required=True)
    run_p.add_argument("--n-obs", type=int, required=True)
    run_p.add_argument("--N-sources", type=int, required=True)
    run_p.add_argument("--seed", type=int, required=True)
    run_p.add_argument("--output-base", type=Path)
    run_p.add_argument("--num-warmup", type=int, default=5000)
    run_p.add_argument("--num-samples", type=int, default=20000)
    run_p.add_argument("--n-reps", type=int, default=10)
    run_p.add_argument("--model-type", "-m", choices=["linear", "poisson", "logistic"], nargs="+",
                       help="Which model families to run")
    run_p.add_argument("--rep", "-r", type=int, nargs="+",
                       help="Which repetition indices to run")

    # NEW: vectorization & CPU parallelism flags
    run_p.add_argument("--use-vectorized-model", action="store_true",
                       help="Use models_vectorized.HDP_model_vectorized with packed tensors.")
    run_p.add_argument("--num-chains", type=int, default=4, help="Number of MCMC chains.")
    run_p.add_argument("--n-threads", type=int, default=None,
                       help="OMP/XLA Eigen threads per process (CPU).")
    run_p.add_argument("--parallel-chains", action="store_true",
                       help="Try chain_method=parallel (one chain per logical CPU device).")
    run_p.add_argument("--outcome-cap", type=int, default=None,
                       help="Analyze only the first M outcomes per source (cap after simulation).")
    run_p.add_argument("--O-sim", type=int, default=None,
                       help="Override O for simulation only (e.g., simulate 50, then cap to 20).")

    # plot-summary (unchanged)
    plot_p = sub.add_parser("plot-summary", help="Plot summary from results")
    plot_p.add_argument("--results-root", type=Path, required=True)
    plot_p.add_argument("--families", nargs="+", default=["linear", "logistic"])

    args = parser.parse_args()
    if args.command == "run-hdp":
        if args.output_base is None:
            today = date.today().strftime("%Y-%m-%d")
            args.output_base = Path(f"results_{today}")

        run_hdp_experiments(
            K=args.K, O=args.O, S=args.S, n_obs=args.n_obs,
            N_sources=args.N_sources, seed=args.seed,
            output_base=str(args.output_base),
            num_warmup=args.num_warmup, num_samples=args.num_samples,
            n_reps=args.n_reps,
            model_types=args.model_type,
            rep_override=args.rep,
            use_vectorized_model=args.use_vectorized_model,
            num_chains=args.num_chains,
            n_threads=args.n_threads,
            parallel_chains=args.parallel_chains,
            # NEW:
            outcome_cap=args.outcome_cap,
            O_sim=args.O_sim,
        )


if __name__ == "__main__":
    main()


