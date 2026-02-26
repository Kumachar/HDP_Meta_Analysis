from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.stats import norm
import arviz as az
import re

# ──────────────────────────────────────────────────────────────────────────────
# Helpers for β variable selection & naming
# ──────────────────────────────────────────────────────────────────────────────
_BETA_SRC_ONLY_RE = re.compile(r"^beta_(\d+)$")
_BETA_SRC_OUTCOME_RE = re.compile(r"^beta_(\d+)_(\d+)$")
_BETA_EITHER_RE = re.compile(r"^beta_(\d+)(?:_(\d+))?$")   # beta_<s> or beta_<s>_<o>

# ─────────────────────────────────────────────────────────────────────
# NEW: posterior sample mean of source mixture mean, μ_s_hat
# μ_s_hat = (1/n) * sum_t sum_j pi_norm[t,s,j] * mu[t,j]
# Returns DataFrame: ['source', 'mu_s_hat'] with 1-based source index.
# ─────────────────────────────────────────────────────────────────────
def true_source_moments(pis_true: np.ndarray,
                        beta_mean: np.ndarray,
                        beta_sds:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean, var) for each source's *true* mixture:
        mean_s = sum_j pi_sj * mu_j
        var_s  = sum_j pi_sj * (sigma_j^2 + mu_j^2) - mean_s^2
    """
    pis_true  = np.asarray(pis_true, float)    # (S, K)
    beta_mean = np.asarray(beta_mean, float)   # (K,)
    beta_sds  = np.asarray(beta_sds,  float)   # (K,)

    means = pis_true @ beta_mean                       # (S,)
    second_moment = pis_true @ (beta_sds**2 + beta_mean**2)
    vars_ = np.maximum(second_moment - means**2, 0.0)
    return means, vars_

def compute_mu_s_hat(idata: az.InferenceData) -> pd.DataFrame:
    post = idata.posterior
    if ("pi_norm" not in post) or ("mu" not in post):
        # graceful fallback (empty if not present)
        return pd.DataFrame({"source": [], "mu_s_hat": []})

    pi = post["pi_norm"].values    # (chain, draw, S, K)
    mu = post["mu"].values         # (chain, draw, K)

    mu_expanded = mu[:, :, None, :]               # (chain, draw, 1, K)
    mu_s_draws = (pi * mu_expanded).sum(axis=-1)  # (chain, draw, S)
    mu_s_hat = mu_s_draws.mean(axis=(0, 1))       # (S,)

    S = mu_s_hat.shape[0]
    return pd.DataFrame({
        "source": np.arange(1, S + 1, dtype=int),
        "mu_s_hat": mu_s_hat.astype(float)
    })


# ─────────────────────────────────────────────────────────────────────
# NEW: per-source mean error across outcomes
# outcome_error_mean_s = (1/O_s) * sum_o ( E[beta_{s,o}] - beta_true_{s,o} )
# Uses posterior mean of beta_{s,o}. If absent, falls back to per-source β_s.
# Expects beta_df with columns: source, outcome, beta_true
# Returns: ['source','O','outcome_error_mean']
# ─────────────────────────────────────────────────────────────────────
def outcome_error_summary(idata: az.InferenceData, beta_df: pd.DataFrame) -> pd.DataFrame:
    post = idata.posterior

    # Lookup: posterior means for explicit per-outcome variables beta_{s}_{o}
    post_means: Dict[Tuple[int, int], float] = {}
    varnames = [v for v in post.data_vars if re.fullmatch(r"beta_\d+_\d+", v)]
    for v in varnames:
        s, o = [int(x) for x in v.split("_")[1:3]]  # parse beta_{s}_{o}
        post_means[(s, o)] = float(post[v].values.ravel().mean())

    # Fallback if (s,o) is missing: use a per-source β_s draw getter you already have
    def _fallback_mean_for_source(s: int) -> float:
        # This assumes you already have this helper in your utils
        # If not, replace with your preferred per-source β mean lookup
        if "_get_posterior_draws_for_source" in globals():
            draws = _get_posterior_draws_for_source(idata, s)
            return float(np.asarray(draws).mean())
        return np.nan

    rows = []
    for s, grp in beta_df.groupby("source"):
        errs = []
        for _, r in grp.iterrows():
            o = int(r["outcome"])
            b_true = float(r["beta_true"])
            b_hat = post_means.get((s, o), _fallback_mean_for_source(s))
            if np.isfinite(b_hat):
                errs.append(b_hat - b_true)
        O = len(errs)
        outcome_err_mean = float(np.mean(errs)) if O > 0 else np.nan
        rows.append({"source": int(s), "O": O, "outcome_error_mean": outcome_err_mean})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# NEW: per-outcome trace plots for beta_{s,o}
# Saves to: {output_dir}/trace_beta_{s}_{o}.png
# ─────────────────────────────────────────────────────────────────────
def save_beta_trace_plots(
    idata: az.InferenceData,
    *,
    output_dir: str | Path,
    regex: str = r"beta_\d+_\d+",
    dpi: int = 220,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    post = idata.posterior
    varnames = [v for v in post.data_vars if re.fullmatch(regex, v)]
    count = 0
    for v in sorted(varnames, key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2]))):
        try:
            axes = az.plot_trace(idata, var_names=[v], compact=True, backend="matplotlib")
            # az returns a numpy array of axes; find figure
            if hasattr(axes, "ravel"):
                fig = axes.ravel()[0].figure
            else:
                fig = axes[0, 0].figure  # defensive
            fig.suptitle(f"Trace: {v}", fontsize=10)
            fig.tight_layout()
            fig.savefig(output_dir / f"trace_{v}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            count += 1
        except Exception:
            # be robust to any plot issues; continue
            continue
    print(f"[trace] Saved {count} trace plots → {output_dir.resolve()}")

def true_mixture_density(pi_s: np.ndarray,
                         beta_mean: np.ndarray,
                         beta_sds:  np.ndarray,
                         xgrid: np.ndarray) -> np.ndarray:
    """
    Density of sum_j pi_sj * N(mu_j, sigma_j^2) evaluated on xgrid.
    """
    pi_s     = np.asarray(pi_s, float).ravel()         # (K,)
    beta_mean= np.asarray(beta_mean, float).ravel()    # (K,)
    beta_sds = np.asarray(beta_sds,  float).ravel()    # (K,)
    xgrid    = np.asarray(xgrid, float).ravel()        # (G,)
    comp = (1.0 / (np.sqrt(2*np.pi) * beta_sds))[None, :] * \
           np.exp(-0.5 * ((xgrid[:, None] - beta_mean[None, :]) / beta_sds[None, :])**2)
    return (comp @ pi_s).ravel()

def true_source_median_mode(
    pis_true: np.ndarray,        # (S, K)
    beta_mean: np.ndarray,       # (K,)
    beta_sds: np.ndarray,        # (K,)
    *,
    grid_len: int = 4096,
    tol: float = 1e-8,
    max_iter: int = 80,
    pad_sigmas: float = 8.0,     # search bounds: [min(mu-8σ), max(mu+8σ)]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the TRUE per-source median and (global) mode of the mixture:
        f_s(x) = sum_j pi_{s,j} N(x | mu_j, sigma_j^2)
    Median: root of F_s(m) = 0.5 using bisection on CDF.
    Mode:   argmax_x f_s(x) on a dense grid (global mode).

    Returns:
      medians: (S,)
      modes:   (S,)
    """
    pis_true  = np.asarray(pis_true,  float)   # (S,K)
    mu        = np.asarray(beta_mean, float)   # (K,)
    sigma     = np.asarray(beta_sds,  float)   # (K,)
    S, K      = pis_true.shape

    lo = float((mu - pad_sigmas * sigma).min())
    hi = float((mu + pad_sigmas * sigma).max())
    xgrid = np.linspace(lo, hi, grid_len)

    # Precompute component PDFs/CDFs on grid for fast modes; medians via bisection
    # Mode (density argmax on grid)
    comp_pdf = (1.0 / (np.sqrt(2*np.pi) * sigma[None, :])) \
             * np.exp(-0.5 * ((xgrid[:, None] - mu[None, :]) / sigma[None, :])**2)   # (G,K)

    medians = np.empty(S, dtype=float)
    modes   = np.empty(S, dtype=float)

    for s in range(S):
        pi_s = pis_true[s]  # (K,)

        # mode from grid
        dens = comp_pdf @ pi_s
        modes[s] = xgrid[np.argmax(dens)]

        # median by bisection on the exact mixture CDF
        a, b = lo, hi
        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            # mixture CDF at mid: sum_j pi_sj * Phi((mid - mu_j)/sigma_j)
            Fmid = float(np.sum(pi_s * norm.cdf((mid - mu) / sigma)))
            if Fmid < 0.5:
                a = mid
            else:
                b = mid
            if b - a < tol:
                break
        medians[s] = 0.5 * (a + b)

    return medians, modes


def _extract_source(varname: str) -> int:
    """
    Return the source index from 'beta_<s>' or 'beta_<s>_<o>'.
    """
    m = _BETA_EITHER_RE.fullmatch(varname)
    if not m:
        raise ValueError(f"Unrecognized beta var name: {varname}")
    return int(m.group(1))

def _pick_beta_vars_per_source(posterior, regex: str = r"beta_\d+(?:_\d+)?") -> Dict[int, str]:
    """
    For each source s, pick ONE posterior site name.
    Prefer 'beta_<s>' over 'beta_<s>_<o>'.
    Returns: dict {s: varname}
    """
    candidates = [v for v in posterior.data_vars if re.fullmatch(regex, v)]
    by_src: Dict[int, str] = {}
    for v in candidates:
        m = _BETA_EITHER_RE.fullmatch(v)
        if not m:
            continue
        s = int(m.group(1))
        # prefer the shorter name (no outcome suffix)
        if s not in by_src or len(v) < len(by_src[s]):
            by_src[s] = v
    return by_src

def _outcomes_for_source(df_sim: pd.DataFrame, s: int) -> List[int]:
    """Return sorted list of outcome ids for source s from df_sim."""
    if "outcome" not in df_sim.columns:
        return [1]
    return sorted(df_sim.loc[df_sim["source"] == s, "outcome"].unique().tolist())

def _get_posterior_draws_for_source(idata, s: int) -> np.ndarray:
    """
    Retrieve posterior draws for source s from idata.posterior, trying 'beta_<s>'
    then 'beta_<s>_1'. Returns a 1D array (samples,).
    """
    post = idata.posterior
    if f"beta_{s}" in post.data_vars:
        return post[f"beta_{s}"].values.ravel()
    elif f"beta_{s}_1" in post.data_vars:
        return post[f"beta_{s}_1"].values.ravel()
    # Last resort: if user didn't run the shim, try to locate vectorized array
    # named 'beta_s' or 'beta_by_source' and index it.
    for vec_name in ("beta_s", "beta_by_source"):
        if vec_name in post.data_vars:
            da = post[vec_name]
            src_dim = next(d for d in da.dims if d not in ("chain", "draw"))
            return da.isel({src_dim: s - 1}).values.ravel()
    raise KeyError(f"No posterior variable found for source {s}")

# ──────────────────────────────────────────────────────────────────────────────
# Save β posteriors to disk
# ──────────────────────────────────────────────────────────────────────────────
# --- REPLACE store_beta_posteriors with this version -------------------------
def store_beta_posteriors(
    idata,
    df_sim: pd.DataFrame,
    *,
    beta_df: pd.DataFrame | None = None,
    output_folder: str | Path,
    file_ext: str = "npz",
    compress: bool = True,
    regex: str = r"beta_\d+(?:_\d+)?",
    per_outcome: bool = True,
    also_aggregate: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Persist posterior samples of β parameters and their empirical profiles.

    **Per-outcome model aware**:
      If the posterior contains variables named beta_{s}_{o}, we save *that* outcome's
      draws for (s,o). If not, we fall back to a single per-source variable.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    post = idata.posterior

    # Sources we need to write (1-based)
    sources_avl = sorted(int(s) for s in df_sim["source"].unique().tolist())
    result: Dict[str, Dict[str, np.ndarray]] = {}

    # Precompute prior samples dicts (unchanged)
    prior_by_so: Dict[Tuple[int,int], np.ndarray] = {}
    prior_by_s: Dict[int, np.ndarray] = {}
    if beta_df is not None:
        if "outcome" in beta_df.columns:
            for (s, o), grp in beta_df.groupby(["source", "outcome"]):
                draws = np.concatenate([np.atleast_1d(x) for x in grp["beta_true"].to_list()])
                prior_by_so[(s, o)] = draws
        for s, grp in beta_df.groupby("source"):
            draws = np.concatenate([np.atleast_1d(x) for x in grp["beta_true"].to_list()])
            prior_by_s[s] = draws

    def _post_draws(s: int, o: int | None) -> np.ndarray:
        # Prefer per-outcome variable if present
        if o is not None and f"beta_{s}_{o}" in post.data_vars:
            return post[f"beta_{s}_{o}"].values.ravel()
        # Fallbacks: beta_s or beta_s_1 or vectorized array
        try:
            return _get_posterior_draws_for_source(idata, s)
        except KeyError:
            raise KeyError(f"No posterior β draws found for source={s}, outcome={o}")

    for s in sources_avl:
        outcomes = _outcomes_for_source(df_sim, s)

        # Aggregate file beta_<s>.* if requested — concatenate all outcome draws
        if also_aggregate:
            all_draws = [ _post_draws(s, o) for o in outcomes ]
            samples_s = np.concatenate(all_draws, axis=0)
            mask = df_sim["source"] == s
            sub = (df_sim.loc[mask, ["point", "value"]]
                   .groupby("point", sort=True)["value"].sum().sort_index())
            bundle_s = dict(samples=samples_s, grid=sub.index.to_numpy(),
                            loglik=sub.values, prior_samples=prior_by_s.get(s, np.array([])))
            key_s = f"beta_{s}"; result[key_s] = bundle_s
            out_s = output_folder / f"{key_s}.{file_ext}"
            (np.savez_compressed if compress and file_ext == "npz" else np.savez)(out_s, **bundle_s)

        # Per-outcome files beta_<s>_<o>.*
        if per_outcome:
            for o in outcomes:
                mask = (df_sim["source"] == s) & (df_sim["outcome"] == o)
                sub = df_sim.loc[mask, ["point", "value"]].sort_values("point")
                grid, loglik = sub["point"].to_numpy(), sub["value"].to_numpy()
                samples = _post_draws(s, o)
                bundle = dict(samples=samples, grid=grid, loglik=loglik,
                              prior_samples=prior_by_so.get((s, o), np.array([])))
                key = f"beta_{s}_{o}"; result[key] = bundle
                out = output_folder / f"{key}.{file_ext}"
                (np.savez_compressed if compress and file_ext == "npz" else np.savez)(out, **bundle)

    print(f"Stored {len(result)} β archives in {output_folder}")
    return result



def load_beta_posteriors(
    folder: str | Path,
    *,
    file_ext: str = "npz",
    regex: str = r"beta_\d+(?:_\d+)?\.",  # matches beta_<s>.<ext> and beta_<s>_<o>.<ext>
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Read back the objects produced by `store_beta_posteriors`.
    Returns dict keyed by 'beta_<s>' and/or 'beta_<s>_<o>'.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")

    pattern = re.compile(regex + re.escape(file_ext) + r"$")
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for f in sorted(folder.iterdir()):
        if not pattern.fullmatch(f.name):
            continue
        varname = f.stem  # e.g. "beta_1_2"
        if file_ext == "npz":
            with np.load(f, allow_pickle=True) as data:
                out[varname] = {
                    "samples": data["samples"],
                    "grid": data["grid"],
                    "loglik": data["loglik"],
                    "prior_samples": data["prior_samples"] if "prior_samples" in data else np.array([]),
                }
        elif file_ext == "npy":
            out[varname] = np.load(f, allow_pickle=True).item()
        else:
            raise ValueError("file_ext must be 'npz' or 'npy'")

    if not out:
        raise RuntimeError(f"No *.{file_ext} files matching {regex} found in {folder}")
    print(f"Loaded {len(out)} β archives from {folder}")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
# --- REPLACE save_beta_density_hist with this version ------------------------
def save_beta_density_hist(
    idata,
    *,
    output_folder: str | Path,
    experiment: str,
    df_sim: pd.DataFrame | None = None,
    regex: str = r"beta_\d+(?:_\d+)?",
    per_outcome: bool = True,
    bins: int = 40,
    dpi: int = 300,
):
    """
    Save histograms (and KDE) of posterior β draws.

    **Per-outcome model aware**:
    If `beta_{s}_{o}` exists in the posterior, use its draws for (s,o).
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    post = idata.posterior

    def _plot_and_save(draws: np.ndarray, label: str):
        fig, (ax_kde, ax_hist) = plt.subplots(
            2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"height_ratios": (2, 1)}
        )
        az.plot_kde(draws, ax=ax_kde, plot_kwargs={"color": "steelblue"})
        ax_kde.set_ylabel("Density"); ax_kde.set_xlabel(""); ax_kde.grid(True)
        ax_hist.hist(draws, bins=bins, density=False, color="steelblue", alpha=0.6)
        ax_hist.set_ylabel("Frequency"); ax_hist.set_xlabel(""); ax_hist.grid(True)
        fig.tight_layout()
        fig.savefig(output_folder / f"{experiment}_{label}.jpg", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # sources present (1-based)
    sources_avl = sorted(int(s) for s in (df_sim["source"].unique() if df_sim is not None else []))
    if not sources_avl:
        # fallback: discover from posterior variable names
        from re import fullmatch
        candidates = [v for v in post.data_vars if fullmatch(regex, v)]
        sources_avl = sorted({ int(v.split("_")[1]) for v in candidates })

    for s in sources_avl:
        if per_outcome and df_sim is not None:
            for o in _outcomes_for_source(df_sim, s):
                var_so = f"beta_{s}_{o}"
                if var_so in post.data_vars:   # preferred
                    draws = post[var_so].values.ravel()
                else:                          # fallback
                    draws = _get_posterior_draws_for_source(idata, s)
                _plot_and_save(draws, f"beta_{s}_{o}")
        else:
            draws = _get_posterior_draws_for_source(idata, s)
            _plot_and_save(draws, f"beta_{s}")

    print(f"Saved β density histograms to {output_folder}")

def plot_df_prior_vs_posterior(
    idata,
    beta_df: pd.DataFrame,
    *,
    kde_bw: str | float | None = None,
    density_cut: float = 1e-2,
    xgrid_len: int = 2_000,
    save_to: str | Path | None = None,
    save_dir: str | Path | None = None,   # << if set, write per-(s,o) figures to this folder
    by_outcome: bool = True,              # << default per-outcome
):
    """
    Compare prior vs posterior for β.
    If by_outcome=True and save_dir is set, write one figure per (s,o) named:
        {save_dir}/ppc_beta_{s}_{o}.png
    Otherwise, make a single grid figure (per source) and use save_to.

    Notes:
      • Posterior draws are per-source (β_s), reused across all outcomes for that source.
      • Prior samples come from beta_df; for by_outcome=True, we expect 'outcome' column.
    """
    post = idata.posterior
    by_src = _pick_beta_vars_per_source(post, regex=r"beta_\d+(?:_\d+)?")
    if not by_src:
        raise ValueError("No beta_* variables found in idata.posterior")

    # Build prior dictionaries
    prior_by_so: Dict[Tuple[int,int], np.ndarray] = {}
    prior_by_s: Dict[int, np.ndarray] = {}
    if "outcome" in beta_df.columns:
        for (s, o), grp in beta_df.groupby(["source", "outcome"]):
            draws = np.concatenate([np.atleast_1d(x) for x in grp["beta_true"].to_list()])
            prior_by_so[(s, o)] = draws
    for s, grp in beta_df.groupby("source"):
        prior_by_s[s] = np.concatenate([np.atleast_1d(x) for x in grp["beta_true"].to_list()])

    # Per-outcome files
    if by_outcome and save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for s, var in sorted(by_src.items()):
            post_draws = post[var].values.ravel()
            # Derive outcomes available from beta_df for source s
            outcomes = sorted({o for (ss, o) in prior_by_so.keys() if ss == s}) or [1]
            for o in outcomes:
                prior_draws = prior_by_so.get((s, o))
                if prior_draws is None or prior_draws.size == 0:
                    # fallback to pooled source prior if per-outcome not present
                    prior_draws = prior_by_s.get(s, np.array([]))
                if prior_draws is None or prior_draws.size == 0:
                    # skip if no prior information at all
                    continue

                # KDEs
                prior_kde = gaussian_kde(prior_draws, bw_method=kde_bw)
                post_kde  = gaussian_kde(post_draws,  bw_method=kde_bw)

                xmin = min(prior_draws.min(), post_draws.min())
                xmax = max(prior_draws.max(), post_draws.max())
                xgrid = np.linspace(xmin, xmax, xgrid_len)

                p_prior = prior_kde(xgrid)
                p_post  = post_kde(xgrid)

                mask = np.maximum(p_prior, p_post) > density_cut

                fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
                if mask.any():
                    ax.set_xlim(xgrid[mask].min(), xgrid[mask].max())
                ax.plot(xgrid, p_prior, label=f"Prior (s={s}, o={o})")
                ax.plot(xgrid, p_post,  label="Posterior")
                ax.set_title(f"Prior vs Posterior β (s={s}, o={o})")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel("Density")
                ax.grid(True); ax.legend()
                fig.tight_layout()
                fig.savefig(save_dir / f"ppc_beta_{s}_{o}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                count += 1
        print(f"Wrote {count} per-outcome PPC figures to {save_dir}")
        return

    # Otherwise: a single grid figure grouped by source (pooled prior)
    sources = sorted(by_src)
    n = len(sources)
    ncols, nrows = 2, int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = axes.flatten()

    for idx, s in enumerate(sources):
        var = by_src[s]
        post_draws = post[var].values.ravel()
        prior_draws = prior_by_s.get(s, None)
        if prior_draws is None or prior_draws.size == 0:
            continue

        prior_kde = gaussian_kde(prior_draws, bw_method=kde_bw)
        post_kde  = gaussian_kde(post_draws,  bw_method=kde_bw)

        xmin = min(prior_draws.min(), post_draws.min())
        xmax = max(prior_draws.max(), post_draws.max())
        xgrid = np.linspace(xmin, xmax, xgrid_len)

        p_prior = prior_kde(xgrid)
        p_post  = post_kde(xgrid)

        mask = np.maximum(p_prior, p_post) > density_cut
        if mask.any():
            axes[idx].set_xlim(xgrid[mask].min(), xgrid[mask].max())

        axes[idx].plot(xgrid, p_prior, label="Prior (pooled over outcomes)")
        axes[idx].plot(xgrid, p_post,  label="Posterior")
        axes[idx].set_title(f"Source {s}")
        axes[idx].set_xlabel(r"$\beta$")
        axes[idx].set_ylabel("Density")
        axes[idx].grid(True); axes[idx].legend()

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Pi summaries and G0 utilities (unchanged except for robust array handling)
# ──────────────────────────────────────────────────────────────────────────────
def _as_ndarray(obj) -> np.ndarray:
    """Return obj as a NumPy array, handling xarray objects transparently."""
    if isinstance(obj, (np.ndarray, list, tuple)):
        return np.asarray(obj)
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        return obj.values
    raise TypeError(
        "Expected NumPy array-like or xarray object for `pis`, "
        f"got {type(obj).__name__}"
    )

def largest_mean_pi_norm(
    idata,
    *,
    pis: Any | None = None,
    pis_var: str = "pis",
) -> List[Dict[str, Any]]:
    mean_pi = idata.posterior["pi_norm"].mean(dim=("chain", "draw"))
    mean_vals = mean_pi.values
    best_comp_idx = mean_vals.argmax(axis=1)
    if pis is None:
        if pis_var in idata.posterior:
            pis = idata.posterior[pis_var]
        else:
            pis = None
    pis_vals = None
    if pis is not None:
        pis_vals = _as_ndarray(pis)
        if pis_vals.shape != mean_vals.shape:
            raise ValueError(
                "Shape mismatch between `pis` "
                f"{pis_vals.shape} and pi_norm mean {mean_vals.shape}"
            )
    results = []
    for src, comp in enumerate(best_comp_idx):
        true_prop = float(pis_vals[src].max()) if pis_vals is not None else None
        results.append(
            {
                "source": src,
                "component": int(comp),
                "mean_value": float(mean_vals[src, comp]),
                "true_proportion": true_prop,
            }
        )
    return results

def compute_g0_components(idata, output_path=None, xgrid_len=200, weight_thres=0.0):
    post = idata.posterior
    pi_norm_vals = post["pi_norm"].values
    alpha0_vals  = post["alpha0"].values
    mu_vals      = post["mu"].values
    sigma_vals   = post["sigma"].values

    n_chains, n_draws, N_sources, k = pi_norm_vals.shape
    n_samples = n_chains * n_draws
    pi_norm = pi_norm_vals.reshape(n_samples, N_sources, k)
    alpha0  = alpha0_vals.reshape(n_samples)
    mu      = mu_vals.reshape(n_samples, k)
    sigma   = sigma_vals.reshape(n_samples, k)

    activated = (pi_norm > weight_thres)
    m_samps  = activated.sum(axis=1)

    zeta = (m_samps + alpha0[:, None] / k) \
         / (m_samps.sum(axis=1)[:, None] + alpha0[:, None])

    x_min = mu.min() - 3 * sigma.max()
    x_max = mu.max() + 3 * sigma.max()
    xgrid = np.linspace(x_min, x_max, xgrid_len)

    g0 = np.zeros_like(xgrid)
    for i in range(n_samples):
        for h in range(k):
            coef = zeta[i, h]
            pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma[i, h])) * \
                  np.exp(-0.5 * ((xgrid - mu[i, h]) / sigma[i, h])**2)
            g0 += coef * pdf
    g0 /= n_samples

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 xgrid=xgrid, g0=g0, mu=mu, sigma=sigma,
                 pi_norm=pi_norm, alpha0=alpha0, zeta=zeta, m_samps=m_samps)
    else:
        return {
            "xgrid": xgrid, "g0": g0, "mu": mu, "sigma": sigma,
            "pi_norm": pi_norm, "alpha0": alpha0, "zeta": zeta, "m_samps": m_samps
        }

def plot_g0_density_from_file(npz_path, density_cut, save_to=None):
    data = np.load(npz_path)
    xgrid = data["xgrid"]
    g0    = data["g0"]
    mask = g0 > density_cut
    if not mask.any():
        raise ValueError(f"No points exceed density_cut={density_cut}")
    plt.figure(figsize=(8, 4))
    plt.plot(xgrid[mask], g0[mask])
    plt.title(r"Posterior Mean of $G_0$ Density")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Density of $G_0$")
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def kl_prior_posterior_beta(
    idata,
    beta_df: pd.DataFrame,
    *,
    posterior_bins: int = 200,
    eps: float = 1e-12,
):
    """
    KL(posterior || prior) for each source using Monte Carlo histograms.
    This version is per-source (pooled prior). If you need per-outcome KL,
    compute on beta_<s> posterior with prior_by_so[(s,o)] externally.
    """
    # pooled prior per source
    prior_by_s = {
        s: np.concatenate([np.atleast_1d(x) for x in grp["beta_true"].to_list()])
        for s, grp in beta_df.groupby("source")
    }
    by_src = _pick_beta_vars_per_source(idata.posterior)
    sources_avl = sorted(by_src)
    if not sources_avl:
        raise ValueError("No beta_* variables found in idata.posterior")

    kl_dict = {}
    for s in sources_avl:
        prior_draws = prior_by_s.get(s, None)
        if prior_draws is None or prior_draws.size == 0:
            continue
        post_draws = _get_posterior_draws_for_source(idata, s)

        xmin = min(prior_draws.min(), post_draws.min())
        xmax = max(prior_draws.max(), post_draws.max())
        edges = np.linspace(xmin, xmax, posterior_bins + 1)
        widths = np.diff(edges)
        prior_hist, _ = np.histogram(prior_draws, bins=edges, density=True)
        post_hist, _  = np.histogram(post_draws,  bins=edges, density=True)
        p_post  = post_hist  + eps
        p_prior = prior_hist + eps
        kl_val = np.sum(widths * p_post * np.log(p_post / p_prior))
        kl_dict[s] = kl_val
    return kl_dict

def summarise_beta_prior_posterior(
    idata,
    beta_df: pd.DataFrame,
    *,
    kl_dict: dict[int, float] | None = None,
    glm_est_df: pd.DataFrame | None = None,
    experiment_label: int | str = 1,
    mode_bins: int = 30,
    # NEW: pass truths to override empirical prior summaries
    true_source_means:   np.ndarray | None = None,
    true_source_medians: np.ndarray | None = None,
    true_source_modes:   np.ndarray | None = None,
):
    """
    Summary (per source) comparing prior vs posterior.

    If true_source_* arrays are provided (shape (S,)), we set:
      prior_mean   := true_source_means[s-1]
      prior_median := true_source_medians[s-1]
      prior_mode   := true_source_modes[s-1]
    Then:
      mean_error   = prior_mean   − post_mean
      median_error = prior_median − post_median
      CI coverage booleans are evaluated against these 'prior_*' values
      (which now represent truth).
    """
    # pooled empirical prior per source (used as fallback)
    prior_by_s = {
        s: np.concatenate([np.atleast_1d(x) for x in grp["beta_true"].to_list()])
        for s, grp in beta_df.groupby("source")
    }

    by_src = _pick_beta_vars_per_source(idata.posterior)
    sources_avl = sorted(by_src)
    if not sources_avl:
        raise ValueError("No beta_* variables found in idata.posterior")
    kl_dict = kl_dict or {}

    records = []
    for s in sources_avl:
        emp_prior = prior_by_s.get(s, np.array([]))

        # defaults from empirical prior (for back-compat)
        if emp_prior.size:
            counts, bins = np.histogram(emp_prior, bins=mode_bins)
            bin_centers  = 0.5 * (bins[:-1] + bins[1:])
            emp_mode     = float(bin_centers[np.argmax(counts)])
            emp_mean     = float(emp_prior.mean())
            emp_median   = float(np.median(emp_prior))
        else:
            emp_mode = emp_mean = emp_median = np.nan

        # OVERRIDES with truth if provided (sources are 1-based)
        prior_mean   = float(true_source_means[s - 1])   if true_source_means   is not None else emp_mean
        prior_median = float(true_source_medians[s - 1]) if true_source_medians is not None else emp_median
        prior_mode   = float(true_source_modes[s - 1])   if true_source_modes   is not None else emp_mode

        post = _get_posterior_draws_for_source(idata, s)
        post_mean   = float(post.mean())
        post_median = float(np.median(post))
        ci_lower, ci_upper = np.percentile(post, [2.5, 97.5])

        rec = {
            "experiment":             experiment_label,
            "source":                 s,
            "prior_mean":             prior_mean,     # now truth if provided
            "prior_median":           prior_median,   # now truth if provided
            "prior_mode":             prior_mode,     # now truth if provided
            "post_mean":              post_mean,
            "post_median":            post_median,
            "mean_error":             prior_mean   - post_mean,
            "median_error":           prior_median - post_median,
            "kl_divergence":          kl_dict.get(s, np.nan),
            "post_ci_lower":          ci_lower,
            "post_ci_upper":          ci_upper,
            "ci_covers_prior_mean":   (ci_lower <= prior_mean   <= ci_upper),
            "ci_covers_prior_median": (ci_lower <= prior_median <= ci_upper),
            "ci_covers_prior_mode":   (ci_lower <= prior_mode   <= ci_upper),
        }
        if glm_est_df is not None:
            est_row = glm_est_df.loc[glm_est_df["source"] == s, "beta_est"]
            rec["glm_beta_est"] = float(est_row.values[0]) if not est_row.empty else np.nan

        records.append(rec)

    return pd.DataFrame.from_records(records)


