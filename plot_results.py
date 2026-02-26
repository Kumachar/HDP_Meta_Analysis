from __future__ import annotations
import pandas as pd
import seaborn as sns

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils import load_beta_posteriors, true_mixture_density

from matplotlib.patches import Patch


def _sorted_sources(values):
    """Sort source labels numerically when possible; otherwise lexicographically."""
    def _key(v):
        try:
            return (0, float(v))
        except Exception:
            return (1, str(v))
    return sorted(values, key=_key)


def _dual_axis_grouped_boxplot(
    df: pd.DataFrame,
    *,
    metric: str,
    families: list[str],
    palette: dict[str, str],
    secondary_model: str = "logistic",
    out_path: Path,
    title: str,
    xlabel: str = "Source",
    dpi: int = 300,
    # --- y-axis controls ---
    center_zero: bool = True,
    whis: float = 1.5,
    ypad_frac: float = 0.08,
    force_zero_line: bool = True,
) -> None:
    """Grouped boxplots by source with a secondary y-axis for one model family.

    Notes
    -----
    - We often set `showfliers=False` for cleaner plots. If we then scale the axis
      using raw min/max, a few extreme values (hidden as fliers) can blow up the
      y-range and make all visible boxes look tiny.
    - To prevent that, we compute y-limits from *whisker* extents (per-box),
      which matches what is actually visible in the plot.
    - Optionally `center_zero=True` forces 0 to be at the vertical midpoint
      for each axis (left and right separately).
    """
    # Local import for robustness (avoids NameError if global import changes)
    from matplotlib.patches import Patch

    cols = ["source", "model", metric]
    dfp = df.loc[:, [c for c in cols if c in df.columns]].copy()
    if dfp.empty or metric not in dfp.columns:
        return

    dfp = dfp.dropna(subset=["source", "model", metric])
    if dfp.empty:
        return

    models_present = [m for m in families if m in set(dfp["model"].unique())]
    if secondary_model not in models_present or len(models_present) <= 1:
        return

    left_models = [m for m in models_present if m != secondary_model]
    if not left_models:
        return

    sources = _sorted_sources(dfp["source"].unique())
    base = np.arange(len(sources), dtype=float)

    # Allocate group width across all models so x-positions align with seaborn-style dodge
    total_n = len(models_present)
    group_width = 0.78
    box_w = group_width / total_n
    offsets = {
        m: (-group_width / 2.0 + box_w / 2.0) + i * box_w
        for i, m in enumerate(models_present)
    }

    fig, axL = plt.subplots(figsize=(10.0, 5.2), dpi=dpi)
    axR = axL.twinx()
    axR.grid(False)

    def _draw_boxes(ax, model_name: str, *, zorder: int):
        data = []
        pos = []
        for i, s in enumerate(sources):
            vals = dfp.loc[(dfp["source"] == s) & (dfp["model"] == model_name), metric].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            pos.append(base[i] + offsets[model_name])
        if not data:
            return

        bp = ax.boxplot(
            data,
            positions=pos,
            widths=box_w * 0.92,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=zorder,
            whis=whis,
        )
        face = palette.get(model_name, None)
        for b in bp.get("boxes", []):
            if face is not None:
                b.set_facecolor(face)
            b.set_alpha(0.70)
            b.set_linewidth(1.0)
        for k in ("whiskers", "caps", "medians"):
            for line in bp.get(k, []):
                line.set_linewidth(1.0)

    # Draw left-axis models first, then secondary model on the right axis
    for m in left_models:
        _draw_boxes(axL, m, zorder=3)
    _draw_boxes(axR, secondary_model, zorder=4)

    # X formatting
    axL.set_xticks(base)
    axL.set_xticklabels([str(s) for s in sources])
    axL.set_xlabel(xlabel)

    pretty_metric = metric.replace("_", " ")
    axL.set_ylabel(f"{pretty_metric} (non-{secondary_model})")
    axR.set_ylabel(f"{pretty_metric} ({secondary_model})")
    axL.set_title(title)

    handles = [
        Patch(facecolor=palette.get(m, "#cccccc"), edgecolor="black", label=m, alpha=0.70)
        for m in models_present
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize="small")

    def _whisker_min_max(vals: np.ndarray) -> tuple[float, float]:
        """Return whisker-extreme min/max for one box (matches matplotlib default)."""
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (np.inf, -np.inf)
        if vals.size < 4:
            return (float(vals.min()), float(vals.max()))

        q1, q3 = np.quantile(vals, [0.25, 0.75])
        iqr = q3 - q1
        if (not np.isfinite(iqr)) or (iqr < 1e-12):
            return (float(vals.min()), float(vals.max()))

        lo_bound = q1 - whis * iqr
        hi_bound = q3 + whis * iqr

        lo_candidates = vals[vals >= lo_bound]
        hi_candidates = vals[vals <= hi_bound]

        lo = float(lo_candidates.min()) if lo_candidates.size else float(vals.min())
        hi = float(hi_candidates.max()) if hi_candidates.size else float(vals.max())
        return (lo, hi)

    def _set_ylim(ax, models: list[str]) -> None:
        # IMPORTANT: use *whisker* extents, not raw min/max, so hidden fliers do not inflate the axis.
        g_lo, g_hi = np.inf, -np.inf
        for m in models:
            for s in sources:
                v = dfp.loc[(dfp["source"] == s) & (dfp["model"] == m), metric].to_numpy()
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                lo, hi = _whisker_min_max(v)
                if np.isfinite(lo):
                    g_lo = min(g_lo, lo)
                if np.isfinite(hi):
                    g_hi = max(g_hi, hi)

        if not (np.isfinite(g_lo) and np.isfinite(g_hi)):
            return

        if center_zero:
            M = max(abs(g_lo), abs(g_hi))
            if (not np.isfinite(M)) or (M < 1e-12):
                M = 1.0
            pad = float(ypad_frac) * M
            ax.set_ylim(-(M + pad), (M + pad))
        else:
            rng = g_hi - g_lo
            if rng < 1e-12:
                pad = 1.0 if g_hi == 0 else abs(g_hi) * 0.25
            else:
                pad = float(ypad_frac) * rng
            ax.set_ylim(g_lo - pad, g_hi + pad)

    _set_ylim(axL, left_models)
    _set_ylim(axR, [secondary_model])

    if force_zero_line:
        # draw on left axis; on a twinx figure this still visually marks the shared zero level
        axL.axhline(0.0, color="black", linewidth=0.9, alpha=0.35, zorder=1)

    fig.tight_layout(rect=[0.0, 0.0, 0.96, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def newest_csv(model_dir: Path) -> Path | None:
    """Return the most recent beta_summary_stats.csv under model_dir, or None."""
    csvs = sorted(model_dir.glob("beta_summary_stats.csv"))
    return csvs[-1] if csvs else None


def generate_summary_plots(
    results_root: Path,
    families: list[str],
    out_fig_dir: Path,
) -> None:
    """
    Load & combine beta_summary_stats.csv for each family,
    then produce and save:
      1) Boxplots of errors & KL divergence,
      2) Coverage‐rate barplots (mean, median, mode, and combined facet),
      3) Boxplots of (post_mean − glm_beta_est) & (post_median − glm_beta_est).
    """
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Load & combine ----
    frames = []
    for fam in families:
        mdl_dir = results_root / fam
        csv = newest_csv(mdl_dir)
        if csv is None:
            print(f"[warn] No summary CSV in {mdl_dir}; skipping.")
            continue
        df = pd.read_csv(csv)
        df["model"] = fam
        frames.append(df)
    if not frames:
        raise RuntimeError("No CSVs found for any family; aborting.")

    stats_all = pd.concat(frames, ignore_index=True)

    # fixed palette for consistency
    palette = {
        fam: col for fam, col in zip(
            families,
            ["#4c72b0", "#dd8452", "#55a868"][: len(families)]
        )
    }

    sns.set_theme(style="whitegrid")

    # ---- 2) Box‐plots for errors & KL divergence ----
    for metric in ["mean_error", "median_error", "kl_divergence"]:
        g = sns.catplot(
            data=stats_all,
            x="source", y=metric, hue="model",
            hue_order=families, palette=palette,
            kind="box", height=5, aspect=1.8,
            showfliers=False,
        )
        g.set_axis_labels("β parameter (source)", metric.replace("_", " ").title())
        g.fig.suptitle(f"Distribution of {metric.replace('_',' ')} across repetitions")
        g.tight_layout()
        g.savefig(out_fig_dir / f"{metric}_boxplot.png", dpi=300)
        plt.close(g.fig)

    


    # ---- 2b) (Optional) Box-plots for outcome_error_mean / mu_s_hat_error (if present) ----
    for metric in ["outcome_error_mean", "mu_s_hat_error"]:
        if metric not in stats_all.columns:
            continue

        out_path = out_fig_dir / f"{metric}_boxplot.png"

        # Special case: give 'logistic' its own right-side y-axis for outcome_error_mean
        if metric == "outcome_error_mean" and ("logistic" in families) and (len(families) > 1):
            _dual_axis_grouped_boxplot(
                stats_all,
                metric=metric,
                families=families,
                palette=palette,
                secondary_model="logistic",
                out_path=out_path,
                title=f"Distribution of {metric.replace('_', ' ')} across repetitions",
                xlabel="Source",
                dpi=300,
            )
            if out_path.exists():
                continue  # dual-axis plot successfully written; skip seaborn fallback

        # Default: standard single-axis seaborn boxplot
        g = sns.catplot(
            data=stats_all,
            x="source", y=metric, hue="model",
            hue_order=families, palette=palette,
            kind="box", height=5, aspect=1.8,
            showfliers=False,
        )
        g.set_axis_labels("Source", metric.replace("_", " ").title())
        g.fig.suptitle(f"Distribution of {metric.replace('_', ' ')} across repetitions")
        g.tight_layout()
        g.savefig(out_path, dpi=300)
        plt.close(g.fig)
    # ---- 3) Coverage-rate bar-plots ----
    cov_cols = ["ci_covers_prior_mean", "ci_covers_prior_median", "ci_covers_prior_mode"]

    # combined facet plot
    coverage_melt = (
        stats_all
        .groupby(["model", "source"])[cov_cols]
        .mean()
        .reset_index()
        .melt(
            id_vars=["model", "source"],
            value_vars=cov_cols,
            var_name="coverage_type",
            value_name="coverage_rate",
        )
    )
    coverage_melt["coverage_type"] = coverage_melt["coverage_type"].map({
        "ci_covers_prior_mean":   "Prior Mean",
        "ci_covers_prior_median": "Prior Median",
        "ci_covers_prior_mode":   "Prior Mode",
    })

    g = sns.catplot(
        data=coverage_melt,
        x="source", y="coverage_rate", hue="model", col="coverage_type",
        hue_order=families, palette=palette,
        kind="bar", height=4.5, aspect=0.8, sharey=True,
    )
    g.set_axis_labels("β parameter (source)", "Coverage rate")
    g.set(ylim=(0, 1))
    g.fig.suptitle("95% CI Coverage Rates", y=1.02)
    g.tight_layout(rect=[0, 0, 1, 0.95])
    g.savefig(out_fig_dir / "ci_coverage_all_barplots.png", dpi=300)
    plt.close(g.fig)

    # individual coverage plots
    coverage = (
        stats_all
        .groupby(["model", "source"])[cov_cols]
        .mean()
        .reset_index()
    )
    coverage_map = {
        "ci_covers_prior_mean":   ("Prior Mean",   "ci_coverage_mean_barplot.png"),
        "ci_covers_prior_median": ("Prior Median", "ci_coverage_median_barplot.png"),
        "ci_covers_prior_mode":   ("Prior Mode",   "ci_coverage_mode_barplot.png"),
    }
    for col, (label, fname) in coverage_map.items():
        g = sns.catplot(
            data=coverage,
            x="source", y=col, hue="model",
            hue_order=families, palette=palette,
            kind="bar", height=5, aspect=1.8,
        )
        g.set_axis_labels("β parameter (source)", "Coverage rate")
        g.set(ylim=(0, 1))
        g.fig.suptitle("95% CI Coverage Rates")
        g.tight_layout()
        g.savefig(out_fig_dir / fname, dpi=300)
        plt.close(g.fig)

    # ---- 4) Posterior vs GLM differences ----
    stats_all["post_glm_mean_diff"] = stats_all["post_mean"] - stats_all["glm_beta_est"]
    stats_all["post_glm_median_diff"] = stats_all["post_median"] - stats_all["glm_beta_est"]

    for diff_col, title in [
        ("post_glm_mean_diff",  "Posterior Mean − GLM Estimate"),
        ("post_glm_median_diff","Posterior Median − GLM Estimate"),
    ]:
        g = sns.catplot(
            data=stats_all,
            x="source", y=diff_col, hue="model",
            hue_order=families, palette=palette,
            kind="box", height=5, aspect=1.8, showfliers=False,
        )
        g.set_axis_labels("β parameter (source)", title)
        g.fig.suptitle(f"Distribution of ({title}) Across Sources")
        g.tight_layout()
        fname = diff_col.replace("_", "") + "_boxplot.png"
        g.savefig(out_fig_dir / fname, dpi=300)
        plt.close(g.fig)

    print(f"✓ All figures saved to: {out_fig_dir.resolve()}")

def plot_and_save_g0_by_rep(
    output_base: str,
    model_type: str,
    reps: list[int],
    max_curves: int = 100,
    traj_alpha: float = 0.1,
    traj_color: str = "tab:blue",
    mean_color: str = "black",
    figsize=(6, 4),
    dpi: int = 300
):
    """
    For each rep in `reps`, load its g0_components.npz, plot up to `max_curves`
    individual G₀ trajectories (faint lines), overlay the posterior mean G₀
    as a dashed line, and save the figure.
    """
    fig_dir = Path(output_base) / model_type / "figures" / "g0_trajectory"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for rep in reps:
        npz_path = Path(output_base) / model_type / "data" / f"rep{rep}" / "g0_components.npz"
        data     = np.load(npz_path)
        xgrid    = data["xgrid"]      # (X,)
        mu       = data["mu"]         # (n_samples, k)
        sigma    = data["sigma"]      # (n_samples, k)
        zeta     = data["zeta"]       # (n_samples, k)
        g0_mean  = data["g0"]         # (X,)

        n_samples, k = zeta.shape
        picks = np.random.default_rng(0).choice(n_samples,
                                               size=min(max_curves, n_samples),
                                               replace=False)

        fig, ax = plt.subplots(figsize=figsize)

        # add a dummy line for the sample trajectories legend entry
        ax.plot([], [],
                color=traj_color,
                alpha=traj_alpha,
                linewidth=1,
                label="Posterior samples")

        # plot individual trajectories (no label)
        for i in picks:
            pdfs = (1.0 / (np.sqrt(2*np.pi) * sigma[i, :]))[:, None] * \
                   np.exp(-0.5 * ((xgrid[None, :] - mu[i, :][:, None]) / sigma[i, :][:, None])**2)
            g0_i = np.sum(zeta[i, :][:, None] * pdfs, axis=0)
            ax.plot(xgrid, g0_i,
                    color=traj_color,
                    alpha=traj_alpha,
                    linewidth=1)

        # overlay posterior mean as dashed line
        ax.plot(xgrid, g0_mean,
                color=mean_color,
                linestyle="--",
                linewidth=1,
                label="Posterior mean $G_0$")

        ax.set_title(f"$G_0$ trajectories (rep {rep})")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        fig.savefig(fig_dir / f"rep{rep}_g0_trajs.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_g0_and_beta_single_axis(
    output_base: str,
    model_type: str,
    reps: list[int],
    xlim: tuple[float,float] = (-3, 3),
    figsize=(6, 4),
    dpi: int = 300
):
    """
    For each rep:
      - load g0_components → xgrid, g0_mean
      - load β-posteriors via load_beta_posteriors()
      - plot β KDEs on left y-axis (labeled by source number)
      - overlay G0_mean on right y-axis
      - xlim fixed to (-3,3)
    """
    fig_dir = Path(output_base) / model_type / "figures" / "g0_and_beta"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for rep in reps:
        rep_folder = Path(output_base) / model_type / "data" / f"rep{rep}"
        # — load G0 —
        data = np.load(rep_folder / "g0_components.npz")
        xgrid, g0_mean = data["xgrid"], data["g0"]

        # — load β posteriors —
        beta_s = load_beta_posteriors(folder=str(rep_folder))
        # beta_s keys are like 'beta_1', 'beta_2', …

        # — set up twin axes —
        fig, ax_beta = plt.subplots(figsize=figsize, dpi=dpi)
        ax_g0 = ax_beta.twinx()

        # plot β densities, labeling by source number only
        for beta_name, info in sorted(beta_s.items(), key=lambda kv: int(kv[0].split("_")[1])):
            # extract numeric source
            src = int(beta_name.split("_")[1])
            samples = info["samples"]
            kde = gaussian_kde(samples)
            xs = np.linspace(xlim[0], xlim[1], 200)
            ax_beta.plot(xs, kde(xs), label=f"Source {src}", alpha=0.7)

        ax_beta.set_xlim(*xlim)
        ax_beta.set_xlabel(r"$\beta$")
        ax_beta.set_ylabel("Posterior β density")

        # overlay G0 mean on right axis
        mask = (xgrid >= xlim[0]) & (xgrid <= xlim[1])
        ax_g0.plot(xgrid[mask], g0_mean[mask],
                   color="black", linestyle="--", linewidth=1,
                   label="Posterior mean $G_0$")
        ax_g0.set_ylabel("Posterior mean $G_0$")

        # combine legends from both axes
        h0, l0 = ax_beta.get_legend_handles_labels()
        h1, l1 = ax_g0.get_legend_handles_labels()
        ax_beta.legend(h0 + h1, l0 + l1, loc="best", fontsize="small")

        ax_beta.grid(True)
        plt.title(f"β densities & $G_0$ mean")
        fig.tight_layout()

        fig.savefig(fig_dir / f"rep{rep}_g0_beta.png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def plot_true_vs_posterior_sources(
    output_base: str | Path,
    model_type: str,
    reps: list[int],
    x_pad: float = 4.0,
    dpi: int = 300,
):
    """
    For each rep and each source:
      - load truth_source_mixtures.npz (beta_mean, beta_sds, pis),
      - pool posterior draws across all outcomes for that source,
      - overlay true mixture density vs posterior β KDE.
    Writes PNGs to: {output_base}/{model_type}/figures/true_vs_post/rep{rep}_src{s}.png
    """
    output_base = Path(output_base)
    fig_dir = output_base / model_type / "figures" / "true_vs_post"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for rep in reps:
        rep_dir = output_base / model_type / "data" / f"rep{rep}"
        truth_npz = np.load(rep_dir / "truth_source_mixtures.npz")
        beta_mean = truth_npz["beta_mean"]   # (K,)
        beta_sds  = truth_npz["beta_sds"]    # (K,)
        pis_true  = truth_npz["pis"]         # (S, K)  saved by main()

        beta_arch = load_beta_posteriors(folder=str(rep_dir))  # keys: beta_s_o
        # discover available sources and outcomes
        def _src_idx(k: str) -> int | None:
            parts = k.split("_")
            return int(parts[1]) if len(parts) >= 3 else None
        sources = sorted({ _src_idx(k) for k in beta_arch.keys() if _src_idx(k) is not None })

        for s in sources:
            draws_list = [info["samples"]
                          for k, info in beta_arch.items()
                          if k.startswith(f"beta_{s}_")]
            if not draws_list:
                continue
            draws = np.concatenate(draws_list, axis=0).ravel()
            kde   = gaussian_kde(draws)

            lo = min(draws.min(), (beta_mean - x_pad * beta_sds).min())
            hi = max(draws.max(), (beta_mean + x_pad * beta_sds).max())
            xs = np.linspace(lo, hi, 1000)

            dens_post = kde(xs)
            dens_true = true_mixture_density(pis_true[s - 1], beta_mean, beta_sds, xs)

            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
            ax.plot(xs, dens_true, color="black", linestyle="--", linewidth=1.2,
                    label="True per-source mixture")
            ax.plot(xs, dens_post, alpha=0.8, label="Posterior β samples (pooled over outcomes)")
            ax.set_title(f"Source {s}: True vs Posterior β density")
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel("Density")
            ax.grid(True); ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"rep{rep}_src{s}.png", bbox_inches="tight")
            plt.close(fig)
