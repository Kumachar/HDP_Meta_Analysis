#!/usr/bin/env python3
"""
Posterior density ribbons vs true population density with a two-patch layout.

- Per-source figure (no subplots):
    Top patch:    posterior density ribbon + mean vs true density
    Bottom patch: blanket bars (truths), μ_s 95% CI band, posterior mean μ_s, true μ_s
  Only the TOP (density) patch shows the x-axis. The blanket patch hides its x-axis.

- Summary:
    Barplot of average density coverage across families and reps.
    CSV with raw coverage metrics.

This script avoids `tight_layout` or constrained/auto layout on patch-based figures,
and ensures directories exist before saving.
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional: ArviZ style if present
try:
    import arviz as az  # type: ignore
except Exception:
    az = None

# You should have this in your codebase; used for the "true population" curve
from utils import true_mixture_density  # f(x) = Σ_j π_j N(x|μ_j, σ_j^2)

_BETA_SO_RE = re.compile(r"^beta_(\d+)_(\d+)\.npz$", re.IGNORECASE)


# -------------------------- Layout helpers (no subplot) --------------------------
def _new_figure(figsize=(7.0, 4.8), dpi=220) -> plt.Figure:
    """
    Create a figure with all automatic layout engines disabled, so manual
    patch axes positions are respected and no tight/constrained layout warnings appear.
    """
    # Ensure global auto-layout is off (styles like ArviZ can turn this on)
    mpl.rcParams['figure.autolayout'] = False
    try:
        plt.rcParams['figure.constrained_layout.use'] = False  # may not exist on older mpl
    except Exception:
        pass

    # Matplotlib >=3.8: layout=None, constrained_layout=False
    try:
        fig = plt.figure(figsize=figsize, dpi=dpi, layout=None, constrained_layout=False)  # type: ignore
    except TypeError:
        # Older Matplotlib
        fig = plt.figure(figsize=figsize, dpi=dpi)
        try:
            fig.set_constrained_layout(False)  # type: ignore
            fig.set_tight_layout(False)        # type: ignore
        except Exception:
            pass
    return fig


def _figure_no_layout(figsize=(7.0, 4.8), dpi=220):
    """Create a figure with layout engines disabled (no tight/constrained layout)."""
    import matplotlib as mpl, matplotlib.pyplot as plt
    mpl.rcParams['figure.autolayout'] = False
    try:
        plt.rcParams['figure.constrained_layout.use'] = False
    except Exception:
        pass
    try:
        return plt.figure(figsize=figsize, dpi=dpi, layout=None, constrained_layout=False)  # mpl>=3.8
    except TypeError:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        try:
            fig.set_constrained_layout(False)
            fig.set_tight_layout(False)
        except Exception:
            pass
        return fig

def _two_patch_axes(fig,
                    *,
                    left=0.10, right=0.98,
                    bottom=0.12, top=0.96,
                    top_height_ratio=0.74, vgap=0.03):
    """
    Create two stacked axes ("patches") without using subplot/gridspec.

    Returns
    -------
    axTop, axBot
    """
    W = right - left
    H = top - bottom
    h_top = H * top_height_ratio
    gap = H * vgap
    h_bot = H - h_top - gap
    axTop = fig.add_axes([left, bottom + h_bot + gap, W, h_top])  # Density coverage (TOP) - shows x-axis
    axBot = fig.add_axes([left, bottom, W, h_bot])                # Blanket / rug (BOTTOM) - hide x-axis
    return axTop, axBot


# -------------------------- Data discovery --------------------------
def discover_families(results_root: Path) -> List[str]:
    fams: List[str] = []
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and (d / "data").exists():
            fams.append(d.name)
    return fams


def list_rep_dirs(family_dir: Path) -> List[Path]:
    d = family_dir / "data"
    if not d.exists():
        return []
    return sorted(p for p in d.glob("rep*") if p.is_dir())


# -------------------------- Math helpers --------------------------
def _normal_pdf(xgrid: np.ndarray, mu_row: np.ndarray, sigma_row: np.ndarray) -> np.ndarray:
    z = (xgrid[:, None] - mu_row[None, :]) / sigma_row[None, :]
    return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma_row[None, :])


def _compute_density_ribbon_for_source(
    xgrid: np.ndarray,
    mu: np.ndarray,         # (n_draws, K)
    sigma: np.ndarray,      # (n_draws, K)
    pi_norm_s: np.ndarray,  # (n_draws, K) for this source
    max_draws: int = 400,
    quantiles: Tuple[float, float, float] = (2.5, 50.0, 97.5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_draws, K = mu.shape
    idx = np.arange(n_draws)
    if n_draws > max_draws:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_draws, size=max_draws, replace=False)
    idx = np.asarray(idx, dtype=int)

    dens_list = []
    for i in idx:
        pdfs = _normal_pdf(xgrid, mu[i], sigma[i])  # (G,K)
        dens = pdfs @ pi_norm_s[i]                  # (G,)
        dens_list.append(dens)
    dens_mat = np.stack(dens_list, axis=0)         # (n_idx, G)

    lo, med, hi = np.percentile(dens_mat, [quantiles[0], 50.0, quantiles[2]], axis=0)
    mean_dens = dens_mat.mean(axis=0)
    return lo, med, hi, mean_dens


def _compute_mu_s_draws(mu: np.ndarray, pi_norm_s: np.ndarray) -> np.ndarray:
    return np.sum(pi_norm_s * mu, axis=1)  # (n_draws,)


def _collect_true_betas_for_source(rep_dir: Path, s: int) -> np.ndarray:
    """
    Return array of true β_{s,o} via NPZ 'prior_samples'.
    """
    vals: List[float] = []
    for f in rep_dir.iterdir():
        m = _BETA_SO_RE.fullmatch(f.name)
        if not m:
            continue
        ss, _ = int(m.group(1)), int(m.group(2))
        if ss != s:
            continue
        try:
            with np.load(f, allow_pickle=True) as data:
                if "prior_samples" in data.files and data["prior_samples"].size > 0:
                    arr = np.asarray(data["prior_samples"]).ravel()
                    vals.append(float(arr.mean()))
        except Exception:
            continue
    return np.asarray(vals, dtype=float)


# -------------------------- Plotting (two-patch, no tight_layout) --------------------------

def plot_one_source(
    out_png: Path,
    xgrid: np.ndarray,
    post_lo: np.ndarray,
    post_med: np.ndarray,
    post_hi: np.ndarray,
    post_mean: np.ndarray,
    true_density: np.ndarray,
    mu_s_ci: Tuple[float, float],
    mu_s_mean: float,
    mu_s_true: float,
    *,
    title: str = "",
    truths_s: Optional[np.ndarray] = None,    # β* 的位置（用于 rug/blanket）
    has_truth: Optional[np.ndarray] = None,   # 可选：bool 掩码，哪些 β* 是“真值”，其余视作 fallback
    xlim: Optional[Tuple[float, float]] = None,
    dpi: int = 220,
    rug_height_frac: float = 0.10,            # 更扁一些：可设 0.06~0.10
    top_ylim_mode: str = "truth",          # 'truth' => stable y-scale by true density
    top_ymax_mult: float = 2.0,                # headroom when top_ylim_mode='truth'
):
    """
    布局：Top(密度覆盖) + Bottom(rug/blanket)，与示例 add_subplot(subgs[0/1]) 风格一致。
    - 仅“下面”显示 x 轴；“上面”隐藏 x tick labels。
    - rug 更扁：通过 rug_height_frac 控制条高（相对轴高的比例）。
    - 只保留一个标题（上面子图）；不显示 “μ covered”。
    """
    fig = plt.figure(figsize=(7.2, 4.9), dpi=dpi)
    subgs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=(3.6, 1.0), hspace=0.05)
    axTop = fig.add_subplot(subgs[0])
    axBot = fig.add_subplot(subgs[1], sharex=axTop)

    # ---- 统一 x 轴数据（如指定 xlim 则窗口内插值）----
    if xlim is not None:
        x = np.linspace(xlim[0], xlim[1], xgrid.size)
        lo = np.interp(x, xgrid, post_lo)
        hi = np.interp(x, xgrid, post_hi)
        mean_d = np.interp(x, xgrid, post_mean)
        truth_d = np.interp(x, xgrid, true_density)
    else:
        x = xgrid
        lo, hi, mean_d, truth_d = post_lo, post_hi, post_mean, true_density

    # 覆盖率（角标文本仅保留这一项）
    inside = (truth_d >= lo) & (truth_d <= hi)
    density_coverage = float(np.mean(inside))

    # =================== Top: Density coverage ===================
    axTop.fill_between(x, lo, hi, alpha=0.22, label="Posterior 95% band")
    axTop.plot(x, mean_d, linewidth=1.4, label="Posterior mean density")
    axTop.fill_between(x, 0.0, truth_d, alpha=0.15, color="black")
    axTop.plot(x, truth_d, color="black", linestyle="--", linewidth=1.2, label="True population density")

    # 左上角角标 —— 只显示 truth-in-band
    axTop.text(
        0.01, 0.97,
        f"truth-in-band: {density_coverage*100:.1f}%",
        transform=axTop.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
    )

    if title:
        axTop.set_title(title)     # ← 仅保留这个标题
    axTop.set_ylabel("Density")

    # Optional: enforce a stable y-axis scale across runs that share the same dataset
    # (e.g., same rep+source but different outcome_cap / number of outcomes).
    # Using the TRUE population density to set the y-limit makes the scale identical
    # whenever the truth mixture is identical.
    if top_ylim_mode.lower() == "truth":
        ymax_truth = float(np.nanmax(truth_d)) if truth_d.size else float('nan')
        if np.isfinite(ymax_truth) and ymax_truth > 0:
            mult = float(top_ymax_mult) if top_ymax_mult is not None else 1.0
            if mult < 1.0:
                mult = 1.0
            axTop.set_ylim(0.0, ymax_truth * mult)
    if xlim is not None:
        axTop.set_xlim(*xlim)
    axTop.grid(True, alpha=0.3)
    axTop.tick_params(labelbottom=False)  # 上面隐藏 x 标签
    axTop.legend(loc="best", fontsize="small")

    # =================== Bottom: Blanket / Rug ===================
    # μ_s 的区间与参考线（不设 label，避免进入 legend）
    axBot.axvspan(mu_s_ci[0], mu_s_ci[1], color="C0", alpha=0.22, zorder=0)
    axBot.axvline(mu_s_mean, color="C0", alpha=0.95, linewidth=1.1, zorder=1)
    axBot.axvline(mu_s_true, color="black", linestyle=":", linewidth=1.2, zorder=1)

    # rug/blanket：扁一些 —— 用较小高度（轴高 0~1 内的一部分）
    if truths_s is not None and truths_s.size:
        span = float(truths_s.max() - truths_s.min()) if truths_s.size > 1 else 1.0
        bar_w = max(0.0125 * span if span > 0 else 0.05, 1e-3)
        rug_h = float(np.clip(rug_height_frac, 0.02, 0.35))
        heights = np.full_like(truths_s, rug_h, dtype=float)

        if has_truth is not None and has_truth.size == truths_s.size:
            if np.any(has_truth):
                axBot.bar(truths_s[has_truth], heights[has_truth],
                          width=bar_w, bottom=0.0, align="center",
                          alpha=0.85, zorder=2)
            if np.any(~has_truth):
                axBot.bar(truths_s[~has_truth], heights[~has_truth],
                          width=bar_w, bottom=0.0, align="center",
                          alpha=0.35, zorder=2)
        else:
            axBot.bar(truths_s, heights, width=bar_w, bottom=0.0,
                      align="center", alpha=0.85, zorder=2)

    axBot.set_ylim(0, 1)
    axBot.set_yticks([])
    axBot.set_xlabel(r"$\beta^*_{s,o}$")  # 只在下面显示 x 轴与标签
    for spine in ("top", "left", "right"):
        axBot.spines[spine].set_visible(False)
    axBot.grid(False)
    if xlim is not None:
        axBot.set_xlim(*xlim)

    # -------- 保存 --------
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])   # 仅 tight_layout；不使用 suptitle
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------- CLI & aggregation --------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Density coverage (top) + blanket rug (bottom) using subplot layout; "
                    "also writes a family-level average coverage barplot."
    )
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Parent results dir containing family subfolders.")
    ap.add_argument("--families", nargs="+", default=None,
                    help="Subset of families; default: auto-discover.")
    ap.add_argument("--reps", nargs="+", type=int, default=None,
                    help="Subset of reps; default: all rep*.")
    ap.add_argument("--outdir-name", type=str, default="source_density_coverage",
                    help="Figures subfolder under each family.")
    ap.add_argument("--max-draws", type=int, default=400,
                    help="Subsample posterior draws per rep for ribbons.")
    ap.add_argument("--grid-len", type=int, default=1600,
                    help="Grid length for plotting.")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--xlim", nargs=2, type=float, default=(-5.0, 5.0),
                    metavar=("LOW", "HIGH"), help="x-axis window [LOW HIGH] (default: -5 5)")
    ap.add_argument("--rug-height-frac", type=float, default=0.10,
                    help="Relative bar height (0-1) for the rug/blanket panel; "
                         "smaller = flatter (e.g., 0.06).")

    ap.add_argument(
        "--top-ylim-mode",
        type=str,
        default="truth",
        choices=["auto", "truth"],
        help=(
            "Top density-panel y-axis scaling. "
            "'truth' fixes ymax based on the true population density (stable across runs that share truth). "
            "'auto' lets matplotlib auto-scale (may differ across outcome counts)."
        ),
    )
    ap.add_argument(
        "--top-ymax-mult",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to max(true_density) when --top-ylim-mode=truth. "
            "Increase if the posterior band clips; keep the same value across runs to maintain comparability."
        ),
    )
    args = ap.parse_args()

    root = args.results_root.resolve()
    families = args.families or discover_families(root)
    if not families:
        raise SystemExit(f"No families found under {root}")

    coverage_records: list[dict] = []

    for fam in families:
        fam_dir = root / fam
        rep_dirs = list_rep_dirs(fam_dir)
        if args.reps:
            rep_dirs = [fam_dir / "data" / f"rep{r}" for r in args.reps]
        # keep only existing rep dirs
        rep_dirs = [p for p in rep_dirs if p.is_dir()]
        if not rep_dirs:
            print(f"[warn] {fam}: no reps found to process")
            continue

        out_family = fam_dir / "figures" / args.outdir_name

        for rep_dir in rep_dirs:
            g0_npz = rep_dir / "g0_components.npz"
            truth_npz = rep_dir / "truth_source_mixtures.npz"
            if not g0_npz.exists() or not truth_npz.exists():
                print(f"[skip] {fam}/{rep_dir.name}: missing g0_components.npz or truth_source_mixtures.npz")
                continue

            # --- posterior objects from g0_components (robustify shapes) ---
            dat = np.load(g0_npz, allow_pickle=True)
            mu = np.asarray(dat["mu"])        # (n_draws, K) or (K,)
            sigma = np.asarray(dat["sigma"])  # (n_draws, K) or (K,)
            pi_norm = np.asarray(dat["pi_norm"])  # (n_draws, S, K) or (S,K)

            if mu.ndim == 1:
                mu = mu[None, :]             # -> (1, K)
            if sigma.ndim == 1:
                sigma = sigma[None, :]       # -> (1, K)
            if pi_norm.ndim == 2:
                pi_norm = pi_norm[None, :, :]  # -> (1, S, K)

            n_draws, S, K = pi_norm.shape[0], pi_norm.shape[1], mu.shape[-1]

            # --- truths for analytic mixture + μ_s^true ---
            truth = np.load(truth_npz)
            beta_mean_true = np.asarray(truth["beta_mean"]).ravel()   # (K,)
            beta_sds_true  = np.asarray(truth["beta_sds"]).ravel()    # (K,)
            pis_true       = np.asarray(truth["pis"])                 # (S, K)

            # uniform grid over requested xlim for consistent resolution
            xgrid = np.linspace(args.xlim[0], args.xlim[1], args.grid_len)

            # parse rep index from folder name "rep{n}"
            try:
                rep_idx = int(rep_dir.name.replace("rep", ""))
            except Exception:
                rep_idx = None

            # ---- per-source figures ----
            for s in range(1, S + 1):
                pi_s_draws = pi_norm[:, s - 1, :]  # (n_draws, K)

                # posterior ribbon over x
                lo, med, hi, mean_dens = _compute_density_ribbon_for_source(
                    xgrid, mu, sigma, pi_s_draws, max_draws=args.max_draws
                )

                # posterior μ_s CI + mean
                mu_s_draws = _compute_mu_s_draws(mu, pi_s_draws)
                mu_s_ci = (float(np.percentile(mu_s_draws, 2.5)),
                           float(np.percentile(mu_s_draws, 97.5)))
                mu_s_mean = float(mu_s_draws.mean())

                # true population density (per-source proportions with true component params)
                pi_true_s = pis_true[s - 1]  # (K,)
                true_dens = true_mixture_density(pi_true_s, beta_mean_true, beta_sds_true, xgrid)
                mu_s_true = float(np.sum(pi_true_s * beta_mean_true))

                # coverage stats for aggregation
                inside = (true_dens >= lo) & (true_dens <= hi)
                density_cov = float(np.mean(inside))
                mu_cov = bool(mu_s_ci[0] <= mu_s_true <= mu_s_ci[1])

                coverage_records.append(
                    {
                        "family": fam,
                        "rep": rep_idx,
                        "source": s,
                        "density_coverage": density_cov,
                        "mu_covered": mu_cov,
                    }
                )

                # optional rug from per-outcome truths (posterior priors saved per (s,o))
                truths_s = _collect_true_betas_for_source(rep_dir, s)

                out_png = out_family / f"rep{rep_idx}" / f"src{s}.png"
                title = f"{fam} • rep {rep_idx} • source {s}"
                # ⬇️ This plot function should be your subplot version with a flatter rug
                plot_one_source(
                    out_png,
                    xgrid,
                    lo,
                    med,
                    hi,
                    mean_dens,
                    true_dens,
                    mu_s_ci,
                    mu_s_mean,
                    mu_s_true,
                    title=title,
                    truths_s=truths_s,
                    xlim=tuple(args.xlim),
                    dpi=args.dpi,
                    # mimic your layout: top hides x labels, bottom shows x labels; flatter rug:
                    rug_height_frac=args.rug_height_frac,
                    top_ylim_mode=args.top_ylim_mode,
                    top_ymax_mult=args.top_ymax_mult,
                )
                print(f"[OK] wrote {out_png}")

    # ------------------------ Average density coverage barplot ------------------------
    if coverage_records:
        cov_sum_by_family: dict[str, float] = {}
        count_by_family: dict[str, int] = {}
        for rec in coverage_records:
            fam = rec["family"]
            cov_sum_by_family[fam] = cov_sum_by_family.get(fam, 0.0) + rec["density_coverage"]
            count_by_family[fam] = count_by_family.get(fam, 0) + 1

        fam_order = [f for f in families if f in cov_sum_by_family] or sorted(cov_sum_by_family.keys())
        avg_cov = [cov_sum_by_family[f] / count_by_family[f] for f in fam_order]

        summary_dir = root / "summary_figs"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Use standard subplots (safe with tight_layout)
        fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=args.dpi)
        ax.bar(fam_order, avg_cov, alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Average density coverage")
        ax.set_xlabel("Model family")
        ax.set_title("Average true-density coverage (95% posterior band)")
        ax.grid(axis="y", alpha=0.3)
        for i, val in enumerate(avg_cov):
            ax.text(i, min(val + 0.02, 0.98), f"{val * 100.0:.1f}%",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()

        bar_png = summary_dir / "density_coverage_barplot.png"
        fig.savefig(str(bar_png), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[OK] wrote barplot {bar_png}")

        # Raw records CSV
        csv_path = summary_dir / "density_coverage_records.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["family", "rep", "source", "density_coverage", "mu_covered"])
            for rec in coverage_records:
                w.writerow(
                    [
                        rec["family"],
                        rec["rep"],
                        rec["source"],
                        f"{rec['density_coverage']:.6f}",
                        int(bool(rec["mu_covered"])),
                    ]
                )
        print(f"[OK] wrote raw coverage records to {csv_path}")


if __name__ == "__main__":
    main()

