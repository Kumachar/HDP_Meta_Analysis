#!/usr/bin/env python3
"""
plot_ci_truth.py

What it makes:
1) Boxplot of 95% CI lengths by *method* (model family) for:
   • per-source β_s (from post_ci_[lower|upper]) when available
   • per-outcome β_{s,o} (from beta_so_ci_[lower|upper]) when available

2) Boxplot of the distribution of true values β* (per-outcome) by method.
   Truths are read from per-outcome NPZ bundles written by utils.store_beta_posteriors()
   (we use the 'prior_samples' field which the pipeline fills from df_betas['beta_true']).

3) A “forest” plot for a single repetition showing, for each outcome, the 0.95 CI
   and the true β* point. Facets = sources. One PNG per family.

Expected results layout:
  {results_root}/{family}/data/rep{r}/
      beta_1_1.npz, beta_1_2.npz, ...   (each contains arrays: samples, grid, loglik, prior_samples)
      beta_summary_stats.csv            (rep-level summary with per-source/posterior CIs)
  {results_root}/{family}/beta_summary_stats.csv     (family-level, aggregated across reps)

References for CSV columns & NPZ structure:
  • summarize_results.py aggregates rep CSVs and keeps per-outcome CI columns when present.
  • utils.store_beta_posteriors() writes the per-outcome NPZ files and includes 'prior_samples'
    derived from df_betas['beta_true'] for each (s,o).

Usage:
  python plot_ci_truth.py --results-root results_YYYY-MM-DD_allmodels [--families linear poisson logistic] [--rep 1]

Outputs:
  {results_root}/summary_figs_ci/
      ci_len_beta_so_box_by_model.png
      ci_len_beta_s_box_by_model.png                (if per-source CIs are present)
      beta_true_box_by_model.png
      {family}_rep{rep}_beta_so_forest.png          (one per family if --rep is provided)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

# -----------------------------
# Helpers
# -----------------------------

_BETA_SO_RE = re.compile(r"^beta_(\d+)_(\d+)\.npz$", re.IGNORECASE)

def discover_families(results_root: Path) -> List[str]:
    fams: List[str] = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "data").exists():
            fams.append(d.name)
    return fams

def newest_family_csv(model_dir: Path) -> Optional[Path]:
    csv = model_dir / "beta_summary_stats.csv"
    return csv if csv.exists() else None

def list_rep_dirs(family_dir: Path) -> List[Path]:
    d = family_dir / "data"
    if not d.exists():
        return []
    return sorted(p for p in d.glob("rep*") if p.is_dir())

def scan_beta_npz_truths(rep_dir: Path) -> Dict[Tuple[int,int], float]:
    """
    Return mapping (s,o) -> true β* from NPZ bundles in rep_dir.
    We use the 'prior_samples' field as written by utils.store_beta_posteriors().
    """
    truths: Dict[Tuple[int,int], float] = {}
    for f in rep_dir.iterdir():
        m = _BETA_SO_RE.fullmatch(f.name)
        if not m:
            continue
        s, o = int(m.group(1)), int(m.group(2))
        try:
            with np.load(f, allow_pickle=True) as data:
                if "prior_samples" in data.files and data["prior_samples"].size > 0:
                    truths[(s, o)] = float(np.mean(np.asarray(data["prior_samples"]).ravel()))
        except Exception:
            continue
    return truths

def scan_beta_npz_ci(rep_dir: Path) -> Dict[Tuple[int,int], Tuple[float, float]]:
    """
    If the family CSV lacks per-outcome CI bounds, compute them from NPZ.posterior 'samples'.
    Returns mapping (s,o) -> (lo, hi).
    """
    ci: Dict[Tuple[int,int], Tuple[float, float]] = {}
    for f in rep_dir.iterdir():
        m = _BETA_SO_RE.fullmatch(f.name)
        if not m:
            continue
        s, o = int(m.group(1)), int(m.group(2))
        try:
            with np.load(f, allow_pickle=True) as data:
                if "samples" in data.files and data["samples"].size > 0:
                    draws = np.asarray(data["samples"]).ravel()
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    ci[(s, o)] = (float(lo), float(hi))
        except Exception:
            continue
    return ci


# -----------------------------
# Plots
# -----------------------------

def boxplot_ci_lengths(stats_all: pd.DataFrame, out_dir: Path) -> None:
    """
    Make boxplots of CI lengths by family:
      - per-outcome beta_so_ci_upper - beta_so_ci_lower
      - per-source post_ci_upper - post_ci_lower (if present)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    made_any = False

    # per-outcome CI length
    if {"beta_so_ci_lower", "beta_so_ci_upper", "model"}.issubset(stats_all.columns):
        df = stats_all.dropna(subset=["beta_so_ci_lower", "beta_so_ci_upper"]).copy()
        if not df.empty:
            df["ci_len_so"] = df["beta_so_ci_upper"] - df["beta_so_ci_lower"]
            g = sns.catplot(
                data=df, x="model", y="ci_len_so",
                kind="box", showfliers=False, height=5, aspect=1.4
            )
            g.set_axis_labels("Method (model family)", "CI length for $\\beta_{s,o}$ (0.95)")
            g.fig.suptitle("Distribution of 0.95 CI lengths (per-outcome) by method")
            g.tight_layout()
            g.savefig(out_dir / "ci_len_beta_so_box_by_model.png", dpi=300)
            plt.close(g.fig)
            made_any = True

    # per-source CI length (optional)
    if {"post_ci_lower", "post_ci_upper", "model"}.issubset(stats_all.columns):
        df = stats_all.dropna(subset=["post_ci_lower", "post_ci_upper"]).copy()
        if not df.empty:
            df["ci_len_s"] = df["post_ci_upper"] - df["post_ci_lower"]
            g = sns.catplot(
                data=df, x="model", y="ci_len_s",
                kind="box", showfliers=False, height=5, aspect=1.4
            )
            g.set_axis_labels("Method (model family)", "CI length for $\\beta_s$ (0.95)")
            g.fig.suptitle("Distribution of 0.95 CI lengths (per-source) by method")
            g.tight_layout()
            g.savefig(out_dir / "ci_len_beta_s_box_by_model.png", dpi=300)
            plt.close(g.fig)
            made_any = True

    if not made_any:
        print("[WARN] No CI columns found to plot; expected beta_so_ci_* and/or post_ci_*")

def boxplot_truth_by_method(results_root: Path, families: List[str], out_dir: Path) -> None:
    """
    Scan NPZ bundles under each family/data/rep*/ to collect true β* (per-outcome),
    then plot a boxplot grouped by method (family).
    """
    rows = []
    for fam in families:
        fam_dir = results_root / fam
        for rep_dir in list_rep_dirs(fam_dir):
            truths = scan_beta_npz_truths(rep_dir)
            for (s, o), b in truths.items():
                rows.append({"model": fam, "rep": rep_dir.name, "source": s, "outcome": o, "beta_true": b})
    if not rows:
        print("[WARN] No truths recovered from NPZs; skipping beta_true boxplot.")
        return

    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, x="model", y="beta_true",
        kind="box", showfliers=False, height=5, aspect=1.4
    )
    g.set_axis_labels("Method (model family)", r"True $\beta^{*}$ (per-outcome)")
    g.fig.suptitle("Distribution of true $\\beta^{*}$ by method")
    g.tight_layout()
    g.savefig(out_dir / "beta_true_box_by_model.png", dpi=300)
    plt.close(g.fig)

def forest_per_outcome_for_rep(results_root: Path, family: str, rep: int, out_dir: Path) -> Optional[Path]:
    """
    Two-panel per SOURCE:
      • Left: residual forest: (post_mean − true) with 95% CI shifted by truth.
      • Right: top = KDE of true β_{s,o}; bottom = a small 'blanket' strip with
               thin bars indicating each true β_{s,o} location (no overlay on KDE).

    Notes:
      • We enumerate outcomes from per-outcome NPZs (beta_<s>_<o>.npz) to avoid
        accidental extra bars from CSV rows with NaN outcome. Truths are taken
        from 'prior_samples' saved by utils.store_beta_posteriors(). :contentReference[oaicite:1]{index=1}
    """
    fam_dir = results_root / family
    rep_dir = fam_dir / "data" / f"rep{rep}"
    if not rep_dir.exists():
        print(f"[WARN] {family}: missing {rep_dir}")
        return None

    # ---- Build tidy frame from NPZs only
    rows = []
    for f in rep_dir.iterdir():
        m = _BETA_SO_RE.fullmatch(f.name)
        if not m:
            continue
        s, o = int(m.group(1)), int(m.group(2))
        try:
            with np.load(f, allow_pickle=True) as data:
                if "samples" not in data.files or data["samples"].size == 0:
                    continue
                draws = np.asarray(data["samples"]).ravel()
                lo, hi = np.percentile(draws, [2.5, 97.5])
                post_mean = float(draws.mean())

                if "prior_samples" in data.files and data["prior_samples"].size > 0:
                    b_true = float(np.mean(np.asarray(data["prior_samples"]).ravel()))
                else:
                    b_true = np.nan

                if np.isfinite(b_true):
                    rows.append({
                        "source": s, "outcome": o,
                        "lo": float(lo), "hi": float(hi),
                        "post_mean": post_mean,
                        "beta_true": b_true
                    })
        except Exception:
            continue

    if not rows:
        print(f"[WARN] {family}/rep{rep}: no per-outcome rows with (truth, CI, mean).")
        return None

    dfp = (pd.DataFrame(rows)
             .dropna(subset=["outcome"])
             .drop_duplicates(subset=["source", "outcome"])
             .sort_values(["source", "outcome"])
             .reset_index(drop=True))

    # ---- Figure: rows = sources; 2 columns (left residuals, right density+blanket)
    srcs  = sorted(dfp["source"].unique())
    nrows = len(srcs)

    fig = plt.figure(figsize=(11.0, 3.1 * nrows))
    gs  = gridspec.GridSpec(nrows=nrows, ncols=2, figure=fig,
                            width_ratios=[1.25, 1.5], wspace=0.28, hspace=0.55)

    for i, s in enumerate(srcs):
        sub = dfp[dfp["source"] == s].sort_values("outcome").copy()
        M   = len(sub)
        xs  = np.arange(1, M + 1)

        # ---------- Left: residual forest (post − true), CI shifted by truth
        axL = fig.add_subplot(gs[i, 0])
        resid_lo = sub["lo"].to_numpy() - sub["beta_true"].to_numpy()
        resid_hi = sub["hi"].to_numpy() - sub["beta_true"].to_numpy()
        resid_pt = sub["post_mean"].to_numpy() - sub["beta_true"].to_numpy()

        axL.axhline(0.0, color="k", linewidth=1, alpha=0.5)
        axL.vlines(xs, resid_lo, resid_hi, linewidth=2)
        axL.scatter(xs, resid_pt, s=18, zorder=3)
        axL.set_title(f"Source {s}: $\\hat\\beta_{{s,o}}-\\beta^*_{{s,o}}$")
        axL.set_xlabel("Outcome")
        axL.set_ylabel("Residual")
        axL.grid(True, alpha=0.3)

        # ---------- Right: KDE (top) + blanket bars (bottom)
        truths_s = sub["beta_true"].to_numpy()

        subgs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[i, 1], height_ratios=[4.0, 0.8], hspace=0.06
        )
        axTop = fig.add_subplot(subgs[0])
        axBot = fig.add_subplot(subgs[1], sharex=axTop)

        # KDE of truths (no markers on this axis)
        if truths_s.size >= 2 and np.std(truths_s) > 0:
            kde = gaussian_kde(truths_s)
            lo, hi = truths_s.min(), truths_s.max()
            pad = 0.10 * (hi - lo + 1e-9)
            grid = np.linspace(lo - pad, hi + pad, 512)
            dens = kde(grid)
            axTop.plot(grid, dens)
            axTop.set_ylim(bottom=0, top=float(dens.max()) * 1.1)
        else:
            grid = np.linspace(truths_s.min() - 1e-3, truths_s.max() + 1e-3, 10)
            dens = np.zeros_like(grid)
            axTop.plot(grid, dens)
            axTop.set_ylim(0, 1)

        axTop.set_title(f"Source {s}: distribution of true $\\beta_{{s,o}}$")
        axTop.set_ylabel("Density")
        axTop.grid(True, alpha=0.3)
        axTop.tick_params(labelbottom=False)

        # Blanket strip of narrow bars at each truth location (separate axis)
        x_span = float((truths_s.max() - truths_s.min()) if truths_s.size > 1 else 1.0)
        bar_w  = 0.0125 * x_span if x_span > 0 else 0.05
        bar_w  = max(bar_w, 1e-3)

        axBot.bar(truths_s, np.ones_like(truths_s),
                  width=bar_w, bottom=0.0, align="center", alpha=0.85)
        axBot.set_ylim(0, 1)
        axBot.set_yticks([])
        for spine in ("top", "left", "right"):
            axBot.spines[spine].set_visible(False)
        axBot.set_xlabel(r"$\beta^*_{s,o}$")
        axBot.grid(False)

    fig.suptitle(f"{family} • rep {rep}: residual forest & true $\\beta_{{s,o}}$ density + locations",
                 y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{family}_rep{rep}_beta_so_forest.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path



# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot CI-length boxplots and per-outcome truth vs 0.95 CI.")
    ap.add_argument("--results-root", type=Path, required=True, help="Parent results dir that contains family subfolders.")
    ap.add_argument("--families", nargs="+", default=None, help="Subset of families (e.g., linear poisson logistic). Default: auto-discover.")
    ap.add_argument("--rep", type=int, default=None, help="If set, draw per-outcome forest plots for this repetition index.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write figures (default: {results_root}/summary_figs_ci)")
    args = ap.parse_args()

    root = args.results_root
    families = args.families or discover_families(root)
    if not families:
        raise SystemExit(f"No model families found under {root}")

    out_dir = args.out_dir or (root / "summary_figs_ci")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & combine family-level CSVs (like plot_results.py)
    frames = []
    for fam in families:
        csv = newest_family_csv(root / fam)
        if csv is None:
            print(f"[warn] No family CSV in {root/fam}; skipping family.")
            continue
        df = pd.read_csv(csv)
        df["model"] = fam
        frames.append(df)
    if not frames:
        raise SystemExit("No family-level CSVs found; aborting. Run summarize_results.py first.")

    stats_all = pd.concat(frames, ignore_index=True, sort=False)
    # --- 1) CI-length boxplots ---
    boxplot_ci_lengths(stats_all, out_dir)

    # --- 2) Truth distribution boxplot ---
    boxplot_truth_by_method(root, families, out_dir)

    # --- 3) Per-outcome forest plots for one rep ---
    if args.rep is not None:
        for fam in families:
            p = forest_per_outcome_for_rep(root, fam, args.rep, out_dir)
            if p is not None:
                print(f"[OK] wrote {p}")

if __name__ == "__main__":
    main()
