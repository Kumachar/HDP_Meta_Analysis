#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robustly coerce coverage column to boolean."""
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float) != 0.0
    # strings like "True"/"False"/"1"/"0"
    sl = s.astype(str).str.lower()
    return sl.isin(["true", "1", "yes", "y"])


def build_poisson_beta_so_fail_summary(
    results_root: Path,
    *,
    out_csv_name: str = "beta_so_fail_summary.csv",
    ci_cols: tuple[str, str] = ("beta_so_ci_lower", "beta_so_ci_upper"),
) -> Path | None:
    """
    Create a summary table for all Poisson (rep, source, outcome) pairs where
    the 0.95 CI fails to cover the true β_{s,o}.

    Outputs:
      • {results_root}/poisson/beta_so_fail_summary.csv
      • Histograms under {results_root}/poisson/figures/diagnosis_plot/
    """
    fam_dir = results_root / "poisson"
    agg_csv = fam_dir / "beta_summary_stats.csv"  # family-level aggregated CSV
    if not agg_csv.exists():
        raise FileNotFoundError(f"Aggregated CSV not found: {agg_csv}")

    df = pd.read_csv(agg_csv)
    if "beta_so_cover_95" not in df.columns:
        print("[WARN] Column 'beta_so_cover_95' not present; nothing to summarize.")
        return None

    # Keep only per-outcome rows that have coverage info
    df_cov = df.dropna(subset=["beta_so_cover_95"]).copy()
    if df_cov.empty:
        print("[INFO] No per-outcome coverage rows found; nothing to do.")
        return None

    cov_bool = _to_bool_series(df_cov["beta_so_cover_95"])
    df_fail = df_cov.loc[~cov_bool].copy()
    if df_fail.empty:
        print("[INFO] No failures (all β_{s,o} covered truth).")
        # Still write an empty CSV for completeness
        out_csv = fam_dir / out_csv_name
        df_fail.to_csv(out_csv, index=False)
        return out_csv

    ci_lower_col, ci_upper_col = ci_cols
    have_ci_bounds = (ci_lower_col in df_fail.columns) and (ci_upper_col in df_fail.columns)

    # Prepare output rows
    rows = []
    total = len(df_fail)
    print(f"[INFO] Found {total} failing (rep, source, outcome) rows.")

    for _, r in df_fail.iterrows():
        try:
            rep = int(r["rep"])
            s = int(r["source"])
            o = int(r["outcome"])
        except Exception:
            # skip any malformed rows (e.g., source/outcome missing)
            continue

        rep_dir = fam_dir / "data" / f"rep{rep}"
        npz_path = rep_dir / f"beta_{s}_{o}.npz"
        if not npz_path.exists():
            print(f"[WARN] Missing NPZ: {npz_path}; skipping.")
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            samples = np.asarray(data["samples"]).ravel()
            # truth from 'prior_samples' that store_beta_posteriors wrote
            if "prior_samples" in data.files and data["prior_samples"].size > 0:
                b_true = float(np.asarray(data["prior_samples"]).ravel().mean())
            else:
                b_true = np.nan  # fallback if not present

        # Posterior point summaries
        post_mean = float(samples.mean())
        post_median = float(np.median(samples))
        # CI (use CSV bounds if present, else recompute)
        if have_ci_bounds and pd.notna(r[ci_lower_col]) and pd.notna(r[ci_upper_col]):
            lo = float(r[ci_lower_col])
            hi = float(r[ci_upper_col])
        else:
            lo, hi = [float(x) for x in np.percentile(samples, [2.5, 97.5])]
        ci_len = hi - lo

        # Errors relative to truth
        err_mean = (post_mean - b_true) if np.isfinite(b_true) else np.nan
        err_median = (post_median - b_true) if np.isfinite(b_true) else np.nan

        rows.append(
            {
                "model": r.get("model", "poisson"),
                "rep": rep,
                "source": s,
                "outcome": o,
                "beta_so_post_mean": post_mean,
                "beta_so_post_median": post_median,
                "beta_so_error_mean": err_mean,
                "beta_so_error_median": err_median,
                "beta_so_ci_lower": lo,
                "beta_so_ci_upper": hi,
                "beta_so_ci_length": ci_len,
                "beta_so_cover_95": False,  # by construction
            }
        )

    if not rows:
        print("[INFO] No rows materialized (missing NPZs?); nothing to write.")
        return None

    out_df = pd.DataFrame.from_records(rows)
    out_csv = fam_dir / out_csv_name
    _ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote fail summary: {out_csv}  (rows={len(out_df)})")

    # ---- Plots ----
    fig_dir = fam_dir / "figures" / "diagnosis_plot"
    _ensure_dir(fig_dir)

    # Extract arrays
    ci_len_vals = out_df["beta_so_ci_length"].dropna().to_numpy()
    err_mean_vals = out_df["beta_so_error_mean"].dropna().to_numpy()
    err_median_vals = out_df["beta_so_error_median"].dropna().to_numpy()

    # Helper: robust KDE (adds tiny jitter if variance ~ 0)
    def _safe_kde(x: np.ndarray):
        x = np.asarray(x).ravel()
        if x.size < 2 or np.std(x) == 0:
            # fake small spread to avoid singular matrix
            x = x + np.random.normal(0.0, 1e-9, size=x.shape)
        try:
            return gaussian_kde(x)
        except Exception:
            eps = max(1e-9, 1e-6 * (np.std(x) if np.std(x) > 0 else 1.0))
            return gaussian_kde(x + np.random.normal(0.0, eps, size=x.shape))

    # -------------------------------
    # Histogram + KDE: MEAN error
    # -------------------------------
    if err_mean_vals.size > 0:
        plt.figure(figsize=(6.2, 4.2), dpi=180)
        # histogram (density)
        plt.hist(err_mean_vals, bins=20, range=(-1, 1), density=True, alpha=0.35, label="Histogram")
        # KDE
        kde = _safe_kde(err_mean_vals)
        xs = np.linspace(-1, 1, 512)
        plt.plot(xs, kde(xs), linewidth=1.8, label="KDE")
        plt.xlim(-1, 1)
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel(r"Mean error (post mean $-$ true $\beta_{s,o}$)")
        plt.ylabel("Density")
        plt.title(r"Histogram + KDE of $\beta_{s,o}$ mean error (failures only)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = fig_dir / "beta_so_mean_error_hist_kde.png"
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"[OK] Wrote {out_png}")

    # ---------------------------------
    # Histogram + KDE: MEDIAN error
    # ---------------------------------
    if err_median_vals.size > 0:
        plt.figure(figsize=(6.2, 4.2), dpi=180)
        plt.hist(err_median_vals, bins=20, range=(-1, 1), density=True, alpha=0.35, label="Histogram")
        kde = _safe_kde(err_median_vals)
        xs = np.linspace(-1, 1, 512)
        plt.plot(xs, kde(xs), linewidth=1.8, label="KDE")
        plt.xlim(-1, 1)
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel(r"Median error (post median $-$ true $\beta_{s,o}$)")
        plt.ylabel("Density")
        plt.title(r"Histogram + KDE of $\beta_{s,o}$ median error (failures only)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = fig_dir / "beta_so_median_error_hist_kde.png"
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"[OK] Wrote {out_png}")

    # -------------------------------
    # Boxplot: mean vs median error
    # -------------------------------
    if (err_mean_vals.size > 0) or (err_median_vals.size > 0):
        plt.figure(figsize=(6.2, 4.2), dpi=180)
        data = []
        labels = []
        if err_mean_vals.size > 0:
            data.append(err_mean_vals)
            labels.append("Mean error")
        if err_median_vals.size > 0:
            data.append(err_median_vals)
            labels.append("Median error")
        bp = plt.boxplot(data, labels=labels, vert=True, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_alpha(0.6)
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
        plt.ylabel(r"Error ($\widehat{\beta}_{s,o}-\beta^{*}_{s,o}$)")
        plt.title(r"Boxplots of $\beta_{s,o}$ error (failures only)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_png = fig_dir / "beta_so_error_boxplot.png"
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"[OK] Wrote {out_png}")

    # -------------------------------
    # Boxplot: CI length
    # -------------------------------
    if ci_len_vals.size > 0:
        plt.figure(figsize=(4.8, 4.2), dpi=180)
        bp = plt.boxplot([ci_len_vals], labels=["CI length"], vert=True, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_alpha(0.6)
        plt.ylabel("CI length (97.5% − 2.5%)")
        plt.title(r"Boxplot of $\beta_{s,o}$ CI length (failures only)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_png = fig_dir / "beta_so_ci_length_boxplot.png"
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"[OK] Wrote {out_png}")

    return out_csv




def main():
    ap = argparse.ArgumentParser(
        description="Summarize Poisson β_{s,o} failures (CI does not cover truth) and plot histograms."
    )
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Root results folder (parent of 'poisson' family).")
    ap.add_argument("--out-csv", default="beta_so_fail_summary.csv",
                    help="Output CSV filename under the Poisson family folder.")
    args = ap.parse_args()

    build_poisson_beta_so_fail_summary(
        results_root=args.results_root,
        out_csv_name=args.out_csv,
    )


if __name__ == "__main__":
    main()
