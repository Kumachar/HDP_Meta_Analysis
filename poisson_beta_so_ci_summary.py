#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Coerce coverage column to boolean robustly."""
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float) != 0.0
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def _load_npz_posterior(npz_path: Path) -> Tuple[np.ndarray, float | None]:
    """
    Return (samples, true_beta) from a beta_{s}_{o}.npz.
    True value is taken from mean(prior_samples) if present, else None.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        samples = np.asarray(data["samples"]).ravel()
        if "prior_samples" in data.files and data["prior_samples"].size > 0:
            b_true = float(np.asarray(data["prior_samples"]).ravel().mean())
        else:
            b_true = None
    return samples, b_true


def build_poisson_beta_so_ci_summary(
    results_root: Path,
    *,
    out_combined_csv: str = "beta_so_ci_summary.csv",
    out_fail_csv: str     = "beta_so_fail_summary.csv",
    out_cover_csv: str    = "beta_so_covered_summary.csv",
    ci_cols: Tuple[str, str] = ("beta_so_ci_lower", "beta_so_ci_upper"),
) -> Path:
    """
    Build combined summary (covered + uncovered) for Poisson β_{s,o}.

    Outputs:
      • {results_root}/poisson/beta_so_ci_summary.csv (combined)
      • {results_root}/poisson/beta_so_fail_summary.csv (uncovered only)
      • {results_root}/poisson/beta_so_covered_summary.csv (covered only)
      • Plots into {results_root}/poisson/figures/diagnosis_plot/
          - beta_so_ci_length_hist_overlay.png
          - beta_so_ci_length_boxplot_by_coverage.png
    """
    fam_dir = results_root / "poisson"
    agg_csv = fam_dir / "beta_summary_stats.csv"
    if not agg_csv.exists():
        raise FileNotFoundError(f"Aggregated CSV not found: {agg_csv}")

    df = pd.read_csv(agg_csv)

    if "beta_so_cover_95" not in df.columns:
        raise ValueError("Column 'beta_so_cover_95' not present. Re-run summaries with per-outcome coverage.")

    per_outcome = df.dropna(subset=["beta_so_cover_95"]).copy()
    if per_outcome.empty:
        raise ValueError("No per-outcome rows with coverage info found.")

    cov_bool = _to_bool_series(per_outcome["beta_so_cover_95"])
    per_outcome["covered"] = cov_bool.values.astype(bool)

    lo_col, hi_col = ci_cols
    have_ci = (lo_col in per_outcome.columns) and (hi_col in per_outcome.columns)

    rows = []

    for idx, r in per_outcome.iterrows():
        try:
            rep = int(r["rep"]); s = int(r["source"]); o = int(r["outcome"])
        except Exception:
            continue

        rep_dir  = fam_dir / "data" / f"rep{rep}"
        npz_path = rep_dir / f"beta_{s}_{o}.npz"
        if not npz_path.exists():
            # silently skip missing NPZs
            continue

        samples, b_true = _load_npz_posterior(npz_path)
        post_mean   = float(samples.mean())
        post_median = float(np.median(samples))

        if have_ci and pd.notna(r[lo_col]) and pd.notna(r[hi_col]):
            lo = float(r[lo_col]); hi = float(r[hi_col])
        else:
            lo, hi = [float(x) for x in np.percentile(samples, [2.5, 97.5])]
        ci_len = hi - lo

        # errors (allow NaN b_true if unavailable)
        err_mean   = (post_mean   - b_true) if b_true is not None and np.isfinite(b_true) else np.nan
        err_median = (post_median - b_true) if b_true is not None and np.isfinite(b_true) else np.nan

        rows.append({
            "model":      r.get("model", "poisson"),
            "rep":        rep,
            "source":     s,
            "outcome":    o,
            "covered":    bool(r["covered"]),
            "beta_so_post_mean":    post_mean,
            "beta_so_post_median":  post_median,
            "beta_so_error_mean":   err_mean,
            "beta_so_error_median": err_median,
            "beta_so_ci_lower":     lo,
            "beta_so_ci_upper":     hi,
            "beta_so_ci_length":    ci_len,
            "true_beta_so":         b_true,
        })

    if not rows:
        raise RuntimeError("No rows materialized (NPZs missing?).")

    out_df = pd.DataFrame.from_records(rows)

    # Write CSVs
    _ensure_dir(fam_dir)
    combined_csv = fam_dir / out_combined_csv
    out_df.to_csv(combined_csv, index=False)

    fail_csv = fam_dir / out_fail_csv
    out_df.loc[out_df["covered"] == False].to_csv(fail_csv, index=False)

    cover_csv = fam_dir / out_cover_csv
    out_df.loc[out_df["covered"] == True].to_csv(cover_csv, index=False)

    print(f"[OK] wrote combined: {combined_csv} (n={len(out_df)})")
    print(f"[OK] wrote failures: {fail_csv}   (n={(~out_df['covered']).sum()})")
    print(f"[OK] wrote covered:  {cover_csv}  (n={(out_df['covered']).sum()})")

    # ---------- Plots (covered vs uncovered together) ----------
    fig_dir = fam_dir / "figures" / "diagnosis_plot"
    _ensure_dir(fig_dir)

    # 1) Overlay histogram + KDE of CI length by coverage
    from scipy.stats import gaussian_kde

    def _safe_kde(x: np.ndarray):
        x = np.asarray(x).ravel()
        if x.size < 2 or np.std(x) == 0:
            x = x + np.random.normal(0, 1e-9, size=x.shape)
        try:
            return gaussian_kde(x)
        except Exception:
            eps = max(1e-9, 1e-6 * (np.std(x) if np.std(x) > 0 else 1.0))
            return gaussian_kde(x + np.random.normal(0, eps, size=x.shape))

    ci_cov = out_df.loc[out_df["covered"] == True,  "beta_so_ci_length"].dropna().to_numpy()
    ci_miss= out_df.loc[out_df["covered"] == False, "beta_so_ci_length"].dropna().to_numpy()

    if ci_cov.size > 0 or ci_miss.size > 0:
        plt.figure(figsize=(6.8, 4.4), dpi=180)
        # Use same bin edges for comparability
        all_ci = np.concatenate([ci_cov, ci_miss]) if ci_cov.size and ci_miss.size else (ci_cov if ci_cov.size else ci_miss)
        if all_ci.size == 0:
            bins = 20
        else:
            bins = 20
        # Covered
        if ci_cov.size > 0:
            plt.hist(ci_cov, bins=bins, density=True, alpha=0.35, label="Covered")
            kde_cov = _safe_kde(ci_cov); xs = np.linspace(all_ci.min(), all_ci.max(), 512)
            plt.plot(xs, kde_cov(xs), linewidth=1.6)
        # Uncovered
        if ci_miss.size > 0:
            plt.hist(ci_miss, bins=bins, density=True, alpha=0.35, label="Uncovered")
            kde_miss = _safe_kde(ci_miss); xs2 = np.linspace(all_ci.min(), all_ci.max(), 512)
            plt.plot(xs2, kde_miss(xs2), linewidth=1.6)
        plt.xlabel("CI length (97.5% − 2.5%)")
        plt.ylabel("Density")
        plt.title("β_{s,o} CI length — covered vs uncovered")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = fig_dir / "beta_so_ci_length_hist_overlay.png"
        plt.savefig(out_png, bbox_inches="tight"); plt.close()
        print(f"[OK] wrote {out_png}")

    # 2) Boxplot of CI length (covered vs uncovered)
    if (ci_cov.size + ci_miss.size) > 0:
        plt.figure(figsize=(5.4, 4.4), dpi=180)
        data = []; labels = []
        if ci_cov.size > 0:  data.append(ci_cov);  labels.append("Covered")
        if ci_miss.size > 0: data.append(ci_miss); labels.append("Uncovered")
        bp = plt.boxplot(data, labels=labels, vert=True, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_alpha(0.6)
        plt.ylabel("CI length (97.5% − 2.5%)")
        plt.title("Boxplot of β_{s,o} CI length")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_png = fig_dir / "beta_so_ci_length_boxplot_by_coverage.png"
        plt.savefig(out_png, bbox_inches="tight"); plt.close()
        print(f"[OK] wrote {out_png}")

    return combined_csv


def main():
    ap = argparse.ArgumentParser(
        description="Build summary CSVs for Poisson β_{s,o} CI lengths (covered & uncovered) and plot them together."
    )
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Root results folder (parent of 'poisson').")
    ap.add_argument("--combined-csv", default="beta_so_ci_summary.csv",
                    help="Combined CSV filename.")
    ap.add_argument("--fail-csv", default="beta_so_fail_summary.csv",
                    help="Uncovered-only CSV filename.")
    ap.add_argument("--cover-csv", default="beta_so_covered_summary.csv",
                    help="Covered-only CSV filename.")
    args = ap.parse_args()

    build_poisson_beta_so_ci_summary(
        results_root=args.results_root,
        out_combined_csv=args.combined_csv,
        out_fail_csv=args.fail_csv,
        out_cover_csv=args.cover_csv,
    )


if __name__ == "__main__":
    main()
