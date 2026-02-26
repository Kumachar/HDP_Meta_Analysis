#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from utils import true_source_moments  # uses your existing implementation


def _compute_mu_stats_from_g0(
    rep_dir: Path,
    true_means: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Reconstruct μ_s_hat and its 95% CI for a single repetition, using
    g0_components.npz produced by compute_g0_components().
    """
    g0_path = rep_dir / "g0_components.npz"
    if not g0_path.exists():
        # nothing to add
        return pd.DataFrame(
            {
                "source": [],
                "mu_s_hat": [],
                "mu_ci_lower": [],
                "mu_ci_upper": [],
                "mu_cover_95": [],
            }
        )

    data = np.load(g0_path)
    # Saved by compute_g0_components: pi_norm (n_samples, S, K), mu (n_samples, K)
    pi = np.asarray(data["pi_norm"])  # (n_samples, S, K)
    mu = np.asarray(data["mu"])       # (n_samples, K)

    # Posterior draws of mixture mean μ_s for each source s:
    # μ_s^(t) = sum_j pi_norm[t,s,j] * mu[t,j]
    mu_expanded = mu[:, None, :]                # (n_samples, 1, K)
    mu_s_draws = (pi * mu_expanded).sum(axis=-1)  # (n_samples, S)

    mu_s_hat = mu_s_draws.mean(axis=0)
    mu_lo = np.percentile(mu_s_draws, 2.5, axis=0)
    mu_hi = np.percentile(mu_s_draws, 97.5, axis=0)

    S = mu_s_hat.shape[0]
    cover: list[bool | float] = []
    if true_means is not None:
        true_means = np.asarray(true_means, float)
        for s, (lo, hi) in enumerate(zip(mu_lo, mu_hi), start=1):
            if s - 1 < len(true_means):
                cover.append(bool(lo <= float(true_means[s - 1]) <= hi))
            else:
                cover.append(np.nan)
    else:
        cover = [np.nan] * S

    return pd.DataFrame(
        {
            "source": np.arange(1, S + 1, dtype=int),
            "mu_s_hat": mu_s_hat.astype(float),
            "mu_ci_lower": mu_lo.astype(float),
            "mu_ci_upper": mu_hi.astype(float),
            "mu_cover_95": cover,
        }
    )


def _compute_outcome_error_and_coverage(
    rep_dir: Path,
    rep_label: str,
    rep_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Approximate outcome-level errors and coverage from saved beta_{s,o}.npz files.

    We reconstruct an approximate “true” β_{s,o} as the grid value at which the
    profile log-likelihood is maximised. This is not the original simulated
    beta_true but is typically close; it lets us recover a coverage-style
    diagnostic without re-running the simulation.
    """
    err_rows: list[dict] = []
    cov_rows: list[dict] = []

    # s -> list of (o, b_true_approx, post_mean, lo, hi, cover)
    betas_by_source: dict[int, list[tuple[int, float, float, float, float, bool | float]]] = {}

    for npz_path in rep_dir.glob("beta_*_*.npz"):
        stem = npz_path.stem  # e.g. 'beta_3_17'
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        _, s_str, o_str = parts
        try:
            s = int(s_str)
            o = int(o_str)
        except ValueError:
            continue

        data = np.load(npz_path)
        samples = np.asarray(data["samples"]).ravel()
        grid = np.asarray(data.get("grid", []))
        loglik = np.asarray(data.get("loglik", []))

        # Approximate "true" β_{s,o} as argmax of profile log-likelihood
        if grid.size and loglik.size and grid.shape == loglik.shape:
            idx_max = int(np.argmax(loglik))
            b_true = float(grid[idx_max])
        else:
            b_true = np.nan

        if samples.size:
            lo, hi = np.percentile(samples, [2.5, 97.5])
            post_mean = float(samples.mean())
        else:
            lo = hi = post_mean = np.nan

        if np.isfinite(b_true) and np.isfinite(lo) and np.isfinite(hi):
            cover = bool(lo <= b_true <= hi)
        else:
            cover = np.nan

        betas_by_source.setdefault(s, []).append(
            (o, b_true, post_mean, float(lo), float(hi), cover)
        )

    # Per-source outcome_error_mean and per-outcome coverage rows
    for s, rows in betas_by_source.items():
        errs: list[float] = []
        for (o, b_true, post_mean, lo, hi, cover) in rows:
            if np.isfinite(b_true) and np.isfinite(post_mean):
                errs.append(post_mean - b_true)

            cov_rows.append(
                {
                    "experiment": rep_label,
                    "rep": rep_idx,
                    "source": s,
                    "outcome": o,
                    "beta_so_ci_lower": lo,
                    "beta_so_ci_upper": hi,
                    "beta_so_cover_95": cover,
                }
            )

        O = len(rows)
        outcome_err_mean = float(np.mean(errs)) if errs else np.nan
        err_rows.append(
            {
                "source": s,
                "O": O,
                "outcome_error_mean": outcome_err_mean,
            }
        )

    err_df = pd.DataFrame(err_rows)
    cov_df = pd.DataFrame(cov_rows)
    return err_df, cov_df


def rebuild_rep_summary(rep_dir: Path, rep_idx: int | None = None) -> pd.DataFrame:
    """
    Rebuild the richer, 'previous' per-rep summary for a single repetition.

    Requires (under rep_dir):
      - beta_summary_stats.csv      (current per-rep summary; used as base)
      - truth_source_mixtures.npz   (beta_mean, beta_sds, pis)
      - g0_components.npz           (mu, sigma, pi_norm, alpha0, ...)
      - beta_{s,o}.npz              (posterior samples, grid, loglik, prior_samples)
    """
    rep_dir = Path(rep_dir)

    base_path = rep_dir / "beta_summary_stats.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base summary not found: {base_path}")
    base = pd.read_csv(base_path)

    rep_label = rep_dir.name  # e.g. 'rep37'
    if rep_idx is None:
        try:
            rep_idx = int(rep_label.replace("rep", ""))
        except Exception:
            rep_idx = -1

    if "experiment" not in base.columns:
        base["experiment"] = rep_label
    if "rep" not in base.columns:
        base["rep"] = rep_idx

    # Load true mixture ingredients to get true source means (for μ_s coverage)
    truth_path = rep_dir / "truth_source_mixtures.npz"
    true_means = None
    if truth_path.exists():
        truth = np.load(truth_path)
        beta_mean = truth["beta_mean"]
        beta_sds = truth["beta_sds"]
        pis_true = truth["pis"]
        true_means, _true_vars = true_source_moments(pis_true, beta_mean, beta_sds)
    else:
        true_means = None

    # μ_s_hat and μ_s CI + coverage
    mu_df = _compute_mu_stats_from_g0(rep_dir, true_means=true_means)

    # Outcome-level error summary and β_{s,o} coverage (approximate)
    err_df, cov_df = _compute_outcome_error_and_coverage(
        rep_dir, rep_label=rep_label, rep_idx=rep_idx
    )

    # Merge μ_s_hat and outcome_error_mean into per-source stats
    stats = base.merge(mu_df, on="source", how="left")
    if not err_df.empty:
        stats = stats.merge(err_df, on="source", how="left")

    # μ_s_hat_error = μ_s_hat − prior_mean (prior_mean here is the true source mean)
    if "mu_s_hat" in stats.columns and "prior_mean" in stats.columns:
        stats["mu_s_hat_error"] = stats["mu_s_hat"] - stats["prior_mean"]

    # Append per-outcome coverage rows as in the previous pipeline
    if not cov_df.empty:
        stats_prev = pd.concat([stats, cov_df], ignore_index=True, sort=False)
    else:
        stats_prev = stats

    return stats_prev


def regenerate_previous_summary(
    results_root: Path,
    families: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Regenerate 'previous-style' beta summary tables for all families and reps
    under a results root, without re-running the HDP experiments.

    For each family f:
      - builds per-rep beta_summary_stats_prev.csv under f/data/rep{r}/
      - builds family-level beta_summary_stats_prev.csv under f/
    At the root:
      - builds beta_summary_stats_prev_all.csv aggregating all families.
    """
    results_root = Path(results_root)
    if families is None:
        families = sorted(d.name for d in results_root.iterdir() if d.is_dir())

    all_frames: list[pd.DataFrame] = []

    for fam in families:
        data_dir = results_root / fam / "data"
        if not data_dir.exists():
            print(f"[skip] {data_dir} not found")
            continue

        rep_dirs = sorted(p for p in data_dir.glob("rep*") if p.is_dir())
        fam_frames: list[pd.DataFrame] = []

        print(f"[{fam}] processing reps: {', '.join(d.name for d in rep_dirs)}")

        for rep_dir in rep_dirs:
            try:
                rep_idx = int(rep_dir.name.replace("rep", ""))
            except Exception:
                rep_idx = -1

            try:
                df_prev = rebuild_rep_summary(rep_dir, rep_idx=rep_idx)
            except Exception as e:
                print(f"[warn] {fam}/{rep_dir.name}: {e}")
                continue

            # Per-rep "previous-style" summary next to the base CSV
            out_rep = rep_dir / "beta_summary_stats.csv"
            df_prev.to_csv(out_rep, index=False)
            print(f"  [write] {out_rep}  rows={len(df_prev)}")

            df_prev = df_prev.copy()
            df_prev.insert(0, "model", fam)
            fam_frames.append(df_prev)

        if not fam_frames:
            print(f"[{fam}] nothing to aggregate")
            continue

        fam_summary = pd.concat(fam_frames, ignore_index=True, sort=False)
        out_fam = results_root / fam / "beta_summary_stats_prev.csv"
        fam_summary.to_csv(out_fam, index=False)
        print(f"[{fam}] family-level summary -> {out_fam}  shape={fam_summary.shape}")

        all_frames.append(fam_summary)

    if not all_frames:
        raise RuntimeError("No summaries regenerated; check results_root and families.")

    all_summary = pd.concat(all_frames, ignore_index=True, sort=False)
    out_all = results_root / "beta_summary_stats_prev_all.csv"
    all_summary.to_csv(out_all, index=False)
    print(f"[ALL] aggregated previous-style summary -> {out_all}  shape={all_summary.shape}")
    return all_summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate 'previous-style' beta summary tables from saved outputs."
    )
    ap.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Root results folder (e.g., results_sample_2025-09-25)",
    )
    ap.add_argument(
        "--families",
        nargs="+",
        default=None,
        help=(
            "Families (subdirectories under results_root) to process; "
            "default: auto-discover."
        ),
    )
    args = ap.parse_args()

    regenerate_previous_summary(results_root=args.results_root, families=args.families)


if __name__ == "__main__":
    main()
