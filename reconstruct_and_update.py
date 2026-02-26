#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

# Reuse your reconstruction helpers
# (same math used by the CLI that reconstructs a single source)
from reconstruct_source_mixture import (
    reconstruct_source_mixture_from_betas,     # π̂ from {β_s,o}
    _try_load_g0_components,                   # load mu, sigma, beta, alpha0 (strict)
    collect_betas_for_source,                  # scan rep/ for beta_{s}_{o}.npz means
)  # :contentReference[oaicite:1]{index=1}

# --------------------------
# Robust G0 loader (fallbacks)
# --------------------------
def _safe_load_g0(g0_npz: Path, alpha0_override: float | None = None):
    """
    Load (mu, sigma, beta_global, alpha0) for reconstruction.
    Falls back to zeta/pi_norm if 'beta' is not present in the npz produced
    by compute_g0_components().  (compute_g0_components saves mu, sigma,
    alpha0, pi_norm and zeta; not always a 'beta' vector.) :contentReference[oaicite:2]{index=2}
    """
    try:
        mu, sigma, beta, alpha0 = _try_load_g0_components(str(g0_npz), alpha0_override)
        # Returned as 1-D arrays
        return mu, sigma, beta, alpha0
    except Exception:
        # Fallback: open and derive from available arrays
        data = np.load(g0_npz, allow_pickle=True)
        # Average component params over posterior draws -> 1D (k,)
        mu = np.asarray(data["mu"]).mean(axis=0)
        sigma = np.asarray(data["sigma"]).mean(axis=0)

        if "zeta" in data.files:
            # zeta is (n_samples, k), already a per-draw global mixture on G0.
            beta = np.asarray(data["zeta"]).mean(axis=0)
        elif "pi_norm" in data.files:
            # pi_norm: (n_samples, N_sources, k) → pool over sources & draws
            beta = np.asarray(data["pi_norm"]).mean(axis=(0, 1))
        else:
            raise KeyError(
                f"'beta'/'zeta'/'pi_norm' not found in {g0_npz}; keys={data.files}"
            )
        beta = np.clip(beta, 1e-12, None)
        beta = beta / beta.sum()

        if alpha0_override is not None:
            alpha0 = float(alpha0_override)
        elif "alpha0" in data.files:
            alpha0 = float(np.asarray(data["alpha0"]).mean())
        else:
            alpha0 = 1.0
        return mu, sigma, beta, alpha0


def reconstruct_for_rep(rep_dir: Path, *, outcomes_cap: int | None = None,
                        alpha0_override: float | None = None) -> pd.DataFrame:
    """
    For one rep folder (…/family/data/repX):
      • read its existing beta_summary_stats.csv to learn which sources exist,
      • reconstruct π̂_s from {β_s,o} for each source,
      • return a tidy per-source table with recon fields.
    """
    rep_dir = Path(rep_dir)
    stats_csv = rep_dir / "beta_summary_stats.csv"
    if not stats_csv.exists():
        raise FileNotFoundError(f"Missing {stats_csv}")

    df = pd.read_csv(stats_csv)
    sources = sorted(df["source"].unique().tolist())
    g0_npz = rep_dir / "g0_components.npz"
    if not g0_npz.exists():
        raise FileNotFoundError(f"Missing {g0_npz}")

    mu, sigma, beta_global, alpha0 = _safe_load_g0(g0_npz, alpha0_override)

    rows = []
    for s in sources:
        # collect posterior-mean β_s,o across outcomes (uses your robust scanner)
        betas_so = collect_betas_for_source(str(rep_dir), int(s), outcome_cap=outcomes_cap)  # :contentReference[oaicite:3]{index=3}
        grid, density, pi_hat, resp = reconstruct_source_mixture_from_betas(
            betas_so, mu, sigma, beta_global, alpha0
        )  # :contentReference[oaicite:4]{index=4}
        k = len(pi_hat)
        rows.append({
            "source": int(s),
            "recon_k": k,
            "recon_n_outcomes": int(len(betas_so)),
            "recon_alpha0_used": float(alpha0),
            "recon_pi_json": json.dumps([float(x) for x in pi_hat]),
            "recon_max_component": int(np.argmax(pi_hat) + 1),
            "recon_max_weight": float(np.max(pi_hat)),
        })

    return pd.DataFrame(rows)


def update_rep_summary(rep_dir: Path, recon_df: pd.DataFrame) -> Path:
    """
    Merge the reconstruction outputs into the existing rep-level summary and
    write beta_summary_stats_recon.csv next to it.
    """
    rep_dir = Path(rep_dir)
    stats_csv = rep_dir / "beta_summary_stats.csv"
    out_csv   = rep_dir / "beta_summary_stats_recon.csv"

    base = pd.read_csv(stats_csv)
    # one recon row per source → left-merge
    updated = base.merge(recon_df, on="source", how="left")

    # Also expand recon_pi_json into wide columns recon_pi_1..recon_pi_K
    # (use K of the first non-null row)
    first = recon_df["recon_pi_json"].dropna().iloc[0]
    pi_list = json.loads(first)
    for j in range(len(pi_list)):
        updated[f"recon_pi_{j+1}"] = updated["recon_pi_json"].apply(
            lambda s: (json.loads(s)[j] if isinstance(s, str) else np.nan)
        )

    updated.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}  rows={len(updated)}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(
        description="Batch: reconstruct π̂ from per-outcome β samples and update rep summaries."
    )
    ap.add_argument("--results-root", required=True, help="results root (the parent dir of families)")
    ap.add_argument("--families", nargs="+", default=["linear", "poisson", "logistic"])
    ap.add_argument("--reps", nargs="+", type=int, default=None, help="repetition indices (default: all under data/)")
    ap.add_argument("--outcomes-cap", type=int, default=None, help="optionally use only first O outcomes")
    ap.add_argument("--alpha0", type=float, default=None, help="override alpha0 for reconstruction")
    args = ap.parse_args()

    root = Path(args.results_root)

    for fam in args.families:
        data_dir = root / fam / "data"
        if not data_dir.exists():
            print(f"[skip] {data_dir} not found")
            continue

        # choose reps to process
        rep_dirs = sorted([p for p in data_dir.glob("rep*") if p.is_dir()])
        if args.reps:
            rep_dirs = [data_dir / f"rep{r}" for r in args.reps]

        print(f"[{fam}] reps={', '.join(p.name for p in rep_dirs)}")

        for rep_dir in rep_dirs:
            try:
                recon = reconstruct_for_rep(rep_dir,
                                            outcomes_cap=args.outcomes_cap,
                                            alpha0_override=args.alpha0)
                update_rep_summary(rep_dir, recon)
            except Exception as e:
                print(f"[warn] {fam}/{rep_dir.name}: {e}")


if __name__ == "__main__":
    main()
