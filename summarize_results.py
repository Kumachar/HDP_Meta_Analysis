#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from plot_results import generate_summary_plots  # builds figures into out_fig_dir


def _pick_rep_csv(rep_dir: Path, prefer_recon: bool, strict_recon: bool) -> Optional[Path]:
    """
    Choose which per-rep CSV to use:
      - if prefer_recon: use beta_summary_stats_recon.csv when present, else fallback
      - if strict_recon: require recon file; return None if missing
    """
    recon = rep_dir / "beta_summary_stats_recon.csv"
    base  = rep_dir / "beta_summary_stats.csv"
    if strict_recon:
        return recon if recon.exists() else None
    if prefer_recon and recon.exists():
        return recon
    return base if base.exists() else None


def aggregate_family_summary(
    results_root: Path,
    family: str,
    *,
    prefer_recon: bool = True,
    strict_recon: bool = False,
) -> Path:
    """
    Concatenate all per-rep summary CSVs under:
        {results_root}/{family}/data/rep*/beta_summary_stats[_recon].csv
    into one file at:
        {results_root}/{family}/beta_summary_stats.csv

    Notes:
      • If 'beta_summary_stats_recon.csv' exists, it is preferred (or required
        if strict_recon=True). Otherwise we fallback to 'beta_summary_stats.csv'.
      • Any extra columns (e.g., recon_pi_json, recon_pi_*) are preserved.
    """
    model_dir = Path(results_root) / family
    data_dir = model_dir / "data"
    rep_dirs = sorted([p for p in data_dir.glob("rep*") if p.is_dir()])
    used_paths: List[Path] = []

    for rep_dir in rep_dirs:
        p = _pick_rep_csv(rep_dir, prefer_recon=prefer_recon, strict_recon=strict_recon)
        if p is not None:
            used_paths.append(p)

    if not used_paths:
        raise FileNotFoundError(
            f"[{family}] No per-rep summary CSVs found under {data_dir}/rep*/ "
            f"(prefer_recon={prefer_recon}, strict_recon={strict_recon})"
        )

    frames: List[pd.DataFrame] = []
    n_recon = 0
    for csv in used_paths:
        df = pd.read_csv(csv)
        if "rep" not in df.columns:
            # infer rep index from folder name "rep{n}"
            try:
                rep_idx = int(csv.parent.name.replace("rep", ""))
            except Exception:
                rep_idx = None
            df["rep"] = rep_idx
        df["model"] = family
        df["used_recon"] = csv.name.endswith("_recon.csv")
        n_recon += int(df["used_recon"].iloc[0])
        frames.append(df)

    summary = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["rep", "source", "outcome"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    out_csv = model_dir / "beta_summary_stats.csv"  # family-level file for plots
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(
        f"[OK] {family}: wrote {out_csv}  rows={len(summary)}  reps={len(used_paths)} "
        f"(recon_used_in={n_recon} reps)"
    )

    # Also keep a copy with explicit suffix, useful for debugging
    out_csv_recon = model_dir / "beta_summary_stats_aggregated_source.csv"
    summary.to_csv(out_csv_recon, index=False)

    return out_csv


def discover_families(results_root: Path) -> list[str]:
    """
    Find subfolders under results_root that contain either:
      data/rep*/beta_summary_stats_recon.csv  OR  data/rep*/beta_summary_stats.csv.
    """
    fams: list[str] = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        data_dir = d / "data"
        if not data_dir.exists():
            continue
        has_any = list(data_dir.glob("rep*/beta_summary_stats_recon.csv")) or \
                  list(data_dir.glob("rep*/beta_summary_stats.csv"))
        if has_any:
            fams.append(d.name)
    return sorted(fams)


def main():
    p = argparse.ArgumentParser(
        description="Aggregate per-rep summaries (preferring reconstruction) and generate family-level plots."
    )
    p.add_argument("--results-root", type=Path, required=True,
                   help="Root results folder (e.g., results_YYYY-MM-DD_allmodels)")
    p.add_argument("--families", nargs="+", default=None,
                   help="Families to include (e.g., linear poisson logistic). "
                        "If omitted, auto-discover under results_root.")
    p.add_argument("--out-fig-dir", type=Path, default=None,
                   help="Where to write figures (default: {results_root}/summary_figs)")
    p.add_argument("--prefer-recon", action="store_true", default=True,
                   help="Prefer beta_summary_stats_recon.csv when present (default).")
    p.add_argument("--no-prefer-recon", dest="prefer_recon", action="store_false",
                   help="Disable preference for reconstruction.")
    p.add_argument("--strict-recon", action="store_true", default=False,
                   help="Require reconstructed per-rep CSVs; error if any rep lacks them.")
    args = p.parse_args()

    results_root: Path = args.results_root
    if not results_root.exists():
        raise SystemExit(f"[ERR] results_root not found: {results_root}")

    families = args.families or discover_families(results_root)
    if not families:
        raise SystemExit(f"[ERR] No families discovered under: {results_root}")

    # 1) Aggregate per-rep → family-level CSVs (prefer recon when possible)
    for fam in families:
        aggregate_family_summary(
            results_root,
            fam,
            prefer_recon=args.prefer_recon,
            strict_recon=args.strict_recon,
        )

    # 2) Build summary figures (unchanged API)
    out_fig_dir = args.out_fig_dir or (results_root / "summary_figs")
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    generate_summary_plots(
        results_root=results_root,
        families=families,
        out_fig_dir=out_fig_dir,
    )

    # 3) Show what was produced
    produced = sorted(p.name for p in out_fig_dir.glob("*.png"))
    if produced:
        print("[OK] Figures written:")
        for name in produced:
            print("  -", name)
    else:
        print("[WARN] No PNGs found in", out_fig_dir)


if __name__ == "__main__":
    main()
