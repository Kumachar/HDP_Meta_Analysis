#!/usr/bin/env python3
"""
plot_outcome_coverage.py

Create outcome-level coverage barplots from aggregated summaries produced by
`summarize_results.py`.

What it plots:
1) Method-level average 0.95 coverage for β_{s,o} across all sources/outcomes/reps:
      barplot saved as:  beta_so_coverage_method_avg.png

2) Method × Outcome 0.95 coverage (averaged over sources & reps):
      grouped barplot saved as:  beta_so_coverage_by_outcome.png

Also writes the underlying coverage tables to CSV:
  - beta_so_coverage_method_avg.csv
  - beta_so_coverage_by_outcome.csv

Inputs:
  For each family under {results_root}/{family}/beta_summary_stats.csv
  the script expects a boolean column 'beta_so_cover_95' for per-outcome coverage,
  which is written by the main experiment pipeline when per-outcome β_{s,o} is saved.

Usage:
  python plot_outcome_coverage.py --results-root results_YYYY-MM-DD_allmodels \
                                  --families linear poisson logistic

  # custom output folder
  python plot_outcome_coverage.py --results-root ... --out-dir path/to/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def discover_families(results_root: Path) -> list[str]:
    fams: list[str] = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "beta_summary_stats.csv").exists() or (d / "data").exists():
            fams.append(d.name)
    return fams


def load_family_summary(results_root: Path, family: str) -> pd.DataFrame | None:
    csv = Path(results_root) / family / "beta_summary_stats.csv"
    if not csv.exists():
        print(f"[warn] missing {csv}")
        return None
    df = pd.read_csv(csv)
    df["model"] = family
    return df


def make_barplots(stats_all: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Filter to outcome-level coverage rows
    mask_rows = stats_all["outcome"].notna() if "outcome" in stats_all.columns else pd.Series(False, index=stats_all.index)
    if "beta_so_cover_95" not in stats_all.columns:
        raise SystemExit("Column 'beta_so_cover_95' not found in summary CSVs. "
                         "Make sure your pipeline wrote per-outcome coverage rows.")
    df = stats_all.loc[mask_rows, ["model", "source", "outcome", "beta_so_cover_95"]].copy()
    if df.empty:
        raise SystemExit("No outcome-level rows found (column 'outcome' is empty).")

    # Ensure numeric outcome for sorting
    df["outcome"] = df["outcome"].astype(int)

    # 1) Method-level average (over sources, outcomes, reps)
    meth_avg = (
        df.groupby("model", as_index=False)["beta_so_cover_95"]
          .mean()
          .rename(columns={"beta_so_cover_95": "coverage_rate"})
    )
    meth_avg.to_csv(out_dir / "beta_so_coverage_method_avg.csv", index=False)

    g = sns.catplot(
        data=meth_avg,
        x="model", y="coverage_rate",
        kind="bar", height=4.5, aspect=1.4
    )
    g.set_axis_labels("Method (model family)", "0.95 CI coverage for $\\beta_{s,o}$")
    g.set(ylim=(0, 1))
    g.fig.suptitle("Average outcome-level 0.95 CI coverage by method")
    g.tight_layout()
    g.savefig(out_dir / "beta_so_coverage_method_avg.png", dpi=300)
    plt.close(g.fig)

    # 2) Method × Outcome coverage (averaged over sources & reps)
    meth_out = (
        df.groupby(["model", "outcome"], as_index=False)["beta_so_cover_95"]
          .mean()
          .rename(columns={"beta_so_cover_95": "coverage_rate"})
    )
    meth_out["outcome"] = meth_out["outcome"].astype(int)
    meth_out.to_csv(out_dir / "beta_so_coverage_by_outcome.csv", index=False)

    # palette consistent with plot_results.generate_summary_plots (blue, orange, green)
    families_order = list(pd.unique(stats_all["model"]))  # preserves input order
    base_colors = ["#4c72b0", "#dd8452", "#55a868"]  # match plot_results
    if len(families_order) > len(base_colors):
        extra = sns.color_palette("tab10", n_colors=len(families_order) - len(base_colors)).as_hex()
        base_colors += extra
    palette = {fam: col for fam, col in zip(families_order, base_colors[:len(families_order)])}

    # Decide which outcomes to show as tick labels (all if few; thinned if many)
    all_outcomes = sorted(meth_out["outcome"].unique())
    max_ticks = 20  # cap to keep plots readable for very large O
    if len(all_outcomes) <= max_ticks:
        global_tick_outcomes = all_outcomes
    else:
        # pick ~max_ticks evenly spaced outcomes
        idx = np.linspace(0, len(all_outcomes) - 1, num=max_ticks, dtype=int)
        global_tick_outcomes = sorted({all_outcomes[i] for i in idx})

    # Facet one panel per method so bars of the same family are together
    g2 = sns.FacetGrid(
        meth_out,
        col="model",
        col_order=families_order,
        sharey=True,
        height=4.2,
        aspect=1.15,
        despine=False,
        margin_titles=False,
    )

    def _facet_bar(data, **kws):
        m = data["model"].iloc[0]
        ax = plt.gca()

        # Sorted outcomes actually present in this facet
        present_order = sorted(data["outcome"].unique())

        sns.barplot(
            data=data,
            x="outcome",
            y="coverage_rate",
            order=present_order,  # one bar per outcome you actually have
            color=palette.get(m, "#444444"),
            ax=ax,
        )

        ax.set_xlabel("Outcome")
        ax.set_ylabel(r"0.95 CI coverage for $\beta_{s,o}$")
        ax.set_ylim(0, 1)
        ax.axhline(0.95, linestyle="--", color="k", linewidth=1)  # target line
        ax.grid(True, axis="y", alpha=0.3)

        # Tick labels: subset of global_tick_outcomes that exist in this facet.
        tick_vals = [o for o in global_tick_outcomes if o in present_order]
        if not tick_vals:
            # fallback: label all outcomes in this facet
            tick_vals = present_order

        tick_pos = [present_order.index(o) for o in tick_vals]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(o) for o in tick_vals], rotation=45, ha="right")

    g2.map_dataframe(_facet_bar)
    g2.set_titles("{col_name}")  # each facet title is the method/family
    g2.fig.suptitle(
        "Outcome-level 0.95 CI coverage by method (avg over sources & reps)",
        y=1.02
    )
    g2.tight_layout(rect=[0, 0, 1, 0.95])
    g2.savefig(out_dir / "beta_so_coverage_by_outcome.png", dpi=300)
    plt.close(g2.fig)

    print("[OK] Wrote:")
    print("  -", out_dir / "beta_so_coverage_method_avg.png")
    print("  -", out_dir / "beta_so_coverage_by_outcome.png")
    print("  -", out_dir / "beta_so_coverage_method_avg.csv")
    print("  -", out_dir / "beta_so_coverage_by_outcome.csv")



def main():
    ap = argparse.ArgumentParser(description="Outcome-level coverage barplots from aggregated summaries.")
    ap.add_argument("--results-root", type=Path, required=True, help="Root results dir containing family subfolders.")
    ap.add_argument("--families", nargs="+", default=None, help="Subset of families; default: auto-discover.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to save figures (default: {results_root}/summary_figs_outcome_cov)")
    args = ap.parse_args()

    root = args.results_root
    fams = args.families or discover_families(root)
    if not fams:
        raise SystemExit(f"No families found under {root}")

    frames = []
    for fam in fams:
        df = load_family_summary(root, fam)
        if df is not None:
            frames.append(df)
    if not frames:
        raise SystemExit("No family-level CSVs loaded; aborting.")
    stats_all = pd.concat(frames, ignore_index=True, sort=False)

    out_dir = args.out_dir or (root / "summary_figs_outcome_cov")
    make_barplots(stats_all, out_dir)


if __name__ == "__main__":
    main()
