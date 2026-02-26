#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import re


def _iter_beta_npz(rep_dir: Path):
    """
    Yield (source, outcome, npz_path) for files named beta_<s>_<o>.npz
    under rep_dir.
    """
    pattern = re.compile(r"beta_(\d+)_(\d+)\.npz$")
    for f in sorted(rep_dir.glob("beta_*_*.npz")):
        m = pattern.match(f.name)
        if not m:
            continue
        s = int(m.group(1))
        o = int(m.group(2))
        yield s, o, f


def _plot_single_trace(
    samples: np.ndarray,
    out_path: Path,
    title: str | None = None,
    *,
    n_chains: int = 4,
):
    """
    Plot trace(s) for β assuming samples are stored as a flat concatenation
    of multiple chains:

        flat_samples = [chain1_draws, chain2_draws, ..., chainN_draws]

    Parameters
    ----------
    samples : array-like
        1D or higher-dimensional array. Will be flattened and then reshaped
        to (n_chains, n_draws), where n_draws = total_len / n_chains.
    out_path : Path
        Where to save the PNG.
    title : str, optional
        Figure title.
    n_chains : int, default 4
        Number of chains. For your setup: 4 chains × 20000 draws = 80000.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(samples).ravel()
    n_total = samples.size

    if n_total % n_chains != 0:
        raise ValueError(
            f"Total samples {n_total} not divisible by n_chains={n_chains}. "
            "Check n_chains or how samples are stored."
        )

    n_draws = n_total // n_chains
    series = samples.reshape(n_chains, n_draws)  # e.g. (4, 20000)

    x = np.arange(n_draws)

    fig, ax = plt.subplots(figsize=(6, 3))
    for c in range(n_chains):
        ax.plot(
            x,
            series[c],
            linewidth=0.7,
            alpha=0.8,
            label=f"chain {c+1}",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("beta")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="x-small")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def regenerate_trace_plots(
    results_root: Path,
    families: Sequence[str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Regenerate per-outcome beta trace plots from saved beta_{s,o}.npz files.

    For each family f and rep r:
      - looks under results_root/f/data/rep{r}/ for beta_<s>_<o>.npz
      - writes trace plots to results_root/f/figures/trace/rep{r}/
    """
    results_root = Path(results_root)

    if families is None:
        families = sorted(
            d.name for d in results_root.iterdir() if d.is_dir()
        )

    for fam in families:
        model_dir = results_root / fam
        data_dir = model_dir / "data"
        if not data_dir.exists():
            print(f"[skip] {data_dir} not found")
            continue

        rep_dirs = sorted(p for p in data_dir.glob("rep*") if p.is_dir())
        if not rep_dirs:
            print(f"[{fam}] no rep* directories under {data_dir}")
            continue

        print(f"[{fam}] processing reps: {', '.join(d.name for d in rep_dirs)}")

        for rep_dir in rep_dirs:
            rep_name = rep_dir.name  # e.g. 'rep37'
            trace_dir = model_dir / "figures" / "trace" / rep_name

            beta_files = list(_iter_beta_npz(rep_dir))
            if not beta_files:
                print(f"  [warn] {fam}/{rep_name}: no beta_*_*.npz files found; skipping.")
                continue

            # If traces already exist and no overwrite: skip this rep
            if trace_dir.exists() and not overwrite:
                existing = list(trace_dir.glob("beta_*_*_trace.png"))
                if existing:
                    print(f"  [skip] {fam}/{rep_name}: trace plots exist and --overwrite not set.")
                    continue

            n_plots = 0
            for s, o, npz_path in beta_files:
                out_path = trace_dir / f"beta_{s}_{o}_trace.png"
                if out_path.exists() and not overwrite:
                    continue

                data = np.load(npz_path)
                if "samples" not in data:
                    print(f"    [warn] {npz_path.name}: no 'samples' key; skipping.")
                    continue

                samples = np.asarray(data["samples"])
                if samples.size == 0:
                    print(f"    [warn] {npz_path.name}: empty samples; skipping.")
                    continue

                title = f"{fam} {rep_name} β[{s},{o}]"
                # 4 chains × 20000 draws assumed here; adjust n_chains if needed.
                _plot_single_trace(samples, out_path, title=title, n_chains=4)
                n_plots += 1

            print(f"  [done] {fam}/{rep_name}: wrote {n_plots} trace plots -> {trace_dir}")

    print("[DONE] Regenerated beta trace plots from NPZ samples.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate per-outcome beta trace plots from beta_{s,o}.npz files."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing model family subdirs (e.g. results_2025-10-20).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["linear", "poisson", "logistic"],
        help="Model families to process (subdirectories under results-root).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trace PNGs if they already exist.",
    )

    args = parser.parse_args()
    regenerate_trace_plots(
        results_root=args.results_root,
        families=args.families,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
