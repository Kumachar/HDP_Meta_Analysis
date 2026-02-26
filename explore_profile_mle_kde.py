#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Safe non-interactive backend for clusters
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None

# Matches beta_<s>.npz or beta_<s>_<o>.npz
_BETA_FILE_RE = re.compile(r"^(beta_\d+(?:_\d+)?)\.(npz|npy)$")
_BETA_KEY_RE = re.compile(r"^beta_(\d+)(?:_(\d+))?$")


def _parse_beta_key(varname: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse 'beta_<s>' or 'beta_<s>_<o>' and return (s, o).
    If outcome is not present, o is None.
    """
    m = _BETA_KEY_RE.fullmatch(varname)
    if not m:
        return None, None
    s = int(m.group(1))
    o = int(m.group(2)) if m.group(2) is not None else None
    return s, o


def load_beta_archives(
    rep_dir: Path,
    *,
    file_ext: str = "npz",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Lightweight loader for per-(source,outcome) beta archives written by store_beta_posteriors.

    Expected filename: beta_<s>_<o>.npz (or .npy).
    Expected keys in .npz: samples, grid, loglik, prior_samples (optional).

    Returns dict keyed by stem (e.g., 'beta_1_2') -> dict of arrays.
    """
    rep_dir = Path(rep_dir)
    if not rep_dir.is_dir():
        raise FileNotFoundError(f"{rep_dir} is not a directory")

    file_ext = file_ext.lower().lstrip(".")
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for f in sorted(rep_dir.iterdir()):
        if not f.is_file():
            continue
        m = _BETA_FILE_RE.fullmatch(f.name)
        if not m:
            continue
        stem, ext = m.group(1), m.group(2)
        if ext != file_ext:
            continue

        if file_ext == "npz":
            with np.load(f, allow_pickle=True) as data:
                out[stem] = {
                    "samples": data["samples"] if "samples" in data else np.array([]),
                    "grid": data["grid"] if "grid" in data else np.array([]),
                    "loglik": data["loglik"] if "loglik" in data else np.array([]),
                    "prior_samples": data["prior_samples"] if "prior_samples" in data else np.array([]),
                }
        elif file_ext == "npy":
            out[stem] = np.load(f, allow_pickle=True).item()
        else:
            raise ValueError("file_ext must be 'npz' or 'npy'")

    return out


def _safe_kde(samples: np.ndarray):
    """
    Return a gaussian_kde object or None if KDE is not possible.
    """
    if gaussian_kde is None:
        return None
    x = np.asarray(samples, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    # Avoid singular covariance when all values are identical
    if float(np.std(x)) < 1e-12:
        rng = np.random.default_rng(0)
        x = x + rng.normal(0.0, 1e-6, size=x.shape)
    try:
        return gaussian_kde(x)
    except Exception:
        # last resort: add jitter and try again
        rng = np.random.default_rng(0)
        try:
            return gaussian_kde(x + rng.normal(0.0, 1e-6, size=x.shape))
        except Exception:
            return None


def extract_profile_mles_from_rep(
    rep_dir: Path,
    *,
    file_ext: str = "npz",
    outcome_cap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read beta_{s}_{o}.npz files under one rep directory and compute:
        beta_mle_{s,o} = argmax_b loglik_{s,o}(b)

    Returns a tidy DataFrame with columns:
      rep, source, outcome, beta_mle, ll_max, at_boundary, beta_true, mle_minus_true
    """
    rep_dir = Path(rep_dir)
    rep_name = rep_dir.name  # e.g., "rep12"
    try:
        rep_idx = int(rep_name.replace("rep", ""))
    except Exception:
        rep_idx = None

    beta_arch = load_beta_archives(rep_dir, file_ext=file_ext)

    rows = []
    for varname, info in beta_arch.items():
        s, o = _parse_beta_key(varname)
        if s is None:
            continue
        # We want per-outcome distributions. Skip aggregated beta_<s> files.
        if o is None:
            continue
        if outcome_cap is not None and o > int(outcome_cap):
            continue

        grid = np.asarray(info.get("grid", np.array([])), dtype=float).ravel()
        ll = np.asarray(info.get("loglik", np.array([])), dtype=float).ravel()
        if grid.size == 0 or ll.size == 0:
            continue

        n = min(grid.size, ll.size)
        grid = grid[:n]
        ll = ll[:n]

        idx = int(np.argmax(ll))
        beta_mle = float(grid[idx])
        ll_max = float(ll[idx])
        at_boundary = bool(idx == 0 or idx == (n - 1))

        beta_true = np.nan
        prior = np.asarray(info.get("prior_samples", np.array([])), dtype=float).ravel()
        prior = prior[np.isfinite(prior)]
        if prior.size:
            # In your simulations, this is typically a single value per (s,o).
            beta_true = float(prior.mean())

        rows.append(
            {
                "rep": rep_idx,
                "source": int(s),
                "outcome": int(o),
                "beta_mle": beta_mle,
                "ll_max": ll_max,
                "at_boundary": at_boundary,
                "beta_true": beta_true,
                "mle_minus_true": (beta_mle - beta_true) if np.isfinite(beta_true) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["rep", "source", "outcome"]).reset_index(drop=True)
    return df


def summarise_mles_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise beta_mle by (rep, source).
    """
    if df.empty:
        return df.copy()

    grp = df.groupby(["rep", "source"], as_index=False)
    out = grp["beta_mle"].agg(
        n_outcomes="count",
        mle_mean="mean",
        mle_median="median",
        mle_sd="std",
        mle_min="min",
        mle_max="max",
    )

    # boundary rate
    bnd = grp["at_boundary"].mean().rename(columns={"at_boundary": "boundary_rate"})
    out = out.merge(bnd, on=["rep", "source"], how="left")

    # truth error summary (if present)
    if "mle_minus_true" in df.columns:
        err = grp["mle_minus_true"].agg(
            err_mean="mean",
            err_median="median",
            err_sd="std",
        )
        out = out.merge(err, on=["rep", "source"], how="left")

    return out


def _auto_xlim(values: np.ndarray, pad_sd: float = 2.5) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd < 1e-12:
        pad = 1.0
    else:
        pad = pad_sd * sd
    return (lo - pad, hi + pad)


def plot_source_mle_kde_grid(
    df: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
    bins: int = 20,
    dpi: int = 220,
    xlim: Optional[Tuple[float, float]] = None,
    overlay_truth: bool = False,
) -> None:
    """
    Create a grid of subplots, one per source, showing:
      - histogram of per-outcome beta_mle
      - KDE overlay (when possible)
      - vertical lines for mean/median
      - optional truth markers (beta_true per outcome)
    """
    if df.empty:
        return

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    sources = sorted(df["source"].unique().tolist())
    n = len(sources)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()

    if xlim is None:
        xlim = _auto_xlim(df["beta_mle"].to_numpy())

    for i, s in enumerate(sources):
        ax = axes[i]
        vals = df.loc[df["source"] == s, "beta_mle"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        ax.hist(vals, bins=bins, density=True, alpha=0.55, label="Outcome-wise MLE")

        kde = _safe_kde(vals)
        if kde is not None:
            xs = np.linspace(xlim[0], xlim[1], 400)
            ax.plot(xs, kde(xs), linewidth=1.5, label="KDE")

        if vals.size:
            ax.axvline(float(vals.mean()), linestyle="--", linewidth=1.0, label="Mean")
            ax.axvline(float(np.median(vals)), linestyle=":", linewidth=1.2, label="Median")

        if overlay_truth and ("beta_true" in df.columns):
            truths = df.loc[df["source"] == s, "beta_true"].to_numpy(dtype=float)
            truths = truths[np.isfinite(truths)]
            if truths.size:
                ax.plot(truths, np.zeros_like(truths), marker="x", linestyle="None",
                        markersize=4, label="True β (per outcome)")

        ax.set_title(f"Source {s}  (n={vals.size})")
        ax.set_xlabel(r"$\hat\beta$ (profile MLE)")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.35)

        # Keep legends compact
        if i == 0:
            ax.legend(fontsize="small")
        else:
            if overlay_truth and ("beta_true" in df.columns):
                ax.legend(fontsize="x-small")

    # Turn off unused panels
    for j in range(n, axes.size):
        axes[j].axis("off")

    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def discover_families(results_root: Path) -> list[str]:
    """
    Discover model family subfolders under results_root that contain:
        {family}/data/rep*/beta_*.npz
    """
    results_root = Path(results_root)
    fams = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        data_dir = d / "data"
        if not data_dir.exists():
            continue
        has_beta = any(p.is_file() and p.name.startswith("beta_") for p in data_dir.glob("rep*/beta_*.npz"))
        if has_beta:
            fams.append(d.name)
    return sorted(fams)


def _select_rep_dirs(data_dir: Path, reps: Optional[Iterable[int]] = None) -> list[Path]:
    rep_dirs = sorted([p for p in data_dir.glob("rep*") if p.is_dir()])
    if reps is None:
        return rep_dirs
    picked = []
    for r in reps:
        p = data_dir / f"rep{int(r)}"
        if p.is_dir():
            picked.append(p)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Exploratory plots based on profile likelihood maxima: "
            "for each (source, outcome), compute beta_hat = argmax profile log-likelihood, "
            "then plot per-source KDE/hist across outcomes."
        )
    )
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Root results folder that contains {family}/data/rep*/beta_{s}_{o}.npz")
    ap.add_argument("--families", nargs="+", default=None,
                    help="Families to process (e.g., linear poisson logistic). Default: auto-discover.")
    ap.add_argument("--reps", nargs="+", type=int, default=None,
                    help="Repetition indices (e.g., 1 2 3). Default: all discovered.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for exploratory figures/CSVs. "
                         "Default: {results_root}/{family}/figures/profile_explore")
    ap.add_argument("--file-ext", type=str, default="npz",
                    help="File extension for stored beta archives (default: npz).")
    ap.add_argument("--outcome-cap", type=int, default=None,
                    help="Optionally use only outcomes <= this cap per source.")
    ap.add_argument("--bins", type=int, default=20,
                    help="Histogram bins per source (default: 20).")
    ap.add_argument("--dpi", type=int, default=220,
                    help="DPI for output figures.")
    ap.add_argument("--xlim", nargs=2, type=float, default=None,
                    help="Fixed x-axis limits for all panels: --xlim -3 3")
    ap.add_argument("--overlay-truth", action="store_true",
                    help="If prior_samples exist in beta_{s}_{o}.npz, overlay true β markers.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"[ERR] results_root not found: {results_root}")

    families = args.families or discover_families(results_root)
    if not families:
        raise SystemExit(f"[ERR] No families discovered under {results_root}")

    xlim = tuple(args.xlim) if args.xlim is not None else None

    for fam in families:
        model_dir = results_root / fam
        data_dir = model_dir / "data"
        rep_dirs = _select_rep_dirs(data_dir, args.reps)

        if not rep_dirs:
            print(f"[WARN] {fam}: no rep* dirs found under {data_dir}; skipping.")
            continue

        out_dir = (args.out_dir / fam) if args.out_dir is not None else (model_dir / "figures" / "profile_explore")
        out_dir.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for rep_dir in rep_dirs:
            df_rep = extract_profile_mles_from_rep(
                rep_dir,
                file_ext=args.file_ext,
                outcome_cap=args.outcome_cap,
            )
            if df_rep.empty:
                print(f"[WARN] {fam}/{rep_dir.name}: no beta_{'{s}_{o}'} archives found; skipping rep.")
                continue

            all_rows.append(df_rep)

            # Per-rep CSVs
            df_rep.to_csv(out_dir / f"{rep_dir.name}_profile_mle_by_outcome.csv", index=False)
            sum_rep = summarise_mles_by_source(df_rep)
            sum_rep.to_csv(out_dir / f"{rep_dir.name}_profile_mle_by_source.csv", index=False)

            # Per-rep figure
            plot_source_mle_kde_grid(
                df_rep,
                title=f"{fam}  {rep_dir.name}: profile-likelihood MLEs across outcomes",
                out_png=out_dir / f"{rep_dir.name}_profile_mle_kde.png",
                bins=args.bins,
                dpi=args.dpi,
                xlim=xlim,
                overlay_truth=args.overlay_truth,
            )

        if not all_rows:
            print(f"[WARN] {fam}: no reps produced any MLEs; skipping family-level outputs.")
            continue

        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(out_dir / "allreps_profile_mle_by_outcome.csv", index=False)
        sum_all = summarise_mles_by_source(df_all)
        sum_all.to_csv(out_dir / "allreps_profile_mle_by_source.csv", index=False)

        # Aggregated figure (pooled over reps)
        plot_source_mle_kde_grid(
            df_all,
            title=f"{fam}: profile-likelihood MLEs across outcomes (pooled over reps)",
            out_png=out_dir / "allreps_profile_mle_kde.png",
            bins=args.bins,
            dpi=args.dpi,
            xlim=xlim,
            overlay_truth=args.overlay_truth,
        )

        print(f"[OK] {fam}: wrote outputs to {out_dir.resolve()}")

    print("[DONE] Exploratory profile-likelihood plots completed.")


if __name__ == "__main__":
    main()
