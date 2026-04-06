#!/usr/bin/env python3
"""
main_realdata_optimized.py

Purpose
-------
Run the HDP profile-likelihood model on REAL (non-simulated) profile-likelihood curves,
then generate a set of exploratory + diagnostic figures (trace plots, G0 plots,
source-level densities, profile-MLE KDEs) WITHOUT requiring any "truth".

This version is optimized to avoid out-of-memory (OOM) failure on small-memory
SLURM jobs by:
  1) Avoiding expansion of `beta_s_o` into hundreds of xarray variables.
  2) Avoiding ArviZ InferenceData construction for the full posterior.
  3) Subsampling posterior draws for expensive G0 computations / trajectory plots.
  4) Writing figures incrementally (one-by-one) and aggressively closing matplotlib
     figures to keep peak memory low.

Input format
------------
Long format CSV is recommended:
  columns: source, outcome, point, value
where:
  - point is the beta grid value
  - value is the profile log-likelihood at that point.

You may pass a single CSV via --data-file, or a folder via --data-root. If data-root
contains rep1/, rep2/, ... they are treated as separate repetitions.

Outputs
-------
{output_base}/{family}/
  data/rep{rep}/
    profile_likelihood_used.csv
    mapping_sources.csv
    mapping_outcomes.csv
    profile_mle_by_outcome.csv
    profile_mle_by_source.csv
    beta_{s}_{o}.npz           (posterior samples + the curve used)
    beta_{s}.npz               (pooled posterior samples per source)
    g0_components_subsampled.npz
  figures/
    trace_plots/
      rep{rep}_trace_hypers.png
      rep{rep}_trace_betas.png
    beta_plots/
      rep{rep}_g0.png
      rep{rep}_g0_trajs.png
      rep{rep}_g0_beta.png
    profile_mle/
      rep{rep}_profile_mle_kde.png
    source_density/
      rep{rep}_source_density_grid.png
      rep{rep}_source{s}_density.png

Notes
-----
- This script expects the project modules `models_vectorized.py` and/or `models.py`
  to be importable from the current working directory. Run it from your project root
  or `cd` into the project root in your SLURM script.
- For performance, it is strongly recommended to use the vectorized model.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────
def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (long format)
# ──────────────────────────────────────────────────────────────────────────────
_LONG_COL_ALIASES = {
    "source": {"source", "src", "s"},
    "outcome": {"outcome", "out", "o"},
    "point": {"point", "grid", "beta", "x", "b"},
    "value": {"value", "loglik", "loglike", "ll", "y"},
}

def _normalize_long_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for std, aliases in _LONG_COL_ALIASES.items():
        for a in aliases:
            if a in cols_lower:
                mapping[cols_lower[a]] = std
                break
    return df.rename(columns=mapping).copy()

def discover_rep_dirs(data_root: Path) -> list[Path]:
    reps = sorted([p for p in data_root.iterdir() if p.is_dir() and re.fullmatch(r"rep\d+", p.name)])
    return reps if reps else [data_root]

def load_profile_likelihoods(
    rep_path: Path,
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Load profile likelihood curves into a single long DataFrame.

    Returns columns:
      source_raw, outcome_raw, point, value, file
    """
    rep_path = Path(rep_path)
    if not rep_path.exists():
        raise FileNotFoundError(f"Path not found: {rep_path}")

    files: list[Path]
    if rep_path.is_file():
        if rep_path.suffix.lower() != ".csv":
            raise ValueError(f"--data-file must be a .csv, got: {rep_path}")
        files = [rep_path]
    else:
        files = sorted(rep_path.rglob(pattern) if recursive else rep_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files found under {rep_path} (pattern={pattern}, recursive={recursive})")

    parts = []
    for f in files:
        df = pd.read_csv(f)
        dfn = _normalize_long_columns(df)
        cols = set(map(str.lower, dfn.columns))

        if not {"source", "outcome", "point", "value"}.issubset(cols):
            raise ValueError(
                f"{f} does not look like long-format input. "
                f"Expected columns include source/outcome/point/value. Got: {list(df.columns)}"
            )

        out = dfn[["source", "outcome", "point", "value"]].copy()
        out["source_raw"] = out["source"].astype(str)
        out["outcome_raw"] = out["outcome"].astype(str)
        out["file"] = str(f.name if rep_path.is_file() else f.relative_to(rep_path))
        out = out[["source_raw", "outcome_raw", "point", "value", "file"]]
        parts.append(out)

    df_all = pd.concat(parts, ignore_index=True)
    df_all["point"] = pd.to_numeric(df_all["point"], errors="coerce")
    df_all["value"] = pd.to_numeric(df_all["value"], errors="coerce")
    df_all = df_all.dropna(subset=["source_raw", "outcome_raw", "point", "value"]).copy()

    df_all = df_all.sort_values(["source_raw", "outcome_raw", "point"]).reset_index(drop=True)
    return df_all


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Optional source filtering (real data)
# ──────────────────────────────────────────────────────────────────────────────
def filter_sources_df(
    df_raw: pd.DataFrame,
    *,
    sources_range: tuple[float, float] | None = None,
    sources_include: list[str] | None = None,
    sources_exclude: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter the long profile-likelihood table by *raw* source labels (before remapping).

    Parameters
    ----------
    sources_range:
        Keep only rows whose `source_raw` is numeric and lies in [lo, hi] (inclusive).
        Example: (4, 12).
    sources_include:
        Keep only sources whose raw label (string) is in this list.
    sources_exclude:
        Drop sources whose raw label (string) is in this list.

    Notes
    -----
    - Filtering is applied in the order: include -> exclude -> range.
    - Comparisons for include/exclude are string-based on `source_raw`.
    """
    df = df_raw.copy()

    if sources_include:
        keep = set(map(str, sources_include))
        df = df[df["source_raw"].astype(str).isin(keep)].copy()

    if sources_exclude:
        drop = set(map(str, sources_exclude))
        df = df[~df["source_raw"].astype(str).isin(drop)].copy()

    if sources_range is not None:
        lo, hi = float(sources_range[0]), float(sources_range[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"sources_range must be finite, got {sources_range}")
        if lo > hi:
            lo, hi = hi, lo
        src_num = pd.to_numeric(df["source_raw"], errors="coerce")
        df = df[src_num.between(lo, hi, inclusive="both")].copy()

    return df

# Mapping to consecutive 1-based integer ids (required by file naming)
# ──────────────────────────────────────────────────────────────────────────────
def _sorted_mixed(values: Iterable) -> list:
    vals = list(values)
    def key(v):
        try:
            return (0, float(v))
        except Exception:
            return (1, str(v))
    return sorted(vals, key=key)

@dataclass
class IdMaps:
    source_map: Dict[str, int]
    outcome_map_by_source: Dict[int, Dict[str, int]]

def remap_ids(
    df_raw: pd.DataFrame,
    *,
    outcome_cap: int | None = None,
) -> tuple[pd.DataFrame, IdMaps, pd.DataFrame]:
    """
    Convert arbitrary source/outcome labels to consecutive 1-based integers:
      source:  1..N_sources
      outcome: 1..M_s (per source)

    Returns:
      df_sim: with columns source,outcome,point,value plus source_orig/outcome_orig/file
      maps: mapping objects
      mapping_df: tidy mapping table
    """
    df = df_raw.copy()
    df["source_orig"] = df["source_raw"].astype(str)
    df["outcome_orig"] = df["outcome_raw"].astype(str)

    sources = _sorted_mixed(df["source_orig"].unique())
    source_map = {s: i + 1 for i, s in enumerate(sources)}
    df["source"] = df["source_orig"].map(source_map).astype(int)

    outcome_map_by_source: Dict[int, Dict[str, int]] = {}
    mapping_rows = []
    for s_orig in sources:
        s_new = source_map[s_orig]
        outs = _sorted_mixed(df.loc[df["source"] == s_new, "outcome_orig"].unique())
        out_map = {o: j + 1 for j, o in enumerate(outs)}
        outcome_map_by_source[s_new] = out_map
        for o_orig, o_new in out_map.items():
            mapping_rows.append({
                "source": s_new, "source_orig": s_orig,
                "outcome": o_new, "outcome_orig": o_orig,
            })

    df["outcome"] = df.apply(
        lambda r: outcome_map_by_source[int(r["source"])][str(r["outcome_orig"])], axis=1
    ).astype(int)

    if outcome_cap is not None:
        df = df[df["outcome"] <= int(outcome_cap)].copy()

    df_sim = df[["source", "outcome", "point", "value", "source_orig", "outcome_orig", "file"]].copy()
    df_sim = df_sim.sort_values(["source", "outcome", "point"]).reset_index(drop=True)

    maps = IdMaps(source_map=source_map, outcome_map_by_source=outcome_map_by_source)
    mapping_df = pd.DataFrame(mapping_rows).sort_values(["source", "outcome"]).reset_index(drop=True)
    return df_sim, maps, mapping_df


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized packing
# ──────────────────────────────────────────────────────────────────────────────
def pack_sources_vectorized(
    source_outcome_data_dict: Dict[int, List[np.ndarray]],
    *,
    M_outcomes: int | None = None,
):
    """
    Pack curves into padded tensors for models_vectorized.HDP_model_vectorized.

    Returns:
      x_padded:  (N_sources, M_max, Lmax) float32
      ll_padded: (N_sources, M_max, Lmax) float32
      m_valid:   (N_sources,) int32   number of real outcomes per source
    """
    import jax.numpy as jnp

    src_keys = sorted(source_outcome_data_dict.keys())
    lists = [source_outcome_data_dict[k] for k in src_keys]
    m_valid = np.array([len(lst) for lst in lists], dtype=int)
    M = int(M_outcomes) if M_outcomes is not None else int(m_valid.max(initial=0))
    if M <= 0:
        raise ValueError("No outcomes to pack.")

    Lmax = 0
    for lst in lists:
        for a in lst:
            Lmax = max(Lmax, int(a.shape[0]))
    if Lmax < 2:
        raise ValueError("All curves are too short (<2 points).")

    def _pad_curve(a: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        a = np.asarray(a, dtype=np.float32)
        x = a[:, 0].astype(np.float32)
        y = a[:, 1].astype(np.float32)

        # sort by x
        order = np.argsort(x)
        x = x[order]; y = y[order]

        # finite
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if x.size < 2:
            x = np.array([0.0, 1e-6], dtype=np.float32)
            y = np.array([0.0, 0.0], dtype=np.float32)

        # dedupe x (keep last)
        if np.any(np.diff(x) == 0):
            _, idx_last = np.unique(x[::-1], return_index=True)
            keep = np.sort(len(x) - 1 - idx_last)
            x = x[keep]; y = y[keep]

        pad_len = Lmax - x.shape[0]
        if pad_len > 0:
            tail = x[-1] + np.arange(1, pad_len + 1, dtype=np.float32) * 1e-6
            x = np.concatenate([x, tail], axis=0)
            y = np.pad(y, (0, pad_len), mode="edge")

        return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)

    xs = []
    ys = []
    for lst in lists:
        real = lst[:M]
        dummy = [np.array([[0.0, 0.0], [1e-6, 0.0]], dtype=np.float32)] * max(0, M - len(real))
        rows = real + dummy

        x_rows = []
        y_rows = []
        for a in rows:
            x_pad, y_pad = _pad_curve(a)
            x_rows.append(x_pad)
            y_rows.append(y_pad)

        xs.append(jnp.stack(x_rows, axis=0))
        ys.append(jnp.stack(y_rows, axis=0))

    x_padded = jnp.stack(xs, axis=0)
    ll_padded = jnp.stack(ys, axis=0)
    return x_padded, ll_padded, jnp.asarray(m_valid, dtype=jnp.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Exploratory: Profile MLEs
# ──────────────────────────────────────────────────────────────────────────────
def compute_profile_mle(df_sim: pd.DataFrame) -> pd.DataFrame:
    grp = df_sim.groupby(["source", "outcome"], sort=True)
    idx = grp["value"].idxmax()
    out = df_sim.loc[idx, ["source", "outcome", "point", "source_orig", "outcome_orig"]].copy()
    out = out.rename(columns={"point": "beta_mle"}).sort_values(["source", "outcome"]).reset_index(drop=True)
    return out

def plot_profile_mle_kde(
    mle_df: pd.DataFrame,
    *,
    out_png: Path,
    title: str,
    xlim: tuple[float, float] | None = None,
    dpi: int = 300,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    sources = sorted(mle_df["source"].unique().tolist())
    if not sources:
        return

    nS = len(sources)
    ncols = 3 if nS >= 3 else nS
    nrows = int(np.ceil(nS / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).reshape(-1)

    xs_all = np.asarray(mle_df["beta_mle"], dtype=float)
    xs_all = xs_all[np.isfinite(xs_all)]
    if xs_all.size == 0:
        plt.close(fig); return

    if xlim is None:
        lo = float(np.quantile(xs_all, 0.01))
        hi = float(np.quantile(xs_all, 0.99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(xs_all.min()) - 1.0, float(xs_all.max()) + 1.0
        xlim = (lo, hi)

    xgrid = np.linspace(xlim[0], xlim[1], 500)

    for i, s in enumerate(sources):
        ax = axes[i]
        vals = np.asarray(mle_df.loc[mle_df["source"] == s, "beta_mle"], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.axis("off"); continue

        ax.hist(vals, bins=min(30, max(5, vals.size)), density=True, alpha=0.35)
        if np.unique(vals).size >= 2 and vals.size >= 3:
            kde = gaussian_kde(vals)
            ax.plot(xgrid, kde(xgrid), linewidth=1.2)
        ax.plot(vals, np.zeros_like(vals), "|", markersize=10, alpha=0.5)

        ax.set_title(f"Source {s} (n={vals.size})")
        ax.set_xlabel(r"$\hat\beta_{s,o}^{MLE}$")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)

    for j in range(nS, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Posterior extraction utilities (NO ArviZ)
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_chain_draw_first(a: np.ndarray) -> np.ndarray:
    """
    Ensure array has shape (chains, draws, ...). Accepts either:
      (chains, draws, ...)
      (draws, ...)   -> treated as single chain
    """
    a = np.asarray(a)
    if a.ndim >= 2:
        return a
    # scalar draws -> (1, draws)
    return a[None, :]

def extract_beta_s_o(
    samples: dict,
    *,
    N_sources: int,
    M_max: int,
) -> np.ndarray:
    """
    Return beta_s_o samples with shape (chains, draws, N_sources, M_max).
    """
    if "beta_s_o" not in samples:
        raise KeyError(f"'beta_s_o' not found in samples. Keys: {sorted(samples.keys())}")

    b = _ensure_chain_draw_first(samples["beta_s_o"])
    # Possible shapes:
    #  (chains, draws, N_sources, M_max)  OR  (chains, draws, M_max, N_sources)
    if b.ndim != 4:
        raise ValueError(f"beta_s_o expected 4D, got shape {b.shape}")

    if b.shape[2] == N_sources and b.shape[3] == M_max:
        return b
    if b.shape[2] == M_max and b.shape[3] == N_sources:
        return np.swapaxes(b, 2, 3)
    # Fallback: pick best guess by matching N_sources
    if b.shape[2] == N_sources:
        return b
    if b.shape[3] == N_sources:
        return np.swapaxes(b, 2, 3)
    raise ValueError(f"Cannot align beta_s_o dims to (N_sources={N_sources}, M_max={M_max}), got {b.shape}")


def stick_breaking_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized stick-breaking.
      v: (..., K) in (0,1)
    Returns:
      w: (..., K) nonnegative, sums to <= 1 (often < 1 unless last v=1)
    """
    v = np.asarray(v, dtype=np.float32)
    # remaining stick after each break
    one_minus = 1.0 - v
    # cumprod over last axis, shifted right by 1 with initial 1
    cp = np.cumprod(one_minus + eps, axis=-1)
    cp_shift = np.concatenate([np.ones_like(cp[..., :1]), cp[..., :-1]], axis=-1)
    return v * cp_shift


def compute_g0_from_samples(
    samples: dict,
    *,
    N_sources: int,
    k: int,
    xgrid: np.ndarray,
    weight_threshold: float = 0.10,
    draw_cap: int = 2000,
    batch_size: int = 250,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Compute posterior mean G0 density on xgrid using a subsample of draws.

    Returns:
      g0_mean: (X,)
      payload: dict with subsampled arrays saved to NPZ (mu, sigma, zeta, xgrid)
    """
    need = ["pi_tilt", "alpha0", "mu", "sigma"]
    missing = [n for n in need if n not in samples]
    if missing:
        raise KeyError(f"G0 needs variables {need}; missing {missing}. Available keys: {sorted(samples.keys())}")

    pi_tilt = _ensure_chain_draw_first(samples["pi_tilt"])   # (C,D,N_sources,K) or (C,D,K)?? expected (C,D,N_sources,K)
    alpha0  = _ensure_chain_draw_first(samples["alpha0"])    # (C,D)
    mu      = _ensure_chain_draw_first(samples["mu"])        # (C,D,K)
    sigma   = _ensure_chain_draw_first(samples["sigma"])     # (C,D,K)

    # Validate shapes
    if pi_tilt.ndim != 4:
        raise ValueError(f"pi_tilt expected 4D (chains,draws,N_sources,K), got {pi_tilt.shape}")
    if pi_tilt.shape[2] != N_sources or pi_tilt.shape[3] != k:
        raise ValueError(f"pi_tilt shape mismatch; expected (*,*,{N_sources},{k}), got {pi_tilt.shape}")

    # Flatten chain/draw to samples
    C, D = pi_tilt.shape[0], pi_tilt.shape[1]
    n_total = C * D
    rng = np.random.default_rng(seed)

    # Choose subset of draws
    n_use = int(min(draw_cap, n_total))
    idx = rng.choice(n_total, size=n_use, replace=False)

    def _take_flat(a: np.ndarray, keep_last_dims: int) -> np.ndarray:
        a = np.asarray(a)
        flat = a.reshape((n_total,) + a.shape[2:]) if a.ndim >= 2 else a.reshape(n_total)
        return flat[idx]

    pi_tilt_sub = _take_flat(pi_tilt, keep_last_dims=2).astype(np.float32)  # (n_use,N_sources,K)
    alpha0_sub  = _take_flat(alpha0, keep_last_dims=0).astype(np.float32)   # (n_use,)
    mu_sub      = _take_flat(mu, keep_last_dims=1).astype(np.float32)       # (n_use,K)
    sigma_sub   = _take_flat(sigma, keep_last_dims=1).astype(np.float32)    # (n_use,K)

    # Compute pi_norm from pi_tilt via stick-breaking + renormalize
    pi = stick_breaking_np(pi_tilt_sub)  # (n_use,N_sources,K)
    pi_sum = np.sum(pi, axis=-1, keepdims=True)
    pi_norm = pi / np.maximum(pi_sum, 1e-12)

    # Compute zeta per draw (n_use,K) using same rule as utils.compute_g0_components
    activated = (pi_norm > float(weight_threshold))  # bool (n_use,N_sources,K)
    m_samps = activated.sum(axis=1).astype(np.float32)  # (n_use,K)
    zeta = (m_samps + alpha0_sub[:, None] / float(k)) / (np.sum(m_samps, axis=1, keepdims=True) + alpha0_sub[:, None])

    # Compute G0 mean density on xgrid in batches to limit peak memory
    xgrid = np.asarray(xgrid, dtype=np.float32)
    X = xgrid.size
    g0_accum = np.zeros((X,), dtype=np.float64)

    const = np.float32(1.0 / np.sqrt(2.0 * np.pi))

    for start in range(0, n_use, batch_size):
        end = min(n_use, start + batch_size)
        mu_b = mu_sub[start:end]      # (B,K)
        sig_b = sigma_sub[start:end]  # (B,K)
        zeta_b = zeta[start:end]      # (B,K)

        # comp pdf: (B,K,X)
        # Use float32 for intermediate; accumulate in float64
        dx = (xgrid[None, None, :] - mu_b[:, :, None]) / np.maximum(sig_b[:, :, None], 1e-6)
        comp = const * np.exp(-0.5 * dx * dx) / np.maximum(sig_b[:, :, None], 1e-6)
        mix = np.sum(zeta_b[:, :, None] * comp, axis=1)  # (B,X)
        g0_accum += mix.astype(np.float64).sum(axis=0)

    g0_mean = (g0_accum / float(n_use)).astype(np.float32)

    payload = {
        "xgrid": xgrid.astype(np.float32),
        "g0": g0_mean.astype(np.float32),
        "mu": mu_sub.astype(np.float32),
        "sigma": sigma_sub.astype(np.float32),
        "zeta": zeta.astype(np.float32),
        "alpha0": alpha0_sub.astype(np.float32),
        "weight_threshold": np.float32(weight_threshold),
    }
    return g0_mean.astype(np.float32), payload


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _kde_or_none(x: np.ndarray, bw: float | str | None = None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.unique(x).size < 2:
        return None
    try:
        return gaussian_kde(x, bw_method=bw)
    except Exception:
        return None

def plot_trace_grid(
    traces: list[tuple[str, np.ndarray]],
    *,
    out_png: Path,
    title: str,
    max_draws: int | None = None,
    dpi: int = 220,
) -> None:
    """
    Create a 2-column trace+hist grid, one row per entry in `traces`.

    traces: list of (label, draws_by_chain) where draws_by_chain has shape (C,D).
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not traces:
        _warn(f"No traces to plot for {out_png.name}")
        return

    # Downsample draws for plotting speed
    plotted = []
    for label, arr in traces:
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a[None, :]
        if max_draws is not None and a.shape[1] > max_draws:
            # thin deterministically
            step = int(np.ceil(a.shape[1] / max_draws))
            a = a[:, ::step]
        plotted.append((label, a))

    n = len(plotted)
    fig, axes = plt.subplots(n, 2, figsize=(11, max(2.2, 1.4 * n)), dpi=dpi,
                             gridspec_kw={"width_ratios": [3.2, 1.0]})
    if n == 1:
        axes = np.array([axes])

    for i, (label, a) in enumerate(plotted):
        ax_t = axes[i, 0]
        ax_h = axes[i, 1]

        # Trace lines
        for c in range(a.shape[0]):
            ax_t.plot(a[c, :], linewidth=0.7, alpha=0.85)
        ax_t.set_ylabel(label)
        ax_t.grid(True, alpha=0.25)

        # Histogram (pooled chains)
        pooled = a.reshape(-1)
        ax_h.hist(pooled[np.isfinite(pooled)], bins=35, density=True, alpha=0.7)
        ax_h.grid(True, alpha=0.25)
        ax_h.set_yticks([])

    axes[-1, 0].set_xlabel("Draw")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_g0_density(
    *,
    xgrid: np.ndarray,
    g0: np.ndarray,
    out_png: Path,
    dpi: int = 300,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=dpi)
    ax.plot(xgrid, g0, linewidth=1.5)
    ax.set_title("Posterior mean $G_0$ density (subsampled draws)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Density")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_g0_trajectories(
    *,
    xgrid: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    zeta: np.ndarray,
    out_png: Path,
    max_curves: int = 120,
    alpha: float = 0.12,
    dpi: int = 300,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = mu.shape[0]
    rng = np.random.default_rng(0)
    picks = rng.choice(n, size=min(max_curves, n), replace=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=dpi)
    ax.plot([], [], alpha=alpha, linewidth=1.0, label="Posterior draws")

    const = 1.0 / np.sqrt(2.0 * np.pi)

    for i in picks:
        dx = (xgrid[None, :] - mu[i, :, None]) / np.maximum(sigma[i, :, None], 1e-6)
        comp = const * np.exp(-0.5 * dx * dx) / np.maximum(sigma[i, :, None], 1e-6)  # (K,X)
        g0_i = np.sum(zeta[i, :, None] * comp, axis=0)
        ax.plot(xgrid, g0_i, alpha=alpha, linewidth=1.0)

    ax.set_title("$G_0$ trajectories (subsampled posterior draws)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_g0_and_pooled_beta(
    *,
    xgrid: np.ndarray,
    g0: np.ndarray,
    pooled_by_src: dict[int, np.ndarray],
    out_png: Path,
    xlim: tuple[float, float] | None = None,
    max_samples_per_src: int = 120_000,
    dpi: int = 300,
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # determine xlim from pooled betas if not given
    if xlim is None:
        pooled_all = np.concatenate([v for v in pooled_by_src.values()], axis=0)
        lo = float(np.quantile(pooled_all, 0.005))
        hi = float(np.quantile(pooled_all, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(pooled_all)), float(np.max(pooled_all))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        xlim = (lo - pad, hi + pad)

    xs = np.linspace(xlim[0], xlim[1], 500)

    fig, ax_beta = plt.subplots(figsize=(6.8, 4.2), dpi=dpi)
    ax_g0 = ax_beta.twinx()

    # pooled beta KDE per source
    for s in sorted(pooled_by_src.keys()):
        samp = pooled_by_src[s]
        if samp.size > max_samples_per_src:
            rng = np.random.default_rng(0)
            samp = rng.choice(samp, size=max_samples_per_src, replace=False)
        kde = _kde_or_none(samp)
        if kde is None:
            continue
        ax_beta.plot(xs, kde(xs), alpha=0.75, label=f"Source {s}")

    # overlay G0 mean
    mask = (xgrid >= xlim[0]) & (xgrid <= xlim[1])
    ax_g0.plot(xgrid[mask], g0[mask], linestyle="--", linewidth=1.2, label="Posterior mean $G_0$")

    ax_beta.set_xlim(*xlim)
    ax_beta.set_xlabel(r"$\beta$")
    ax_beta.set_ylabel("Pooled posterior $\\beta$ density")
    ax_g0.set_ylabel("$G_0$ density")

    h0, l0 = ax_beta.get_legend_handles_labels()
    h1, l1 = ax_g0.get_legend_handles_labels()
    ax_beta.legend(h0 + h1, l0 + l1, loc="best", fontsize="small")

    ax_beta.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_source_density_posterior_vs_mle(
    *,
    pooled_by_src: dict[int, np.ndarray],
    mle_df: pd.DataFrame | None,
    out_dir: Path,
    rep: int,
    xlim: tuple[float, float] | None = None,
    kde_bw: float | str | None = None,
    max_post_samples: int = 120_000,
    max_mle_points: int = 5000,
    dpi: int = 300,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare MLE by source
    mle_by_src: dict[int, np.ndarray] = {}
    if mle_df is not None and not mle_df.empty:
        for s, g in mle_df.groupby("source"):
            vals = np.asarray(g["beta_mle"], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                mle_by_src[int(s)] = vals

    # xlim from posterior if not provided
    if xlim is None:
        all_post = np.concatenate(list(pooled_by_src.values()), axis=0)
        lo = float(np.quantile(all_post, 0.005))
        hi = float(np.quantile(all_post, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(all_post)), float(np.max(all_post))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        xlim = (lo - pad, hi + pad)

    xs = np.linspace(xlim[0], xlim[1], 600)

    # Multi-panel
    sources = sorted(pooled_by_src.keys())
    n = len(sources)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.2 * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).reshape(-1)

    for i, s in enumerate(sources):
        ax = axes[i]
        samp = pooled_by_src[s]
        if samp.size > max_post_samples:
            rng = np.random.default_rng(0)
            samp = rng.choice(samp, size=max_post_samples, replace=False)

        kde_post = _kde_or_none(samp, bw=kde_bw)
        if kde_post is not None:
            ax.plot(xs, kde_post(xs), label="Posterior (pooled outcomes)")
        else:
            ax.axvline(float(np.mean(samp)), label="Posterior")

        if s in mle_by_src:
            vals = mle_by_src[s]
            if vals.size > max_mle_points:
                rng = np.random.default_rng(1)
                vals = rng.choice(vals, size=max_mle_points, replace=False)
            kde_mle = _kde_or_none(vals, bw=kde_bw)
            if kde_mle is not None:
                ax.plot(xs, kde_mle(xs), linestyle="--", label="Profile MLE KDE (outcome-level, within source)")
            else:
                ax.axvline(float(np.mean(vals)), linestyle="--", label="Profile MLE")

        ax.set_title(f"Source {s}")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    grid_png = out_dir / f"rep{rep}_source_density_grid.png"
    fig.savefig(grid_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Per-source figures
    for s in sources:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=dpi)
        samp = pooled_by_src[s]
        if samp.size > max_post_samples:
            rng = np.random.default_rng(0)
            samp = rng.choice(samp, size=max_post_samples, replace=False)

        kde_post = _kde_or_none(samp, bw=kde_bw)
        if kde_post is not None:
            ax.plot(xs, kde_post(xs), label="Posterior (pooled outcomes)")
        else:
            ax.axvline(float(np.mean(samp)), label="Posterior")

        if s in mle_by_src:
            vals = mle_by_src[s]
            if vals.size > max_mle_points:
                rng = np.random.default_rng(1)
                vals = rng.choice(vals, size=max_mle_points, replace=False)
            kde_mle = _kde_or_none(vals, bw=kde_bw)
            if kde_mle is not None:
                ax.plot(xs, kde_mle(xs), linestyle="--", label="Profile MLE KDE (outcome-level, within source)")
            else:
                ax.axvline(float(np.mean(vals)), linestyle="--", label="Profile MLE")

        ax.set_title(f"Source {s} density (rep {rep})")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_png = out_dir / f"rep{rep}_source{s}_density.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_beta_density_single(
    *,
    draws: np.ndarray,
    mle: float | None,
    out_png: Path,
    title: str,
    xlim: tuple[float, float] | None = None,
    max_draws: int = 8_000,
    bins: int = 40,
    dpi: int = 220,
) -> None:
    """
    Lightweight posterior density plot (KDE + histogram) for one beta_{s,o}.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return

    # subsample for plotting speed
    if d.size > max_draws:
        rng = np.random.default_rng(0)
        d = rng.choice(d, size=max_draws, replace=False)

    if xlim is None:
        lo = float(np.quantile(d, 0.005))
        hi = float(np.quantile(d, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(d)), float(np.max(d))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        xlim = (lo - pad, hi + pad)

    xs = np.linspace(xlim[0], xlim[1], 300)
    kde = _kde_or_none(d)

    fig, (ax_kde, ax_hist) = plt.subplots(
        2, 1, figsize=(6.0, 4.0), sharex=True, dpi=dpi,
        gridspec_kw={"height_ratios": (2, 1)}
    )

    if kde is not None:
        ax_kde.plot(xs, kde(xs), linewidth=1.2)
    ax_kde.set_ylabel("Density")
    ax_kde.grid(True, alpha=0.25)
    if mle is not None and np.isfinite(mle):
        ax_kde.axvline(float(mle), linestyle="--", linewidth=1.0)
        ax_hist.axvline(float(mle), linestyle="--", linewidth=1.0)

    ax_hist.hist(d, bins=bins, density=False, alpha=0.65)
    ax_hist.set_ylabel("Freq.")
    ax_hist.set_xlabel(r"$\beta$")
    ax_hist.grid(True, alpha=0.25)

    ax_kde.set_title(title)
    ax_hist.set_xlim(*xlim)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────
def setup_numpyro_cpu(num_chains: int, parallel_chains: bool, n_threads: int | None, use_xla_flags: bool) -> str:
    """
    Configure JAX/NumPyro for CPU execution.
    Returns chain_method for numpyro MCMC.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # IMPORTANT:
    # Some cluster builds of XLA/JAX abort if legacy CPU Eigen threading flags
    # are present in XLA_FLAGS (e.g. "--xla_cpu_multi_thread_eigen_thread_count").
    # We only ever *add* those flags when explicitly requested via `use_xla_flags`.
    # Additionally, if they are already present (e.g., inherited from a login env),
    # proactively remove them to avoid hard crashes.
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" in xla_flags:
        os.environ.pop("XLA_FLAGS", None)

    if n_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        # NOTE: leave XLA_FLAGS alone by default; set use_xla_flags=True only
        # on systems where these flags are supported.
        if use_xla_flags:
            os.environ["XLA_FLAGS"] = (
                "--xla_cpu_multi_thread_eigen=true "
                f"--xla_cpu_multi_thread_eigen_thread_count={n_threads}"
            )

    import numpyro
    import jax

    chain_method = "vectorized"
    if parallel_chains:
        numpyro.set_host_device_count(num_chains)
        if jax.local_device_count() >= num_chains:
            chain_method = "parallel"
        else:
            chain_method = "vectorized"
    return chain_method


def run_hdp_realdata(
    *,
    data_root: Path | None,
    data_file: Path | None,
    family: str,
    K: int,
    output_base: Path,
    seed: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    n_threads: int | None,
    parallel_chains: bool,
    use_vectorized_model: bool,
    outcome_cap: int | None,
    grid_thin: int,
    center_loglik: bool,
    trace_plots: bool,
    trace_beta_sources: int,
    trace_beta_outcomes: int,
    trace_max_draws: int,
    pattern: str,
    recursive: bool,
    profile_mle_xlim: tuple[float, float] | None,
    sources_range: tuple[float, float] | None,
    sources_include: list[str] | None,
    sources_exclude: list[str] | None,
    g0_zoom_xlim: tuple[float, float] | None,
    dense_mass: bool,
    target_accept: float,
    max_tree_depth: int,
    g0_draw_cap: int,
    g0_batch_size: int,
    g0_xgrid_len: int,
    g0_weight_threshold: float,
) -> None:
    chain_method = setup_numpyro_cpu(
        num_chains=num_chains,
        parallel_chains=parallel_chains,
        n_threads=n_threads,
        # Default to NOT setting legacy XLA CPU flags. Some clusters abort on
        # unknown XLA_FLAGS options (observed on cn058).
        use_xla_flags=False,
    )
    _info(f"chain_method={chain_method}  vectorized_model={use_vectorized_model}")

    import jax
    from numpyro.infer import NUTS, MCMC

    # Choose model
    if use_vectorized_model:
        # Prefer the optimized variant (no per-(s,o) deterministic sites unless enabled)
        try:
            from models_vectorized_fast import HDP_model_vectorized as model_fn
            _info("Using models_vectorized_fast.HDP_model_vectorized (HDP_EXPORT_BETA_DETERMINISTICS=0 by default).")
        except Exception:
            from models_vectorized import HDP_model_vectorized as model_fn
    else:
        from models import HDP_model as model_fn
    # Output folders
    output_base = Path(output_base)
    model_dir = output_base / family
    data_out_root = model_dir / "data"
    fig_root = model_dir / "figures"
    beta_fig_dir = fig_root / "beta_plots"
    trace_fig_dir = fig_root / "trace_plots"
    mle_fig_dir = fig_root / "profile_mle"
    src_density_dir = fig_root / "source_density"
    for p in [data_out_root, beta_fig_dir, trace_fig_dir, mle_fig_dir, src_density_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # reps
    if data_file is not None:
        rep_paths = [Path(data_file)]
        _info(f"Using single data file as rep1: {Path(data_file).resolve()}")
    else:
        if data_root is None:
            raise ValueError("Either --data-root or --data-file is required.")
        rep_paths = discover_rep_dirs(Path(data_root))
        _info(f"Discovered {len(rep_paths)} rep folder(s) under {data_root}")

    for rep_idx, rep_path in enumerate(rep_paths, start=1):
        rep = rep_idx
        m = re.fullmatch(r"rep(\d+)", rep_path.name)
        if m:
            rep = int(m.group(1))

        _info(f"[rep {rep}] Loading data from {rep_path}")
        df_raw = load_profile_likelihoods(rep_path, pattern=pattern, recursive=recursive)
        # Optional: filter by raw source labels BEFORE remapping.
        if sources_range is not None or sources_include is not None or sources_exclude is not None:
            n0_rows = len(df_raw)
            n0_src = df_raw["source_raw"].nunique()
            df_raw = filter_sources_df(
                df_raw,
                sources_range=sources_range,
                sources_include=sources_include,
                sources_exclude=sources_exclude,
            )
            n1_rows = len(df_raw)
            n1_src = df_raw["source_raw"].nunique()
            _info(f"[rep {rep}] Source filter applied: rows {n0_rows}→{n1_rows}, sources {n0_src}→{n1_src}")
            if n1_rows == 0:
                raise RuntimeError(f"[rep {rep}] Source filter removed all rows. Check sources_range/include/exclude.")
        df_sim, maps, mapping_df = remap_ids(df_raw, outcome_cap=outcome_cap)

        # Center loglik per curve: improves numerical stability; does NOT change argmax
        if center_loglik:
            df_sim["value"] = df_sim["value"] - df_sim.groupby(["source", "outcome"])["value"].transform("max")

        # Thin grid points per curve to speed loglik interpolation
        if grid_thin and grid_thin > 1:
            kept = []
            for (s, o), g in df_sim.groupby(["source", "outcome"], sort=True):
                g = g.sort_values("point").drop_duplicates(subset=["point"], keep="last")
                g_th = g.iloc[::grid_thin, :].copy()
                if g_th["point"].iloc[-1] != g["point"].iloc[-1]:
                    g_th = pd.concat([g_th, g.iloc[[-1]]], ignore_index=True)
                kept.append(g_th)
            df_sim = pd.concat(kept, ignore_index=True).sort_values(["source", "outcome", "point"]).reset_index(drop=True)

        N_sources = int(df_sim["source"].nunique())
        outcomes_by_source = df_sim.groupby("source")["outcome"].nunique().to_dict()
        M_max = int(max(outcomes_by_source.values(), default=0))
        if N_sources <= 0 or M_max <= 0:
            raise RuntimeError("No sources/outcomes after filtering.")

        # Build per-source outcome list
        source_outcome_data: Dict[int, list] = {}
        for s in range(1, N_sources + 1):
            df_s = df_sim[df_sim["source"] == s]
            out_list = []
            for o in sorted(df_s["outcome"].unique()):
                arr = df_s[df_s["outcome"] == o][["point", "value"]].to_numpy(dtype=np.float32)
                out_list.append(arr)
            source_outcome_data[s] = out_list
        m_valid = np.array([len(source_outcome_data[s]) for s in range(1, N_sources + 1)], dtype=int)

        # Rep output
        rep_out_dir = data_out_root / f"rep{rep}"
        rep_out_dir.mkdir(parents=True, exist_ok=True)

        # Save provenance
        df_sim.to_csv(rep_out_dir / "profile_likelihood_used.csv", index=False)
        mapping_df.to_csv(rep_out_dir / "mapping_outcomes.csv", index=False)
        pd.DataFrame([{"source": v, "source_orig": k} for k, v in maps.source_map.items()]).sort_values("source") \
          .to_csv(rep_out_dir / "mapping_sources.csv", index=False)

        # RNG
        rep_seed = int(np.random.SeedSequence([seed, rep]).generate_state(1, dtype=np.uint32)[0])
        rng_key = jax.random.PRNGKey(rep_seed)

        _info(f"[rep {rep}] MCMC: N_sources={N_sources} M_max={M_max} K={K} seed={rep_seed}")
        kernel = NUTS(model_fn, dense_mass=bool(dense_mass), target_accept_prob=float(target_accept), max_tree_depth=int(max_tree_depth))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method=chain_method)

        if use_vectorized_model:
            x_padded, ll_padded, m_valid_j = pack_sources_vectorized(source_outcome_data, M_outcomes=M_max)
            mcmc.run(rng_key, x_padded, ll_padded, m_valid_j, N_sources=N_sources, k=K)
        else:
            num_outcomes_dict = {s: len(source_outcome_data[s]) for s in source_outcome_data}
            data_point_mean = float(df_sim["point"].mean())
            mcmc.run(
                rng_key,
                source_outcome_data=source_outcome_data,
                num_outcomes_dict=num_outcomes_dict,
                N_sources=N_sources,
                k=K,
                data_point_mean=data_point_mean,
            )

        # Posterior samples (NO InferenceData)
        samples = mcmc.get_samples(group_by_chain=True)

        # Trace plots (lightweight)
        if trace_plots:
            hypers: list[tuple[str, np.ndarray]] = []
            for name in ["gamma", "alpha0", "beta_tilt"]:
                if name in samples:
                    arr = _ensure_chain_draw_first(samples[name])
                    # beta_tilt is (C,D,K) -> add each component
                    if arr.ndim == 3:
                        for j in range(arr.shape[2]):
                            hypers.append((f"{name}[{j}]", arr[:, :, j]))
                    else:
                        hypers.append((name, arr))
            for name in ["mu", "sigma"]:
                if name in samples:
                    arr = _ensure_chain_draw_first(samples[name])
                    if arr.ndim == 3:
                        for j in range(arr.shape[2]):
                            hypers.append((f"{name}[{j}]", arr[:, :, j]))
                    else:
                        hypers.append((name, arr))

            plot_trace_grid(
                hypers,
                out_png=trace_fig_dir / f"rep{rep}_trace_hypers.png",
                title=f"Trace plots (hyperparameters) — rep {rep}",
                max_draws=trace_max_draws,
                dpi=220,
            )

            # Beta traces: small subset only
            beta_s_o = extract_beta_s_o(samples, N_sources=N_sources, M_max=M_max)
            beta_traces: list[tuple[str, np.ndarray]] = []
            s_max = min(int(trace_beta_sources), N_sources)
            for s in range(1, s_max + 1):
                o_max = min(int(trace_beta_outcomes), int(m_valid[s - 1]))
                for o in range(1, o_max + 1):
                    beta_traces.append((f"beta[{s},{o}]", beta_s_o[:, :, s - 1, o - 1]))

            plot_trace_grid(
                beta_traces,
                out_png=trace_fig_dir / f"rep{rep}_trace_betas.png",
                title=f"Trace plots (subset of $\\beta_{{s,o}}$) — rep {rep}",
                max_draws=trace_max_draws,
                dpi=220,
            )
        else:
            beta_s_o = extract_beta_s_o(samples, N_sources=N_sources, M_max=M_max)

        # Profile MLE summaries/plots
        mle_df = compute_profile_mle(df_sim)
        mle_df.to_csv(rep_out_dir / "profile_mle_by_outcome.csv", index=False)
        (mle_df.groupby(["source", "source_orig"], as_index=False)["beta_mle"]
              .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
              .to_csv(rep_out_dir / "profile_mle_by_source.csv", index=False))

        plot_profile_mle_kde(
            mle_df,
            out_png=mle_fig_dir / f"rep{rep}_profile_mle_kde.png",
            title=f"Profile-likelihood MLE across outcomes (rep {rep})",
            xlim=profile_mle_xlim,
            dpi=300,
        )

        # Save beta NPZ files and optional per-(s,o) posterior density plots
        pooled_by_src: dict[int, np.ndarray] = {}
        curve_pairs = df_sim.groupby(["source", "outcome"], sort=True)

        _info(f"[rep {rep}] Writing beta NPZ archives for {len(curve_pairs)} (source,outcome) curves...")
        for (s, o), g in curve_pairs:
            if o > int(m_valid[s - 1]):
                continue  # safety (shouldn't happen)

            draws = beta_s_o[:, :, s - 1, o - 1].reshape(-1).astype(np.float32)
            grid = g["point"].to_numpy(dtype=np.float32)
            loglik = g["value"].to_numpy(dtype=np.float32)

            out_npz = rep_out_dir / f"beta_{s}_{o}.npz"
            np.savez_compressed(out_npz, samples=draws, grid=grid, loglik=loglik)

            # Optionally: one density plot per (s,o) — can be many files.
            # We keep it ON because you asked for "all corresponding figure",
            # but it is the most time-consuming part. If you ever want to skip,
            # add a flag and set it false.
            title = f"Posterior density for beta_{s}_{o} (rep {rep})"
            mle_row = mle_df[(mle_df["source"] == s) & (mle_df["outcome"] == o)]
            mle_val = float(mle_row["beta_mle"].iloc[0]) if len(mle_row) == 1 else None
            out_png = beta_fig_dir / f"rep{rep}_beta_{s}_{o}.png"
            plot_beta_density_single(draws=draws, mle=mle_val, out_png=out_png, title=title, xlim=profile_mle_xlim)

        # Per-source pooled betas (for source-level density + g0 overlay)
        _info(f"[rep {rep}] Building per-source pooled beta samples...")
        for s in range(1, N_sources + 1):
            o_s = int(m_valid[s - 1])
            samp = beta_s_o[:, :, s - 1, :o_s].reshape(-1).astype(np.float32)
            pooled_by_src[s] = samp
            np.savez_compressed(rep_out_dir / f"beta_{s}.npz", samples=samp)

        # Source-level density vs MLE
        plot_source_density_posterior_vs_mle(
            pooled_by_src=pooled_by_src,
            mle_df=mle_df,
            out_dir=src_density_dir,
            rep=rep,
            xlim=profile_mle_xlim,
        )

        # G0 plots (subsampled)
        try:
            # Choose xgrid range
            if profile_mle_xlim is not None:
                xmin, xmax = profile_mle_xlim
            else:
                all_post = np.concatenate(list(pooled_by_src.values()), axis=0)
                xmin = float(np.quantile(all_post, 0.005))
                xmax = float(np.quantile(all_post, 0.995))
                if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
                    xmin, xmax = float(np.min(all_post)), float(np.max(all_post))
                pad = 0.08 * (xmax - xmin) if xmax > xmin else 1.0
                xmin, xmax = xmin - pad, xmax + pad

            # Ensure the evaluation grid covers the requested zoom range, if any.
            if g0_zoom_xlim is not None:
                zlo, zhi = float(g0_zoom_xlim[0]), float(g0_zoom_xlim[1])
                if np.isfinite(zlo) and np.isfinite(zhi):
                    if zlo > zhi:
                        zlo, zhi = zhi, zlo
                    xmin = float(min(xmin, zlo))
                    xmax = float(max(xmax, zhi))
            xgrid = np.linspace(xmin, xmax, int(g0_xgrid_len), dtype=np.float32)

            g0, payload = compute_g0_from_samples(
                samples,
                N_sources=N_sources,
                k=K,
                xgrid=xgrid,
                weight_threshold=float(g0_weight_threshold),
                draw_cap=int(g0_draw_cap),
                batch_size=int(g0_batch_size),
                seed=rep_seed,
            )
            npz_path = rep_out_dir / "g0_components_subsampled.npz"
            np.savez_compressed(npz_path, **payload)

            plot_g0_density(
                xgrid=payload["xgrid"],
                g0=payload["g0"],
                out_png=beta_fig_dir / f"rep{rep}_g0.png",
            )
            plot_g0_trajectories(
                xgrid=payload["xgrid"],
                mu=payload["mu"],
                sigma=payload["sigma"],
                zeta=payload["zeta"],
                out_png=beta_fig_dir / f"rep{rep}_g0_trajs.png",
            )
            plot_g0_and_pooled_beta(
                xgrid=payload["xgrid"],
                g0=payload["g0"],
                pooled_by_src=pooled_by_src,
                out_png=beta_fig_dir / f"rep{rep}_g0_beta.png",
                xlim=(xmin, xmax),
            )
            # Also write a zoomed-in version (e.g. x in [-10,10]) if requested.
            if g0_zoom_xlim is not None:
                zlo, zhi = float(g0_zoom_xlim[0]), float(g0_zoom_xlim[1])
                if zlo > zhi:
                    zlo, zhi = zhi, zlo
                tag = f"{zlo:g}_{zhi:g}".replace(".", "p")
                mask = (payload["xgrid"] >= zlo) & (payload["xgrid"] <= zhi)
                if int(np.sum(mask)) >= 10:
                    xz = payload["xgrid"][mask]
                    gz = payload["g0"][mask]
                    plot_g0_density(
                        xgrid=xz,
                        g0=gz,
                        out_png=beta_fig_dir / f"rep{rep}_g0_xlim_{tag}.png",
                    )
                    plot_g0_trajectories(
                        xgrid=xz,
                        mu=payload["mu"],
                        sigma=payload["sigma"],
                        zeta=payload["zeta"],
                        out_png=beta_fig_dir / f"rep{rep}_g0_trajs_xlim_{tag}.png",
                    )
                    plot_g0_and_pooled_beta(
                        xgrid=payload["xgrid"],
                        g0=payload["g0"],
                        pooled_by_src=pooled_by_src,
                        out_png=beta_fig_dir / f"rep{rep}_g0_beta_xlim_{tag}.png",
                        xlim=(zlo, zhi),
                    )
                else:
                    _warn(f"[rep {rep}] g0_zoom_xlim={g0_zoom_xlim} produced too few xgrid points to plot.")
        except Exception as e:
            _warn(f"[rep {rep}] G0 plotting skipped/failed: {e}")

        # Cleanup large objects before next rep (important for low-mem jobs)
        del samples
        del beta_s_o
        del pooled_by_src
        gc.collect()

        _info(f"[rep {rep}] Done.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main_realdata_optimized",
        description="Run HDP on real profile-likelihood curves and generate diagnostic/exploratory figures (optimized).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run-hdp", help="Run HDP on real data (no simulation)")

    grp = run_p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--data-root", type=Path, default=None, help="Folder containing rep*/ or directly containing CSVs.")
    grp.add_argument("--data-file", type=Path, default=None, help="Single long-format CSV file. Treated as rep1.")

    run_p.add_argument("--pattern", default="*.csv", help="Glob for CSVs inside each rep folder (default: *.csv).")
    run_p.add_argument("--recursive", action="store_true", help="Recursively glob CSVs under each rep folder.")
    run_p.add_argument("--family", type=str, default="realdata", help="Output subfolder under output-base.")

    run_p.add_argument("--K", type=int, required=True, help="Number of global mixture components.")
    run_p.add_argument("--seed", type=int, default=0, help="Random seed.")
    run_p.add_argument("--output-base", type=Path, default=None, help="Results folder (default: results_real_YYYY-MM-DD).")

    # MCMC
    run_p.add_argument("--num-warmup", type=int, default=2000)
    run_p.add_argument("--num-samples", type=int, default=4000)
    run_p.add_argument("--num-chains", type=int, default=2)
    run_p.add_argument("--n-threads", type=int, default=None)
    run_p.add_argument("--parallel-chains", action="store_true", help="Try to run chains in parallel (CPU virtual devices).")
    run_p.add_argument("--use-vectorized-model", action="store_true", help="Use models_vectorized.HDP_model_vectorized (recommended).")

    run_p.add_argument("--dense-mass", action="store_true", default=True, help="Use dense mass matrix in NUTS (default on).")
    run_p.add_argument("--no-dense-mass", dest="dense_mass", action="store_false", help="Use diagonal mass matrix (faster, less memory).")
    run_p.add_argument("--target-accept", type=float, default=0.90)
    run_p.add_argument("--max-tree-depth", type=int, default=12)

    # Data handling
    run_p.add_argument("--outcome-cap", type=int, default=None, help="Use only first M outcomes per source (after remapping).")
    # Optional: restrict which sources to use (by raw labels in the input CSV)
    run_p.add_argument("--sources-range", nargs=2, type=float, default=None, metavar=("S_MIN", "S_MAX"),
                       help="Keep only sources whose raw labels are numeric in [S_MIN, S_MAX] (inclusive). "
                            "Example: --sources-range 4 12")
    run_p.add_argument("--sources-include", nargs="+", default=None, metavar="SRC",
                       help="Keep only this explicit list of source raw labels (strings).")
    run_p.add_argument("--sources-exclude", nargs="+", default=None, metavar="SRC",
                       help="Exclude these source raw labels (strings).")
    run_p.add_argument("--grid-thin", type=int, default=1, help="Keep every nth grid point per curve (default 1 = keep all).")
    run_p.add_argument("--no-center-loglik", dest="center_loglik", action="store_false", help="Disable per-curve centering.")
    run_p.set_defaults(center_loglik=True)

    # Trace plots
    run_p.add_argument("--trace-plots", action="store_true", default=False, help="Write lightweight trace plots.")
    run_p.add_argument("--trace-beta-sources", type=int, default=3, help="How many sources to include in beta trace subset.")
    run_p.add_argument("--trace-beta-outcomes", type=int, default=3, help="How many outcomes per source in beta trace subset.")
    run_p.add_argument("--trace-max-draws", type=int, default=4000, help="Max draws per chain to plot in traces (thins if larger).")

    # Profile MLE plot range
    run_p.add_argument("--profile-mle-xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    # Optional: write additional zoomed-in G0 plots on a fixed x-range (e.g. -10 10)
    run_p.add_argument("--g0-zoom-xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"),
                       help="If provided, also write zoomed G0 plots restricted to [XMIN, XMAX]. "
                            "Example: --g0-zoom-xlim -10 10")

    # G0 computation knobs
    run_p.add_argument("--g0-draw-cap", type=int, default=2000, help="Max posterior draws to use for G0 (subsampling).")
    run_p.add_argument("--g0-batch-size", type=int, default=250, help="Batch size for G0 computation.")
    run_p.add_argument("--g0-xgrid-len", type=int, default=2000, help="Grid length for G0 plot.")
    run_p.add_argument("--g0-weight-threshold", type=float, default=0.10, help="Activation threshold used in zeta computation.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.command == "run-hdp":
        out_base = args.output_base
        if out_base is None:
            today = date.today().strftime("%Y-%m-%d")
            out_base = Path(f"results_real_{today}")
        out_base = Path(out_base)

        profile_xlim = None
        if args.profile_mle_xlim is not None:
            profile_xlim = (float(args.profile_mle_xlim[0]), float(args.profile_mle_xlim[1]))

        run_hdp_realdata(
            data_root=args.data_root,
            data_file=args.data_file,
            family=args.family,
            K=int(args.K),
            output_base=out_base,
            seed=int(args.seed),
            num_warmup=int(args.num_warmup),
            num_samples=int(args.num_samples),
            num_chains=int(args.num_chains),
            n_threads=args.n_threads,
            parallel_chains=bool(args.parallel_chains),
            use_vectorized_model=bool(args.use_vectorized_model),
            outcome_cap=args.outcome_cap,
            grid_thin=int(args.grid_thin),
            center_loglik=bool(args.center_loglik),
            trace_plots=bool(args.trace_plots),
            trace_beta_sources=int(args.trace_beta_sources),
            trace_beta_outcomes=int(args.trace_beta_outcomes),
            trace_max_draws=int(args.trace_max_draws),
            pattern=str(args.pattern),
            recursive=bool(args.recursive),
            profile_mle_xlim=profile_xlim,
            sources_range=(float(args.sources_range[0]), float(args.sources_range[1])) if args.sources_range is not None else None,
            sources_include=list(args.sources_include) if args.sources_include is not None else None,
            sources_exclude=list(args.sources_exclude) if args.sources_exclude is not None else None,
            g0_zoom_xlim=(float(args.g0_zoom_xlim[0]), float(args.g0_zoom_xlim[1])) if args.g0_zoom_xlim is not None else None,
            dense_mass=bool(args.dense_mass),
            target_accept=float(args.target_accept),
            max_tree_depth=int(args.max_tree_depth),
            g0_draw_cap=int(args.g0_draw_cap),
            g0_batch_size=int(args.g0_batch_size),
            g0_xgrid_len=int(args.g0_xgrid_len),
            g0_weight_threshold=float(args.g0_weight_threshold),
        )


if __name__ == "__main__":
    main()
