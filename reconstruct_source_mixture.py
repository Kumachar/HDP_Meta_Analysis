
import os, re, argparse
import numpy as np
import matplotlib.pyplot as plt

def _normal_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*z*z

def reconstruct_source_mixture_from_betas(betas_so, mu, sigma, beta_global, alpha0, grid=None):
    """
    betas_so:     (O,)  outcome-level beta *summaries* (e.g., posterior means) for one source
    mu, sigma:    (K,)  global component params (posterior means or a single draw)
    beta_global:  (K,)  global base weights from stick-breaking (means or a single draw)
    alpha0:       scalar concentration (posterior mean, or a sensible constant like 1.0)
    returns: grid (G,), density (G,), pi_hat (K,), responsibilities (O,K)
    """
    betas_so = np.asarray(betas_so)
    mu = np.asarray(mu); sigma = np.asarray(sigma); beta_global = np.asarray(beta_global)
    O, K = betas_so.shape[0], mu.shape[0]

    # Responsibilities r_{o j} ∝ beta_j * N(betas_so[o] | mu_j, sigma_j)
    loglik = _normal_logpdf(betas_so[:, None], mu[None, :], sigma[None, :])           # (O,K)
    logits = np.log(beta_global + 1e-300)[None, :] + loglik                            # (O,K)
    logits = logits - logits.max(axis=1, keepdims=True)                                # stabilize
    r = np.exp(logits); r = r / r.sum(axis=1, keepdims=True)                           # (O,K)

    # Soft counts and Dirichlet posterior mean
    n = r.sum(axis=0)                                                                   # (K,)
    pi_hat = (alpha0 * beta_global + n) / (alpha0 + O)                                  # (K,)

    # Mixture density on a grid
    if grid is None:
        lo = min(betas_so.min(), (mu - 4*sigma).min())
        hi = max(betas_so.max(), (mu + 4*sigma).max())
        grid = np.linspace(lo, hi, 600)
    comp = np.exp(_normal_logpdf(grid[:, None], mu[None, :], sigma[None, :]))          # (G,K)
    density = comp @ pi_hat                                                             # (G,)
    return grid, density, pi_hat, r

def _try_load_g0_components(g0_path, alpha0_override=None):
    """
    Tries to load mu, sigma, beta (global) and alpha0 from g0_components.npz.
    Falls back to heuristics if some keys are missing.
    """
    if not os.path.exists(g0_path):
        raise FileNotFoundError(f"Could not find {g0_path}.")
    data = np.load(g0_path, allow_pickle=True)
    keys = set(data.files)

    # Heuristic key search
    def find_key(candidates):
        for k in candidates:
            if k in keys:
                return k
        # fuzzy search
        for k in keys:
            for cand in candidates:
                if cand in k.lower():
                    return k
        return None

    k_mu = find_key(["mu", "mus", "g0_mu", "mu_components"])
    k_sigma = find_key(["sigma", "sigmas", "g0_sigma", "sigma_components", "sd", "sds"])
    k_beta = find_key(["beta", "g0_beta", "beta_components", "weights", "w"])
    k_alpha = find_key(["alpha0", "alpha", "alpha_0"])

    if k_mu is None or k_sigma is None or k_beta is None:
        raise KeyError(f"g0_components.npz is missing required arrays. Found keys: {sorted(keys)}")

    mu = np.asarray(data[k_mu]).astype(float).ravel()
    sigma = np.asarray(data[k_sigma]).astype(float).ravel()
    beta = np.asarray(data[k_beta]).astype(float).ravel()

    if k_alpha is not None:
        alpha0 = float(np.asarray(data[k_alpha]).mean())
    else:
        alpha0 = 1.0 if alpha0_override is None else float(alpha0_override)

    return mu, sigma, beta, alpha0

def _parse_s_o_from_filename(fname):
    """
    Try common patterns like: beta_3_7.npz, posterior_beta_s3_o7.npz, beta_s3_o7.npz, beta_3-7.npz
    Returns (s, o) or (None, None) if not found.
    """
    base = os.path.basename(fname)
    patterns = [
        r"beta[_-](\d+)[_-](\d+)\.npz",
        r"beta[_-]s(\d+)[_-]o(\d+)\.npz",
        r"posterior[_-]?beta[_-]s(\d+)[_-]o(\d+)\.npz",
        r"beta[_-](\d+)\.npz",  # only source index (treat as o=1)
    ]
    for pat in patterns:
        m = re.search(pat, base, flags=re.IGNORECASE)
        if m:
            nums = [int(x) for x in m.groups()]
            if len(nums) == 1:
                return nums[0], 1
            return nums[0], nums[1]
    return None, None

def _first_array_in_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    # Try common keys first
    for key in ["samples", "beta_samples", "arr_0", "values", "posterior"]:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim >= 1:
                return arr
    # Fallback to the first 1D array we can find
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim >= 1:
            return arr
    raise KeyError(f"No suitable array found in {npz_path}. Keys: {data.files}")

def collect_betas_for_source(rep_dir, source, outcome_cap=None):
    """
    Scans rep_dir for per-outcome beta posterior npz files and returns
    a dict outcome_index -> posterior_mean_beta for the specified source.
    """
    betas = {}
    for root, _, files in os.walk(rep_dir):
        for fname in files:
            if not fname.lower().endswith(".npz"):
                continue
            if "g0_components" in fname.lower():
                continue
            if "beta" not in fname.lower():
                continue
            s, o = _parse_s_o_from_filename(fname)
            if s is None:
                continue
            if s != source:
                continue
            try:
                arr = _first_array_in_npz(os.path.join(root, fname))
            except Exception:
                continue
            # reduce to a scalar summary per outcome; posterior mean by default
            m = float(np.mean(arr))
            betas[o] = m
    if not betas:
        raise FileNotFoundError(f"No per-outcome beta NPZ files found for source {source} under {rep_dir}")
    # sort by outcome index and respect optional cap
    items = sorted(betas.items())
    if outcome_cap is not None:
        items = items[:outcome_cap]
    return np.array([m for _, m in items], dtype=float)

def main():
    ap = argparse.ArgumentParser(description="Reconstruct a source-level mixture from per-outcome beta posteriors.")
    ap.add_argument("--results-root", required=True, help="Path like results_sample_2025-09-12")
    ap.add_argument("--family", required=True, help="Model family subfolder, e.g. linear / poisson / logistic")
    ap.add_argument("--rep", type=int, required=True, help="Repetition index, e.g. 1")
    ap.add_argument("--source", type=int, required=True, help="Source index (1-based)")
    ap.add_argument("--alpha0", type=float, default=None, help="Override alpha0 if not present in g0 npz")
    ap.add_argument("--outcomes", type=int, default=None, help="Use only the first O outcomes (optional)")
    ap.add_argument("--save-prefix", default=None, help="Prefix for saved outputs (PNG/CSV). Defaults to auto.")
    args = ap.parse_args()

    # Expected layout: results_root/family/data/rep{rep}/
    rep_dir = os.path.join(args.results_root, args.family, "data", f"rep{args.rep}")
    if not os.path.isdir(rep_dir):
        raise FileNotFoundError(f"Expected directory not found: {rep_dir}")

    # 1) Load g0 components
    g0_path = os.path.join(rep_dir, "g0_components.npz")
    mu, sigma, beta, alpha0 = _try_load_g0_components(g0_path, alpha0_override=args.alpha0)

    # 2) Gather per-outcome beta summaries for this source
    betas_so = collect_betas_for_source(rep_dir, args.source, outcome_cap=args.outcomes)

    # 3) Reconstruct mixture
    grid, density, pi_hat, resp = reconstruct_source_mixture_from_betas(betas_so, mu, sigma, beta, alpha0)

    # 4) Save plot + weights
    base = args.save_prefix or f"source{str(args.source)}_{args.family}_rep{args.rep}"
    out_png = os.path.join(rep_dir, f"reconstructed_mixture_{base}.png")
    out_csv = os.path.join(rep_dir, f"reconstructed_weights_{base}.csv")

    plt.figure()
    plt.plot(grid, density, label="Reconstructed mixture")
    # also show components scaled by pi_hat
    for j in range(len(mu)):
        comp = (1/np.sqrt(2*np.pi)/sigma[j]) * np.exp(-0.5*((grid - mu[j])/sigma[j])**2)
        plt.plot(grid, comp * pi_hat[j], alpha=0.5)
    plt.scatter(betas_so, np.zeros_like(betas_so), marker="x", label="beta_s,o summaries")
    plt.title(f"Source {args.source}: reconstructed mixture ({args.family}, rep {args.rep})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Save weights
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component_index", "pi_hat_j", "mu_j", "sigma_j"])
        for j, (pj, muj, sj) in enumerate(zip(pi_hat, mu, sigma), start=1):
            w.writerow([j, float(pj), float(muj), float(sj)])

    print(f"Saved mixture plot: {out_png}")
    print(f"Saved weights CSV: {out_csv}")
    print(f"Used O={len(betas_so)} outcomes; alpha0={alpha0:.4f}; K={len(mu)}")

if __name__ == "__main__":
    main()
