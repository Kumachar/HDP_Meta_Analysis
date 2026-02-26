# SIM_DATA_GUIDE

This guide describes the **simulated artifacts written to disk** by the simulation workflow (`main.py`) and how to load them.

It focuses on the file formats actually used in the provided code snapshot:

- **`.npz`** (NumPy compressed archives) — primary storage for truth and posterior “archives”
- **`.csv`** — summary tables and (in real-data runs) cleaned profile-likelihood inputs
- **`.npy`** — supported by the loaders, but not the default output in `main.py`

No `.rds` or Python `pickle` files are written or read in the provided code.

---

## Where simulated data live on disk

A simulated run creates a results directory of the form:

```
<RESULTS_ROOT>/<FAMILY>/data/rep<r>/
```

where:

- `<RESULTS_ROOT>` is `--output-base` (e.g. `results_sample_2025-10-20`)
- `<FAMILY>` is `--model-type` (one of `linear`, `poisson`, `logistic`)
- `<r>` is the repetition index (`--rep`)

Inside each `rep<r>/` directory, `main.py` writes:

- `truth_source_mixtures.npz` — the simulation truth for mixture components and source weights
- `beta_<s>_<o>.npz` — posterior draws for β_{s,o} plus the profile-likelihood curve used
- `g0_components.npz` — posterior draws / derived quantities for the global mixture G0
- `beta_summary_stats.csv` — per-source summary table (posterior vs truth summaries)

---

## 1) The raw simulated profile-likelihood curves (in memory)

Simulation is done by calling one of these functions in `simulation.py`:

- `simulate_profile_likelihoods_linreg(...)`
- `simulate_profile_likelihoods_logistic(...)`
- `simulate_profile_likelihoods_poisson(...)`

Each simulator returns:

```python
df_sim, true_pis, beta_df, est_df = simulate_profile_likelihoods_poisson(...)
```

### `df_sim` structure

`df_sim` is a long table with one row per grid point:

| column | type | meaning |
|---|---|---|
| `source` | int (1-based) | source index s |
| `outcome` | int (1-based) | outcome index o within source |
| `point` | float | β grid value |
| `value` | float | profile log-likelihood at that β (summed over observations) |

Minimal runnable example:

```python
import numpy as np
from simulation import simulate_profile_likelihoods_poisson

grid = np.linspace(-10, 10, 20000)

df_sim, pis_true, beta_df, est_df = simulate_profile_likelihoods_poisson(
    K=5, S=8, O=20, n_obs=1000,
    grid=grid,
    seed=42,
)

print(df_sim.head())
# columns: source, outcome, point, value
```

How it is used:

- `main.py` groups `df_sim` by `(source,outcome)` to build per-curve arrays of shape `(G,2)` holding `(point, value)`.
- Those arrays are fed to the HDP model, which interpolates `value` at sampled β values.

---

## 2) Truth: `truth_source_mixtures.npz`

Written by `main.py` once per rep:

```python
np.savez(rep_dir / "truth_source_mixtures.npz",
         beta_mean=beta_mean, beta_sds=beta_sds, pis=pis_true)
```

### Keys and shapes

| key | shape | meaning |
|---|---:|---|
| `pis` | `(S, K)` | true source-level mixture weights π_s over K components |
| `beta_mean` | `(K,)` | component means μ_k used to generate β_{s,o} |
| `beta_sds` | `(K,)` | component standard deviations σ_k used to generate β_{s,o} |

Load example:

```python
from pathlib import Path
import numpy as np

rep_dir = Path("results_sample_2025-10-20/poisson/data/rep1")

truth = np.load(rep_dir / "truth_source_mixtures.npz")
pis = truth["pis"]              # (S,K)
beta_mean = truth["beta_mean"]  # (K,)
beta_sds  = truth["beta_sds"]   # (K,)

print(pis.shape, beta_mean.shape, beta_sds.shape)
```

How it is used:

- `plot_source_density_coverage.py` uses it to compute the **true density** for each source.
- `main.py` uses it to compute true source-level mean/median/mode for evaluation summaries.

---

## 3) Posterior archives: `beta_<s>_<o>.npz`

Written by `utils.store_beta_posteriors(...)` during `main.py` runs, one file per `(source,outcome)`.

### Keys

| key | shape | meaning |
|---|---:|---|
| `samples` | `(N,)` | posterior draws of β_{s,o} flattened across chains and draws |
| `grid` | `(G,)` | β grid used for the profile likelihood curve |
| `loglik` | `(G,)` | profile log-likelihood values aligned to `grid` |
| `prior_samples` | `(M,)` (often `M=1`) | simulated “true” β_{s,o} values (if available) |

Load example:

```python
from pathlib import Path
import numpy as np

rep_dir = Path("results_sample_2025-10-20/poisson/data/rep1")
dat = np.load(rep_dir / "beta_1_1.npz")

samples = dat["samples"]   # posterior draws
grid = dat["grid"]         # β grid
loglik = dat["loglik"]     # profile log-likelihood curve

beta_true = None
if "prior_samples" in dat.files and dat["prior_samples"].size:
    beta_true = float(dat["prior_samples"].mean())

print(samples.shape, grid.shape, loglik.shape, beta_true)
```

How it is used:

- **Diagnostics / plots**: many scripts read these NPZs directly (`plot_ci_truth.py`, `explore_profile_mle_kde.py`, `regen_trace_plots.py`).
- **Reconstruction**: `reconstruct_source_mixture.py` reads posterior means of β_{s,o} (or other summaries) to estimate a source-level mixture.

> Note: In real-data runs (`main_realdata.py`), `beta_<s>_<o>.npz` is also written, but typically **without** `prior_samples`.

---

## 4) Global mixture posterior: `g0_components.npz`

Written by `utils.compute_g0_components(...)` from an ArviZ `InferenceData`.

### Keys (as written by `compute_g0_components`)

| key | typical shape | meaning |
|---|---:|---|
| `xgrid` | `(G,)` | β grid used to evaluate the G0 density |
| `mu` | `(T, K)` | posterior draws of component means μ_k (flattened across chain×draw) |
| `sigma` | `(T, K)` | posterior draws of component scales σ_k |
| `pi_norm` | `(T, S, K)` | posterior draws of source weights π_s (normalized) |
| `alpha0` | `(T,)` | posterior draws of α0 |
| `zeta` | `(T, K)` | derived “activated” global weights (thresholded) |
| `g0` | `(G,)` | posterior mean of the G0 density evaluated on `xgrid` |

Load example:

```python
from pathlib import Path
import numpy as np

rep_dir = Path("results_sample_2025-10-20/poisson/data/rep1")
g0 = np.load(rep_dir / "g0_components.npz")

xgrid = g0["xgrid"]
g0_density = g0["g0"]
mu = g0["mu"]           # (T,K)
sigma = g0["sigma"]     # (T,K)
pi_norm = g0["pi_norm"] # (T,S,K)
alpha0 = g0["alpha0"]   # (T,)

print(xgrid.shape, g0_density.shape, mu.shape)
```

How it is used:

- `utils.plot_g0_density_from_file(...)` produces rep-level G0 density plots.
- `reconstruct_source_mixture.py` (and `reconstruct_and_update.py`) use `mu`, `sigma`, and a global weight vector derived from `zeta` or `pi_norm` to reconstruct source mixtures.

---

## 5) Summary tables: `beta_summary_stats.csv`

Each rep writes:

```
<RESULTS_ROOT>/<FAMILY>/data/rep<r>/beta_summary_stats.csv
```

This is produced by `utils.summarise_beta_prior_posterior(...)` and contains **per-source** summaries comparing posterior and truth (truth is injected as `prior_*` columns when provided).

Load example:

```python
from pathlib import Path
import pandas as pd

rep_dir = Path("results_sample_2025-10-20/poisson/data/rep1")
stats = pd.read_csv(rep_dir / "beta_summary_stats.csv")
print(stats.columns)
print(stats.head())
```

Common columns (exact set depends on options):

- `experiment` (e.g. `rep1`)
- `source` (1-based)
- `prior_mean`, `prior_median`, `prior_mode` (truth, if provided)
- `post_mean`, `post_median`
- `mean_error`, `median_error`
- `post_ci_lower`, `post_ci_upper` and coverage booleans
- `kl_divergence`
- optional `glm_beta_est` (source-level estimate from statsmodels)

How it is used:

- `summarize_results.py` concatenates per-rep CSVs into `<RESULTS_ROOT>/<FAMILY>/beta_summary_stats.csv`.
- `plot_results.py` uses the family-level CSVs to generate cross-rep summary plots.

---

## Loading `.npy` archives (supported, not default)

The loaders (`utils.load_beta_posteriors` and `explore_profile_mle_kde.load_beta_archives`) support `.npy` files that store a Python dict (saved with `allow_pickle=True`).

Example:

```python
import numpy as np
bundle = np.load("beta_1_1.npy", allow_pickle=True).item()
samples = bundle["samples"]
grid = bundle["grid"]
loglik = bundle["loglik"]
```

---

## How these simulated artifacts flow through the pipeline

1. **Generate profile curves** (`simulation.py`) → `df_sim` (long format).
2. **Fit HDP model** (`models_vectorized.py` or `models.py`) using curve interpolation inside NumPyro.
3. **Sample posterior** (NumPyro NUTS in `main.py`) → posterior draws.
4. **Write rep artifacts**:
   - `beta_<s>_<o>.npz` (posterior + curve + truth),
   - `g0_components.npz` (global mixture draws + derived G0 density),
   - `beta_summary_stats.csv` (per-source evaluation summary),
   - `truth_source_mixtures.npz` (simulation truth).
5. **Post-process across reps**:
   - `summarize_results.py` aggregates CSVs and generates summary figures,
   - additional scripts generate CI, density-coverage, trace, and MLE diagnostic plots.

---

## When something looks “missing”

- If your `beta_<s>_<o>.npz` files do **not** contain `prior_samples`, then scripts that need truth must either:
  - read truth from `truth_source_mixtures.npz` (source-level truth), or
  - approximate truth from the profile-likelihood curve (e.g. argmax grid), which is what `regen_prev_summary.py` does.

- Some plotting scripts expect per-outcome coverage rows in a CSV. If your run only produced per-source CSVs, prefer scripts that compute diagnostics directly from NPZ archives (e.g. `plot_ci_truth.py`).
