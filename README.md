# HDP Meta-analysis with Profile-Likelihood Curves

This repository implements a **hierarchical meta-analysis** model for outcome-specific effect estimates using:

- **Profile likelihood curves** as the per-outcome “data likelihood” for an effect parameter,
- **Hierarchical Dirichlet-process (HDP)** structure to share information across **sources** while allowing source-specific heterogeneity across **outcomes**, and
- **Hamiltonian Monte Carlo (HMC)** (via **NumPyro NUTS**) for posterior inference.

The codebase supports two workflows:

1. **Simulated experiments**: simulate outcome-level profile likelihood curves from linear / logistic / Poisson data, fit the HDP model, and evaluate performance against known truth.
2. **Real-data experiments**: run the same HDP model on *provided* profile-likelihood curves (long-format CSV) and produce diagnostic + exploratory plots (no truth required).

---

## Contents

- [Methodological overview](#methodological-overview)
- [Results layout](#results-layout)
- [Codebase structure](#codebase-structure)
- [Usage](#usage)
  - [Environment setup](#environment-setup)
  - [Run simulated experiments](#run-simulated-experiments)
  - [Run real-data experiments](#run-real-data-experiments)
  - [Post-processing and diagnostics](#post-processing-and-diagnostics)
  - [Reconstruction utilities](#reconstruction-utilities)
- [Notes and known sharp edges](#notes-and-known-sharp-edges)

---

## Methodological overview

### Problem setting

You have:

- **Sources** \(s = 1,\dots,S\) (e.g., studies, sites, cohorts),
- **Outcomes** \(o = 1,\dots,O_s\) per source,
- For each \((s,o)\), a **profile log-likelihood curve** \(\ell_{s,o}(\beta)\) evaluated on a grid of \(\beta\) values.

The goal is to infer the outcome-level effects \(\beta_{s,o}\) and summarize source-level patterns, while borrowing strength **across outcomes** and **across sources**.

### Profile likelihood as the likelihood term

Instead of modeling raw data at inference time, the HDP model consumes the **curve** \(\ell_{s,o}(\beta)\):

- In simulation (`simulation.py`), curves are computed directly (and also via fitting one GLM per source for point estimates).
- In real data (`main_realdata.py`), curves are read from long-format CSV with columns:
  `source, outcome, point, value`, where `value` is the profile log-likelihood at `point`.

In the Bayesian model (`models.py` / `models_vectorized.py`), each latent \(\beta_{s,o}\) contributes:

\[
\log p(\text{curve}_{s,o} \mid \beta_{s,o}) \;\propto\; \ell_{s,o}(\beta_{s,o})
\]

Operationally, the code **linearly interpolates** \(\ell_{s,o}(\beta)\) on the provided grid and adds it to the joint log density via `numpyro.factor(...)`.

> Assumption: profile log-likelihoods are only defined up to an additive constant; this is fine because additive constants cancel in Bayesian inference as long as they do not depend on \(\beta\).

### Hierarchical Dirichlet-process (finite truncation)

The repository uses an HDP-like hierarchy with a **finite truncation** of size `K`:

- Global stick-breaking weights \(\beta = (\beta_1,\dots,\beta_K)\) represent the base measure mixing weights.
- For each source \(s\), source-specific weights \(\pi_s\) are drawn around \(\beta\):

\[
\pi_s \sim \text{Dirichlet}(\alpha_0 \, \beta)
\]

- Each outcome-level effect is drawn from a **source-specific mixture of Normals**:

\[
z_{s,o} \sim \text{Categorical}(\pi_s), \quad
\beta_{s,o} \sim \mathcal{N}(\mu_{z_{s,o}}, \sigma_{z_{s,o}}^2)
\]

This structure pools information:

- **Across outcomes within a source** via \(\pi_s\),
- **Across sources** via the shared global \(\beta\) and shared component parameters \((\mu_k, \sigma_k)\).

Key hyperparameters include:
- `alpha0` (controls how concentrated each \(\pi_s\) is around the global weights),
- `gamma` (stick-breaking concentration for global \(\beta\)),
- priors for component means `mu[k]` and scales `sigma[k]`.

### HMC / NUTS posterior inference

Posterior inference is performed using **NumPyro NUTS**:

- Simulation runner: `main.py run-hdp`
- Real-data runner: `main_realdata.py run-hdp`

Both support multi-chain sampling, and optional CPU-parallel chains through JAX virtual devices.

---

## Results layout

### Simulated experiments (`main.py`)

For an `--output-base <RESULTS_ROOT>` and a given `--model-type <FAMILY>` (one of `linear`, `poisson`, `logistic`), the runner writes:

```
<RESULTS_ROOT>/<FAMILY>/
  data/
    rep1/
      truth_source_mixtures.npz
      beta_summary_stats.csv
      g0_components.npz
      beta_<s>_<o>.npz
      ...
    rep2/
    ...
  figures/
    beta_plots/
      rep<r>_beta_<s>_<o>.jpg
      rep<r>_g0.png
  beta_summary_stats.csv          (family-level concatenation of per-rep CSVs)
```

Additional post-processing scripts create extra figure folders (see `FIGURE_INDEX.md`).

### Real-data experiments (`main_realdata.py`)

For `--output-base <OUTPUT_BASE>` and `--family <FAMILY>` (default `realdata`, but you can set e.g. `logistic`), the runner writes:

```
<OUTPUT_BASE>/<FAMILY>/
  data/rep<r>/
    profile_likelihood_used.csv
    mapping_sources.csv
    mapping_outcomes.csv
    profile_mle_by_outcome.csv
    profile_mle_by_source.csv
    beta_<s>_<o>.npz
    beta_<s>.npz
    g0_components_subsampled.npz
  figures/
    trace_plots/
    beta_plots/
    profile_mle/
    source_density/
```

---

## Codebase structure

Below is a file-by-file map of **every `.py` file** in the provided project snapshot.

> “Experiment type” is labeled as **Simulated**, **Real-data**, or **Shared** (used by both).

| File | What it does | Experiment type | Where it fits in the pipeline |
|---|---|---:|---|
| `main.py` | CLI entrypoint for **simulated** HDP runs. Simulates profile-likelihood curves, runs NUTS, saves posteriors (`beta_<s>_<o>.npz`), computes `g0_components.npz`, writes per-rep + family summaries, and triggers some per-rep plots. | Simulated | **Primary simulation runner** (per family × rep). Produces the on-disk artifacts used by all downstream plotting/summary scripts. |
| `simulation.py` | Simulation of per-(source,outcome) **profile log-likelihood curves** for `linear`, `logistic`, and `poisson` families. Also returns truth (mixture weights and β values) and source-level GLM estimates. | Simulated | Called by `main.py` to create synthetic profile-likelihood inputs and truth used for evaluation. |
| `main_realdata.py` | Optimized real-data runner: loads long-format profile likelihood curves from CSV(s), runs NUTS, and writes a suite of exploratory/diagnostic figures **without truth**. Avoids ArviZ full expansion to reduce memory. | Real-data | **Primary real-data runner**. Produces `beta_<s>_<o>.npz` and real-data figure folders under `<output_base>/<family>/`. |
| `models.py` | Non-vectorized NumPyro HDP model. Uses per-(s,o) interpolation of profile likelihood curves and explicit loops. | Shared | The core probabilistic model (fallback when not using vectorized model). |
| `models_vectorized.py` | Vectorized NumPyro HDP model. Expects packed arrays (`x_padded`, `ll_padded`, `lengths`) and evaluates the profile-likelihood factors via JAX vectorization. Can optionally export per-(s,o) deterministic sites. | Shared | **Recommended** model for speed. Used by `main.py` and `main_realdata.py` when `--use-vectorized-model` is enabled. |
| `utils.py` | Shared utilities: saving/loading beta posterior “archives” (`.npz`/`.npy`), KDE/hist plotting, G0 component extraction + plotting, true-mixture calculations, KL computations, and a few helper summaries. | Shared | Plumbing for **I/O**, **derived quantities**, and **per-rep plots** used throughout. |
| `plot_results.py` | Higher-level plotting utilities for simulation results: summary plots across reps, G0 trajectory plots, “true vs posterior” source density comparisons, and combined G0/β visualizations. | Simulated (post-processing) | Called by `summarize_results.py` (and imported by `main.py`) to generate family- and root-level summary figures. |
| `summarize_results.py` | Aggregates per-rep summary CSVs into family-level `beta_summary_stats.csv` (optionally preferring reconstructed CSVs), and generates summary figures via `plot_results.generate_summary_plots`. | Simulated (post-processing) | After running many reps, run this to build **family-level CSVs** and **root summary figures**. |
| `plot_outcome_coverage.py` | Builds outcome-level coverage plots across outcomes using aggregated CSVs. | Simulated (post-processing) | Requires per-outcome coverage rows in the input CSVs (see “Notes and sharp edges”). Writes `summary_figs_outcome_cov/`. |
| `plot_ci_truth.py` | Summarizes posterior CIs vs truth for β_{s,o}: CI lengths, truth distributions, and optionally forest-style plots per source. Primarily reads `beta_<s>_<o>.npz`. | Simulated (post-processing) | Used via direct CLI or via `plot_ci_truth_per_source.sbatch` to generate `summary_figs_ci/` outputs. |
| `plot_source_density_coverage.py` | For each rep and source: compares posterior **source density** (from β samples) to the **true** source mixture density; writes “coverage band” figures and a root-level coverage barplot. | Simulated (post-processing) | Uses `truth_source_mixtures.npz`, `g0_components.npz`, and `beta_<s>_<o>.npz`. Writes `figures/source_density_coverage/` and `summary_figs/` artifacts. |
| `explore_profile_mle_kde.py` | Extracts profile-likelihood MLEs (argmax grid of `loglik`) from `beta_<s>_<o>.npz` and plots per-source KDE/hist grids. Optionally overlays “truth” from `prior_samples` if present. | Shared (exploration) | Can be used on either simulation or real-data outputs because both store `grid` and `loglik` inside `beta_<s>_<o>.npz`. Writes `figures/profile_explore/`. |
| `merge_beta_summaries.py` | Merges per-rep `beta_summary_stats.csv` into a family-level CSV and optionally an all-families CSV. | Shared (post-processing) | Standalone utility for aggregation; overlaps with what `summarize_results.py` does. |
| `reconstruct_source_mixture.py` | Given global mixture params (`g0_components.npz`) and per-outcome β summaries (`beta_<s>_<o>.npz`), reconstructs an estimated source mixture weight vector \(\hat\pi_s\). | Shared (post-processing) | Reconstruction tool: can run on a single source/rep to get \(\hat\pi_s\) and a reconstructed mixture plot. |
| `reconstruct_and_update.py` | Batch reconstruction across reps and sources. Produces `beta_summary_stats_recon.csv` per rep by merging reconstruction fields into the existing per-rep summary. | Shared (post-processing) | Used when you want per-source reconstructed weights embedded in summary tables. |
| `regen_trace_plots.py` | Regenerates trace plots for β from saved `beta_<s>_<o>.npz` samples, writing to `figures/trace/rep*/`. | Shared (post-processing) | Useful when trace plots were not saved during sampling, but NPZ posterior samples exist. |
| `regen_prev_summary.py` | Rebuilds an older “previous-style” summary table by combining per-source summary with additional reconstructed CI/coverage fields derived from saved NPZs. | Simulated (post-processing) | Intended to reconstruct richer summaries from on-disk artifacts without rerunning MCMC. |
| `poisson_beta_so_fail_summary.py` | For Poisson family: extracts (rep,source,outcome) cases where 95% CI fails to cover truth and makes diagnostic hist/box plots. | Simulated (Poisson diagnostics) | Requires per-outcome coverage info in the aggregated Poisson CSV and access to the corresponding `beta_<s>_<o>.npz` files. |
| `poisson_beta_so_ci_summary.py` | For Poisson family: builds combined covered/uncovered CI-length summaries and diagnostic plots (covered vs uncovered). | Simulated (Poisson diagnostics) | Same prerequisites as above; writes to `poisson/figures/diagnosis_plot/`. |

---

## Usage

### Environment setup

No `environment.yml` is included in the provided snapshot. The minimal dependencies below are inferred from imports across the `.py` files:

- Python 3.x
- `numpy`, `pandas`, `scipy`
- `matplotlib` (and optionally `seaborn` for one diagnostic script)
- `statsmodels` (simulation only)
- `jax` + `numpyro` (HMC/NUTS inference)
- `arviz`, `xarray` (simulation runner uses ArviZ to build `InferenceData`)

A typical conda setup (edit versions to match your system) looks like:

```bash
conda create -n pymc_env python=3.11 -y
conda activate pymc_env

# core scientific stack
conda install -c conda-forge numpy pandas scipy matplotlib arviz xarray statsmodels -y

# HMC backend
pip install "jax[cpu]" numpyro

# optional (used in some scripts)
conda install -c conda-forge seaborn -y
```

Cluster scripts in this repo assume the environment name is **`pymc_env`** and often activate it as:
`$HOME/.conda/envs/pymc_env`.

### Run simulated experiments

#### Local run (single rep)

From the project root:

```bash
python -u main.py run-hdp \
  --K 5 --S 8 --N-sources 8 --O 20 --n-obs 1000 \
  --seed 42 --rep 1 --n-reps 1 \
  --model-type logistic \
  --output-base results_sample_local \
  --use-vectorized-model \
  --num-chains 4 --n-threads 2 --parallel-chains \
  --num-warmup 15000 --num-samples 20000
```

Key options (simulation):

- `--model-type {linear,poisson,logistic}` controls both the simulator and the output family folder.
- `--O-sim` + `--outcome-cap` let you **simulate more outcomes than you analyze** (see `run_all.sh`).
- `--use-vectorized-model` uses `models_vectorized.py` (recommended).
- `--parallel-chains` + `--n-threads` configure CPU parallelism (see cluster scripts for safe defaults).

#### Biostat cluster SLURM scripts (simulation)

These are pre-written job scripts that call `main.py`:

- `run_reps.sh`  
  SLURM array over **(family × rep blocks)**. Runs `main.py run-hdp` for `linear`, `poisson`, `logistic` in blocks of 10 reps per family.  
  Output base is set inside the script (`OUTPUT_BASE="results_sample_2025-10-20"`).

- `run_all.sh`  
  Runs **logistic** with two configurations per rep (50-outcome full analysis, and 20-outcome truncation).  
  Writes to:
  - `results_sample_2025-12-04_50outcomes`
  - `results_sample_2025-12-04_20outcomes`

- `test.sh`  
  Another SLURM array worker/launcher pattern for batches of reps.

Typical usage:

```bash
sbatch run_reps.sh
# or
sbatch run_all.sh
```

> You will almost certainly want to edit the `#SBATCH` account/partition lines and the `OUTPUT_BASE` path inside these scripts.

#### GreatLakes SLURM scripts (simulation / maintenance)

- `retry_timeouts.sh` reruns a hard-coded list of timed-out SLURM array indices with a longer wall time and 8 chains.

- `reconstruct_update_gl.sh` runs the reconstruction/update step (`reconstruct_and_update.py`) on GreatLakes.
- `poisson_fail_summary.sh` runs Poisson CI-failure diagnostics (`poisson_beta_so_fail_summary.py`).
- `ci_summary.sh` runs Poisson covered-vs-uncovered CI-length diagnostics (`poisson_beta_so_ci_summary.py`).

### Run real-data experiments

Real-data runs use `main_realdata.py run-hdp`. The expected input is **long-format CSV** with columns:

- `source` (any label; internally remapped to 1..S),
- `outcome` (any label; internally remapped per source),
- `point` (β grid value),
- `value` (profile log-likelihood at that β grid point).

You can pass either:

- `--data-file path/to/profileLikelihoods_long.csv` (treated as rep1), or
- `--data-root path/to/folder` containing `rep1/`, `rep2/`, … (each rep folder contains one or more CSVs).

Example local command:

```bash
python -u main_realdata.py run-hdp \
  --data-file profileLikelihoods_long.csv \
  --family logistic \
  --K 5 \
  --seed 42 \
  --output-base results_realdata_run \
  --use-vectorized-model \
  --num-chains 4 --n-threads 2 --parallel-chains \
  --num-warmup 15000 --num-samples 20000 \
  --trace-plots \
  --g0-zoom-xlim -10 10
```

Biostat cluster script:

- `hdp_realdata.sh` is a complete SLURM job wrapper around the above command.  
  It currently hard-codes a `CODE_ROOT` and `DATA_FILE` path; edit those before use.

### Post-processing and diagnostics

After running many reps, you typically want to aggregate and plot.

#### Aggregate per-rep CSVs and generate summary figures

```bash
python -u summarize_results.py --results-root <RESULTS_ROOT> --families linear poisson logistic --prefer-recon
```

Or via SLURM:

```bash
sbatch summarize_results.sh <RESULTS_ROOT> "linear poisson logistic"
```

Outputs:

- family-level `beta_summary_stats.csv` under each family folder,
- root-level summary figures under `<RESULTS_ROOT>/summary_figs/`.

#### Outcome-level coverage plots

```bash
python -u plot_outcome_coverage.py --results-root <RESULTS_ROOT> --families linear poisson logistic
# outputs to <RESULTS_ROOT>/summary_figs_outcome_cov/
```

SLURM wrapper:

```bash
sbatch plot_outcome_coverage.sbatch <RESULTS_ROOT> "linear poisson logistic"
```

#### CI vs truth plots

```bash
python -u plot_ci_truth.py --results-root <RESULTS_ROOT> --families poisson --rep 1
# outputs to <RESULTS_ROOT>/summary_figs_ci/
```

SLURM wrapper for “per-rep forest plots per source”:

```bash
sbatch plot_ci_truth_per_source.sbatch <RESULTS_ROOT> "linear poisson logistic" beta_so_forest_per_source
```

#### Source density coverage plots

```bash
python -u plot_source_density_coverage.py --results-root <RESULTS_ROOT> --families linear poisson logistic
# outputs to <RESULTS_ROOT>/<family>/figures/source_density_coverage/ and <RESULTS_ROOT>/summary_figs/
```

SLURM wrapper:

```bash
sbatch plot_source_density_coverage.sh <RESULTS_ROOT> "linear poisson logistic"
```

#### Explore profile-likelihood MLEs (argmax of curve)

```bash
python -u explore_profile_mle_kde.py --results-root <RESULTS_ROOT> --families linear poisson logistic
# outputs to <RESULTS_ROOT>/<family>/figures/profile_explore/
```

SLURM wrapper:

```bash
sbatch explore_profile_mle_kde.sh <RESULTS_ROOT> "linear poisson logistic"
```

#### Regenerate trace plots from saved NPZ archives

If you did not save trace plots during sampling (or want them in a consistent format), you can regenerate them from the saved posterior sample archives `beta_<s>_<o>.npz`:

```bash
python -u regen_trace_plots.py --results-root <RESULTS_ROOT> --families linear poisson logistic --reps 1 2 3
# outputs to <RESULTS_ROOT>/<family>/figures/trace/rep<r>/
```

Biostat cluster wrapper:

```bash
sbatch regen_trace_plots.sh <RESULTS_ROOT> "linear poisson logistic"
```

#### Regenerate “previous-style” per-rep summary tables

`regen_prev_summary.py` reconstructs a richer per-rep summary file from on-disk artifacts.
It writes:

- `beta_summary_stats_prev.csv` under each `data/rep<r>/`
- (optionally) a family-level `beta_summary_stats_prev.csv`

Run directly:

```bash
python -u regen_prev_summary.py --results-root <RESULTS_ROOT> --families linear poisson logistic --overwrite
```

Biostat cluster wrapper:

```bash
sbatch regen_prev_summary.sh <RESULTS_ROOT> "linear poisson logistic" 1
```

#### Poisson-only CI diagnostics (failures / covered vs uncovered)

These scripts are Poisson-specific and write to:
`<RESULTS_ROOT>/poisson/figures/diagnosis_plot/`.

Run directly:

```bash
# failure-only diagnostics
python -u poisson_beta_so_fail_summary.py --results-root <RESULTS_ROOT> --fail-csv beta_so_fail_summary.csv

# combined covered vs uncovered CI-length diagnostics
python -u poisson_beta_so_ci_summary.py \
  --results-root <RESULTS_ROOT> \
  --combined-csv beta_so_ci_summary.csv \
  --fail-csv beta_so_fail_summary.csv \
  --cover-csv beta_so_covered_summary.csv
```

GreatLakes wrappers:

```bash
sbatch poisson_fail_summary.sh <RESULTS_ROOT> beta_so_fail_summary.csv
sbatch ci_summary.sh <RESULTS_ROOT> beta_so_ci_summary.csv beta_so_fail_summary.csv beta_so_covered_summary.csv
```

### Reconstruction utilities

#### Reconstruct per-source mixture weights from saved β archives

Single source/rep:

```bash
python -u reconstruct_source_mixture.py \
  --results-root <RESULTS_ROOT> \
  --family poisson \
  --rep 1 \
  --source 3
```

Batch (all sources × all reps), writing `beta_summary_stats_recon.csv`:

```bash
python -u reconstruct_and_update.py --results-root <RESULTS_ROOT> --families linear poisson logistic
```

GreatLakes SLURM wrapper:

```bash
sbatch reconstruct_update_gl.sh <RESULTS_ROOT> "linear poisson logistic"
```

After reconstruction, rerun:

```bash
python -u summarize_results.py --results-root <RESULTS_ROOT> --prefer-recon
```

so that family-level summaries are built from the reconstructed per-rep CSVs.

---

## Notes and known sharp edges

- **`models_vectorized_fast` is optional and not included in this snapshot.**  
  `main_realdata.py` tries to import it, but falls back to `models_vectorized.py` automatically.

- **Some downstream scripts require per-outcome coverage rows in summary CSVs.**  
  `main.py` currently writes *per-source* summaries (`beta_summary_stats.csv`) and stores per-outcome truth inside `beta_<s>_<o>.npz` as `prior_samples`.  
  Scripts like `plot_outcome_coverage.py` and the Poisson CI-failure diagnostics expect per-outcome coverage columns in a CSV. If those columns are absent, you may need to:
  - generate per-outcome diagnostics directly from NPZ (e.g., `plot_ci_truth.py`), or
  - use `regen_prev_summary.py` to rebuild richer “previous-style” tables from NPZ artifacts.

- **Cluster wrapper argument drift:**  
  `plot_source_density_coverage.sh` contains optional arguments (`--kde-bw`, `--style`) that are **not** accepted by `plot_source_density_coverage.py` in this snapshot. If you enable those options, SLURM will error with “unrecognized arguments”.

- **Threading matters on CPU clusters.**  
  The SLURM scripts intentionally cap BLAS thread pools (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, etc.) because NumPyro parallel chains already consume CPU cores.

- **Grid sizes can be large.**  
  `main.py` simulates profile likelihoods on a 20,000-point grid over \([-10,10]\), which can be heavy. The vectorized model is recommended for these settings.

---

If you want a quick map of result figure folders, see **`FIGURE_INDEX.md`**.  
For details on the saved simulated artifacts (`.npz`/`.csv`) and how to load them, see **`SIM_DATA_GUIDE.md`**.
