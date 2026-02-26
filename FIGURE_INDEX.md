# FIGURE_INDEX

This index is a quick reference for **where plots are written**.

Paths are written relative to your chosen results directory:

- `<RESULTS_ROOT>` is the folder passed as `--output-base` (simulation) or `--output-base` (real data).
- `<FAMILY>` is a model family folder such as `linear`, `poisson`, `logistic` (simulation) or whatever you pass as `--family` (real data).

**One line per figure folder type:**

<RESULTS_ROOT>/<FAMILY>/figures/beta_plots — posterior β density plots (per (source,outcome)) and G0 plots written during runs  
<RESULTS_ROOT>/<FAMILY>/figures/trace_plots — (real data) trace plots for hyperparameters and a subset of β_{s,o}  
<RESULTS_ROOT>/<FAMILY>/figures/trace/rep<r> — (post-run) regenerated β trace plots from saved `beta_<s>_<o>.npz` samples  
<RESULTS_ROOT>/<FAMILY>/figures/profile_mle — (real data) KDE/hist panels of profile-likelihood MLEs across outcomes  
<RESULTS_ROOT>/<FAMILY>/figures/profile_explore — (post-run) KDE/hist panels of profile-likelihood MLEs (argmax of curve) across outcomes  
<RESULTS_ROOT>/<FAMILY>/figures/source_density — (real data) posterior source-level density plots (and a grid overview plot)  
<RESULTS_ROOT>/<FAMILY>/figures/source_density_coverage — (simulation) posterior-vs-true source density “coverage band” plots per rep/source  
<RESULTS_ROOT>/<FAMILY>/figures/g0_trajectory — G0 density trajectories over posterior draws (and posterior mean)  
<RESULTS_ROOT>/<FAMILY>/figures/g0_and_beta — overlay plot combining G0 density and per-source/posterior β densities on one canvas  
<RESULTS_ROOT>/<FAMILY>/figures/true_vs_post — (simulation) per-source comparison of true density vs posterior density  
<RESULTS_ROOT>/summary_figs — cross-family summary plots generated from aggregated CSVs (and some global diagnostics)  
<RESULTS_ROOT>/summary_figs_outcome_cov — outcome-level coverage plots across outcomes (requires per-outcome coverage rows in CSVs)  
<RESULTS_ROOT>/summary_figs_ci — CI-vs-truth diagnostics (CI length boxplots, truth boxplots, and related plots)  
<RESULTS_ROOT>/summary_figs_ci/<FOREST_SUBDIR> — per-source forest-style CI plots produced by `plot_ci_truth_per_source.sbatch`  
<RESULTS_ROOT>/poisson/figures/diagnosis_plot — Poisson-only CI diagnostics (failures, CI length overlays/boxplots)
