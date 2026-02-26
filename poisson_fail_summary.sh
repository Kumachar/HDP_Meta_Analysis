#!/usr/bin/env bash
#SBATCH --job-name=poiss_fail_summary
#SBATCH --account=
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/poiss_fail_summary_%j.out
#SBATCH --error=logs/poiss_fail_summary_%j.err

# Usage:
#   sbatch poisson_fail_summary.sh [RESULTS_ROOT] [OUT_CSV]
# Examples:
#   sbatch poisson_fail_summary.sh results_2025-10-10
#   sbatch poisson_fail_summary.sh results_2025-10-10 beta_so_fail_summary_custom.csv

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

# -----------------------------
# Conda init (same style as your run_all.sh)
# -----------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda not found. Install Miniconda in \$HOME or load a site miniconda module."
  exit 1
fi

conda activate pymc_env || { echo "[ERROR] failed to 'conda activate pymc_env'"; exit 1; }
echo "[INFO] Python: $(which python)"

# -----------------------------
# Light threading env
# -----------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------
# Positional args
# -----------------------------
RESULTS_ROOT="${1:-results_$(date +%F)}"
OUT_CSV="${2:-beta_so_fail_summary.csv}"

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] OUT_CSV=${OUT_CSV}"

# -----------------------------
# Run the summary builder
# -----------------------------
CMD=( python -u poisson_beta_so_fail_summary.py
      --results-root "${RESULTS_ROOT}"
      --out-csv "${OUT_CSV}"
)

echo "[INFO] srun ${CMD[*]}"
srun -c "${SLURM_CPUS_PER_TASK}" --cpu-bind=cores "${CMD[@]}"

echo "[DONE] Fail summary written to ${RESULTS_ROOT}/poisson/${OUT_CSV}"
echo "[DONE] Histograms (if any) in ${RESULTS_ROOT}/poisson/figures/diagnosis_plot/"
