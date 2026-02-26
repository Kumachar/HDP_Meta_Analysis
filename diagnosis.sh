#!/usr/bin/env bash
#SBATCH --job-name=hdp_diag_poisson
#SBATCH --account=fbu0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

# -----------------------------
# Conda init (same pattern as run_all.sh)
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
# Optional threading env (lightweight job)
# -----------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------
# Args (positional with defaults)
# -----------------------------
# Usage:
#   sbatch diagnosis.sh [RESULTS_ROOT] [THRESHOLD] [POINT] [DPI] [MAX_PER_SOURCE]
# Examples:
#   sbatch diagnosis.sh results_2025-10-10
#   sbatch diagnosis.sh results_2025-10-10 0.75 mean 200 6

RESULTS_ROOT="${1:-results_$(date +%F)}"
THRESHOLD="${2:-0.80}"
POINT="${3:-median}"        # median|mean
DPI="${4:-180}"
MAX_PER_SOURCE="${5:-}"     # optional; if empty, uses all

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] THRESHOLD=${THRESHOLD}"
echo "[INFO] POINT=${POINT}"
echo "[INFO] DPI=${DPI}"
echo "[INFO] MAX_PER_SOURCE=${MAX_PER_SOURCE:-<all>}"

# -----------------------------
# Build command
# -----------------------------
CMD=( python -u diagnosis_plots.py
      --results-root "${RESULTS_ROOT}"
      --threshold "${THRESHOLD}"
      --point "${POINT}"
      --dpi "${DPI}"
)

if [ -n "${MAX_PER_SOURCE}" ]; then
  CMD+=( --max-per-source "${MAX_PER_SOURCE}" )
fi

echo "[INFO] srun ${CMD[*]}"
srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores "${CMD[@]}"

echo "[DONE] Diagnosis plots generated (see poisson/figures/diagnosis_plot/)."
