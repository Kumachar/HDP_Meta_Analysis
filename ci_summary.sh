#!/usr/bin/env bash
#SBATCH --job-name=hdp_ci_summary
#SBATCH --account=fbu0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/ci_summary_%j.out
#SBATCH --error=logs/ci_summary_%j.err

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"

# ----- Conda env (match your other scripts) -----
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda not found"; exit 1
fi
conda activate pymc_env || { echo "[ERROR] conda env missing"; exit 1; }
echo "[INFO] Python: $(which python)"

RESULTS_ROOT="${1:-results_$(date +%F)}"
COMBINED="${2:-beta_so_ci_summary.csv}"
FAIL="${3:-beta_so_fail_summary.csv}"
COVER="${4:-beta_so_covered_summary.csv}"

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] COMBINED=${COMBINED}"
echo "[INFO] FAIL=${FAIL}"
echo "[INFO] COVER=${COVER}"

srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python -u poisson_beta_so_ci_summary.py \
    --results-root "${RESULTS_ROOT}" \
    --combined-csv "${COMBINED}" \
    --fail-csv "${FAIL}" \
    --cover-csv "${COVER}"

echo "[DONE] CI length summary & plots generated."
