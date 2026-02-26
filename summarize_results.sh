#!/usr/bin/env bash
#SBATCH --job-name=hdp_summary
#SBATCH --account=
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

# ---- Activate conda ----
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 🔧 FIX: conda activate hooks may reference unset vars under `set -u`
set +u
conda activate "$HOME/.conda/envs/pymc_env" || { echo "[ERR] conda env missing"; exit 1; }
set -u

echo "[INFO] Python: $(which python)"


# ---- Light threading env (safe defaults) ----
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Args ----
RESULTS_ROOT="${1:-results_$(date +%F)}"      # required path to results folder
FAMILIES="${2:-linear poisson logistic}"       # space-separated; optional
STRICT="${3:-}"                                 # pass 'strict' to require recon files

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] FAMILIES=${FAMILIES}"
echo "[INFO] STRICT_RECON=${STRICT:-no}"

# ---- Build command ----
CMD=( python -u summarize_results.py
      --results-root "${RESULTS_ROOT}"
      --families ${FAMILIES}
      --prefer-recon )

if [ "${STRICT:-}" = "strict" ]; then
  CMD+=( --strict-recon )
fi

echo "[INFO] srun ${CMD[*]}"
srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores "${CMD[@]}"

echo "[DONE] Aggregated summaries and generated figures."
