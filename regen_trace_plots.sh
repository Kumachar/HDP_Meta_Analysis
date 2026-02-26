#!/usr/bin/env bash
#SBATCH --job-name=hdp_regen_trace
#SBATCH --account=
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/regen_trace_%j.out
#SBATCH --error=logs/regen_trace_%j.err

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

# ---- Args ----
RESULTS_ROOT="${1:-results_$(date +%F)}"      # required path to results folder
FAMILIES="${2:-"linear poisson logistic"}"    # space-separated; optional
OVERWRITE="${3:-}"                            # pass 'overwrite' to force regeneration

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] FAMILIES=${FAMILIES}"
echo "[INFO] OVERWRITE=${OVERWRITE:-no}"

CMD=(
  python -u regen_trace_plots.py
    --results-root "${RESULTS_ROOT}"
    --families ${FAMILIES}
)

if [ "${OVERWRITE:-}" = "overwrite" ]; then
  CMD+=( --overwrite )
fi

echo "[INFO] srun ${CMD[*]}"
srun "${CMD[@]}"

echo "[DONE] Regenerated beta trace plots."
