#!/usr/bin/env bash
#SBATCH --job-name=outcov
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/outcov_%j.out
#SBATCH --error=logs/outcov_%j.err

set -euo pipefail
mkdir -p logs

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
RESULTS_ROOT="${1:?Usage: sbatch plot_outcome_coverage.sbatch RESULTS_ROOT [\"family1 family2 ...\"]}"
FAMILIES="${2:-}"

CMD=( python -u plot_outcome_coverage.py --results-root "${RESULTS_ROOT}" )

# Optional families (space-separated string)
if [ -n "${FAMILIES}" ]; then
  # shellcheck disable=SC2086
  CMD+=( --families ${FAMILIES} )
fi

echo "[INFO] srun ${CMD[*]}"
srun "${CMD[@]}"
