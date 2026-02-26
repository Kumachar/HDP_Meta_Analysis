#!/usr/bin/env bash
#SBATCH --job-name=hdp_reconstruct
#SBATCH --account=fbu0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/recon_%j.out
#SBATCH --error=logs/recon_%j.err

set -euo pipefail
mkdir -p logs

# -----------------------------
# Conda / Python environment
# -----------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Activate your env (adjust if needed)
conda activate pymc_env || { echo "[ERR] conda env not found"; exit 1; }
echo "[INFO] Python: $(which python)"

# -----------------------------
# Arguments
# -----------------------------
# 1) RESULTS_ROOT: results directory that contains {linear,poisson,logistic}/data/rep*/
RESULTS_ROOT="${1:-results_$(date +%F)}"

# 2) Optional: families (space-separated). Default: linear poisson logistic
FAMILIES="${2:-"linear poisson logistic"}"

# 3) Optional: reps to process (space-separated). Default: discover all rep*/ under each family
REPS="${3:-}"

# 4) Optional: cap number of outcomes per source (integer) for speed; default: use all
OUTCOMES_CAP="${4:-}"

# 5) Optional: override alpha0 if not present in g0_components.npz; default: none
ALPHA0="${5:-}"

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] FAMILIES=${FAMILIES}"
echo "[INFO] REPS=${REPS:-<auto>}"
echo "[INFO] OUTCOMES_CAP=${OUTCOMES_CAP:-<all>}"
echo "[INFO] ALPHA0=${ALPHA0:-<from npz if present>}"

# -----------------------------
# Run the post-processing
# -----------------------------
CMD=( python -u reconstruct_and_update.py
      --results-root "${RESULTS_ROOT}"
      --families ${FAMILIES}
)

# add optional flags only if provided
if [ -n "${REPS}" ]; then
  # shellcheck disable=SC2086
  CMD+=( --reps ${REPS} )
fi

if [ -n "${OUTCOMES_CAP}" ]; then
  CMD+=( --outcomes-cap "${OUTCOMES_CAP}" )
fi

if [ -n "${ALPHA0}" ]; then
  CMD+=( --alpha0 "${ALPHA0}" )
fi

echo "[INFO] srun ${CMD[@]}"
srun "${CMD[@]}"

echo "[DONE] Updated rep summaries with reconstructed weights."
