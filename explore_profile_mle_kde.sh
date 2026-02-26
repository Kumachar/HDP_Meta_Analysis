#!/usr/bin/env bash
#SBATCH --job-name=hdp_explore_profile
#SBATCH --account=
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:20:00
#SBATCH --output=logs/explore_profile_%j.out
#SBATCH --error=logs/explore_profile_%j.err

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

# conda activate hooks may reference unset vars under `set -u`
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
RESULTS_ROOT="${1:-results_$(date +%F)}"              # required path to results folder
FAMILIES="${2:-linear poisson logistic}"               # space-separated; optional
REPS="${3:-}"                                         # optional, space-separated (e.g. "1 2 3")
OUTCOME_CAP="${4:-}"                                  # optional integer
OUT_DIR="${5:-}"                                      # optional output dir
OVERLAY_TRUTH="${6:-}"                                # pass 'truth' to overlay beta_true markers

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] FAMILIES=${FAMILIES}"
echo "[INFO] REPS=${REPS:-<auto>}"
echo "[INFO] OUTCOME_CAP=${OUTCOME_CAP:-<all>}"
echo "[INFO] OUT_DIR=${OUT_DIR:-<default-per-family>}"
echo "[INFO] OVERLAY_TRUTH=${OVERLAY_TRUTH:-no}"

# ---- Build command ----
CMD=( python -u explore_profile_mle_kde.py
      --results-root "${RESULTS_ROOT}"
      --families ${FAMILIES}
      --dpi 220
)

if [ -n "${REPS}" ]; then
  # shellcheck disable=SC2086
  CMD+=( --reps ${REPS} )
fi

if [ -n "${OUTCOME_CAP}" ]; then
  CMD+=( --outcome-cap "${OUTCOME_CAP}" )
fi

if [ -n "${OUT_DIR}" ]; then
  CMD+=( --out-dir "${OUT_DIR}" )
fi

if [ "${OVERLAY_TRUTH:-}" = "truth" ]; then
  CMD+=( --overlay-truth )
fi

echo "[INFO] srun ${CMD[*]}"
srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores "${CMD[@]}"

echo "[DONE] Exploratory profile-likelihood plots generated."
