#!/usr/bin/env bash
#SBATCH --job-name=src_kde_covplots
#SBATCH --account=
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=3:30:00
#SBATCH --output=logs/src_cov_%j.out
#SBATCH --error=logs/src_cov_%j.err

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


# ---- Light threading env ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu

# ---- Args ----
RESULTS_ROOT="${1:-results_$(date +%F)}"
FAMILIES="${2:-linear poisson logistic}"
REPS="${3:-}"                   # e.g. "1 2 3"; default: auto
MAX_DRAWS="${4:-400}"
GRID_LEN="${5:-1600}"
DPI="${6:-220}"
XLIM_LOW="${7:--5}"
XLIM_HIGH="${8:-5}"
KDE_BW="${9:-}"                 # 'scott' | 'silverman' | float
STYLE="${10:-arviz-whitegrid}"

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] FAMILIES=${FAMILIES}"
echo "[INFO] REPS=${REPS:-<auto>}"
echo "[INFO] XLIM=[${XLIM_LOW}, ${XLIM_HIGH}] STYLE=${STYLE}"

CMD=( python -u plot_source_density_coverage.py
      --results-root "${RESULTS_ROOT}"
      --families ${FAMILIES}
      --max-draws "${MAX_DRAWS}"
      --grid-len "${GRID_LEN}"
      --dpi "${DPI}"
      --xlim "${XLIM_LOW}" "${XLIM_HIGH}" )

if [ -n "${REPS}" ]; then
  CMD+=( --reps ${REPS} )
fi
if [ -n "${KDE_BW}" ]; then
  CMD+=( --kde-bw "${KDE_BW}" )
fi

echo "[INFO] srun ${CMD[*]}"
srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores "${CMD[@]}"

echo "[DONE] Wrote posterior-vs-true (KDE) density coverage plots per source/rep."
