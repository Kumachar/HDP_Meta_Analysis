#!/usr/bin/env bash
#SBATCH --job-name=hdp_NCs_logit_real
#SBATCH --account=fall2023-free-users
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=40:00:00
#SBATCH --output=logs/hdp_NCs_%j.out
#SBATCH --error=logs/hdp_NCs_%j.err

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

# -----------------------------
# Conda init
# -----------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# FIX: conda activate hooks may reference unset vars under `set -u`
set +u
conda activate "$HOME/.conda/envs/pymc_env" || { echo "[ERR] conda env missing: $HOME/.conda/envs/pymc_env"; exit 1; }
set -u

echo "[INFO] Python: $(which python)"
python -V

# Robust: resolve the absolute Python binary from the activated conda env.
# (Some clusters run `srun` steps with a reduced PATH, so `python` may not resolve.)
PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[ERR] Python executable not found/executable: ${PYTHON_BIN}"
  exit 2
fi

# -----------------------------
# CPU threading env
# -----------------------------
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# Keep BLAS thread pools small; NumPyro parallel chains already consume CPUs.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
unset XLA_FLAGS || true

# -----------------------------
# Hard-coded project paths
# -----------------------------
CODE_ROOT="/home/wxhua/HDP_Experiment_Outcome_Based_each_betaso_latest"
DATA_FILE="${CODE_ROOT}/profileLikelihoods_NCs_long.csv"
CODE_PY="${CODE_ROOT}/main_realdata.py"

if [ ! -d "${CODE_ROOT}" ]; then
  echo "[ERR] CODE_ROOT not found: ${CODE_ROOT}"
  exit 2
fi
if [ ! -f "${DATA_FILE}" ]; then
  echo "[ERR] DATA_FILE not found: ${DATA_FILE}"
  exit 2
fi
if [ ! -f "${CODE_PY}" ]; then
  echo "[ERR] CODE_PY not found: ${CODE_PY}"
  exit 2
fi

cd "${CODE_ROOT}"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] DATA_FILE=${DATA_FILE}"
echo "[INFO] CODE_PY=${CODE_PY}"

# -----------------------------
# Knobs (edit ONE line to switch settings)
# -----------------------------
# Leave SOURCES_RANGE empty to use all sources.
# Set SOURCES_RANGE="4 12" to run the restricted analysis using only sources 4..12 (inclusive).
SOURCES_RANGE="${SOURCES_RANGE:-}"

# Tag used in output folder name so you can run multiple jobs simultaneously without collisions.
RUN_TAG="${RUN_TAG:-all_sources}"

# Zoomed G0 plot x-range (requested: -10 to 10)
G0_ZOOM_XLIM="${G0_ZOOM_XLIM:--10 10}"

# Main model knobs
K="${K:-5}"
SEED="${SEED:-42}"
FAMILY="${FAMILY:-logistic}"

NUM_WARMUP="${NUM_WARMUP:-15000}"
NUM_SAMPLES="${NUM_SAMPLES:-20000}"
NUM_CHAINS="${NUM_CHAINS:-4}"
THREADS_PER_CHAIN="${THREADS_PER_CHAIN:-2}"

TODAY="$(date +%F)"
OUTPUT_BASE="${OUTPUT_BASE:-results_realdata_${TODAY}_NCs_${RUN_TAG}}"

# Trace plots (lightweight subset)
TRACE_PLOTS="${TRACE_PLOTS:-1}"
TRACE_BETA_SOURCES="${TRACE_BETA_SOURCES:-3}"
TRACE_BETA_OUTCOMES="${TRACE_BETA_OUTCOMES:-3}"
TRACE_MAX_DRAWS="${TRACE_MAX_DRAWS:-4000}"

# G0 computation knobs (subsampled)
G0_DRAW_CAP="${G0_DRAW_CAP:-2000}"
G0_BATCH_SIZE="${G0_BATCH_SIZE:-250}"
G0_XGRID_LEN="${G0_XGRID_LEN:-2000}"
G0_WEIGHT_THRESHOLD="${G0_WEIGHT_THRESHOLD:-0.10}"

echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] SOURCES_RANGE=${SOURCES_RANGE:-<all>}"
echo "[INFO] G0_ZOOM_XLIM=${G0_ZOOM_XLIM}"
echo "[INFO] K=${K} SEED=${SEED} FAMILY=${FAMILY}"
echo "[INFO] WARMUP=${NUM_WARMUP} SAMPLES=${NUM_SAMPLES} CHAINS=${NUM_CHAINS} THREADS_PER_CHAIN=${THREADS_PER_CHAIN}"
echo "[INFO] TRACE_PLOTS=${TRACE_PLOTS} TRACE_BETA_SOURCES=${TRACE_BETA_SOURCES} TRACE_BETA_OUTCOMES=${TRACE_BETA_OUTCOMES} TRACE_MAX_DRAWS=${TRACE_MAX_DRAWS}"

CMD=( "${PYTHON_BIN}" -u "${CODE_PY}" run-hdp
      --data-file "${DATA_FILE}"
      --family "${FAMILY}"
      --K "${K}"
      --seed "${SEED}"
      --output-base "${OUTPUT_BASE}"
      --use-vectorized-model
      --num-chains "${NUM_CHAINS}"
      --n-threads "${THREADS_PER_CHAIN}"
      --parallel-chains
      --num-warmup "${NUM_WARMUP}"
      --num-samples "${NUM_SAMPLES}"
      --g0-draw-cap "${G0_DRAW_CAP}"
      --g0-batch-size "${G0_BATCH_SIZE}"
      --g0-xgrid-len "${G0_XGRID_LEN}"
      --g0-weight-threshold "${G0_WEIGHT_THRESHOLD}"
      --g0-zoom-xlim ${G0_ZOOM_XLIM}
)

if [ -n "${SOURCES_RANGE}" ]; then
  CMD+=( --sources-range ${SOURCES_RANGE} )
fi

if [ "${TRACE_PLOTS}" = "1" ]; then
  CMD+=( --trace-plots
         --trace-beta-sources "${TRACE_BETA_SOURCES}"
         --trace-beta-outcomes "${TRACE_BETA_OUTCOMES}"
         --trace-max-draws "${TRACE_MAX_DRAWS}" )
fi

echo "[INFO] srun ${CMD[*]}"
# --export=ALL ensures the conda environment variables propagate into the step.
srun --export=ALL -c "${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores "${CMD[@]}"

echo "[DONE] Real-data HDP run complete."
