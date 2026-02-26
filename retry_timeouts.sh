#!/usr/bin/env bash
#SBATCH --job-name=hdp_array_retry
#SBATCH --account=
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=6:00:00                 # bump to avoid 4h timeouts
#SBATCH --output=logs/hdp8_retry_%A_%a.out
#SBATCH --error=logs/hdp8_retry_%A_%a.err
#SBATCH --array=3,4,12,13,23,32,35,41,47,50,57%12   # only TIMEOUT tasks

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

# ---- Conda init (same as your run_all.sh) ----
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

# ---- CPU threading env (same as your run_all.sh) ----
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset XLA_FLAGS

python - <<'PY'
import os, jax
print("[INFO] JAX backend:", jax.default_backend())
print("[INFO] Devices:", jax.devices())
print("[INFO] OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

# ---- Experiment knobs (same as your run_all.sh) ----
K=5; O=20; S=8; N_SOURCES=8; N_OBS=200; SEED=42
OUTPUT_BASE="results_sample_2025-09-21"

MODELS=(linear poisson logistic)
N_REPS=100
REPS_PER_TASK=5
BLOCKS_PER_MODEL=$(( N_REPS / REPS_PER_TASK ))    # 20

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$(( TASK_ID / BLOCKS_PER_MODEL ))     # 0..2
BLOCK_INDEX=$(( TASK_ID % BLOCKS_PER_MODEL ))     # 0..19

if (( MODEL_INDEX >= ${#MODELS[@]} )); then
  echo "Array index ${TASK_ID} out of range"; exit 1
fi
MODEL=${MODELS[$MODEL_INDEX]}

START_REP=$(( BLOCK_INDEX * REPS_PER_TASK + 1 ))  # 1,6,11,...,96
END_REP=$(( START_REP + REPS_PER_TASK - 1 ))      # 5,10,15,...,100

echo "[Retry Task $TASK_ID] model=${MODEL} reps=${START_REP}-${END_REP}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK} OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# --- Run reps (vectorized model + 8 parallel chains, 1 thread per chain) ---
for REP in $(seq "$START_REP" "$END_REP"); do
  echo "  -> Running rep=${REP}"
  srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    python main.py run-hdp \
      --K "$K" \
      --O "$O" \
      --S "$S" \
      --n-obs "$N_OBS" \
      --N-sources "$N_SOURCES" \
      --seed "$SEED" \
      --output-base "$OUTPUT_BASE" \
      --model-type "$MODEL" \
      --rep "$REP" \
      --use-vectorized-model \
      --num-chains 8 \
      --n-threads 1 \
      --parallel-chains
done
