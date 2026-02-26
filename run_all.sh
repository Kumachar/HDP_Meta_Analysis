#!/usr/bin/env bash
#SBATCH --job-name=hdp_two_trees_logit
#SBATCH --account=fall2023-free-users
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=40:00:00
#SBATCH --output=logs/hdp_two_%A_%a.out
#SBATCH --error=logs/hdp_two_%A_%a.err
#SBATCH --array=0-19%12   # 1 model × (100/5) = 20 tasks, 12 concurrent

set -euo pipefail
mkdir -p logs

# -----------------------------
# Conda init
# -----------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda not found."
  exit 1
fi
conda activate "$HOME/.conda/envs/pymc_env" || conda activate pymc_env || { echo "[ERROR] conda env missing"; exit 1; }
echo "[INFO] Python: $(which python)"

# --- CPU threading env ---
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export MALLOC_ARENA_MAX=2

# --- knobs ---
K=5;  S=8;  N_SOURCES=8;  N_OBS=1000;  SEED=42
O20=20; O50=50
NUM_WARMUP=15000
NUM_SAMPLES=20000
OUTPUT_BASE_20="results_sample_2025-12-04_20outcomes"
OUTPUT_BASE_50="results_sample_2025-12-04_50outcomes"

MODELS=(logistic)

N_REPS=100
REPS_PER_TASK=5
BLOCKS_PER_MODEL=$(( N_REPS / REPS_PER_TASK ))  # 20

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$(( TASK_ID / BLOCKS_PER_MODEL ))   # 0..0
BLOCK_INDEX=$(( TASK_ID % BLOCKS_PER_MODEL ))   # 0..19

if (( MODEL_INDEX >= ${#MODELS[@]} )); then
  echo "Array index ${TASK_ID} out of range"; exit 1
fi

MODEL=${MODELS[$MODEL_INDEX]}
START_REP=$(( BLOCK_INDEX * REPS_PER_TASK + 1 ))
END_REP=$(( START_REP + REPS_PER_TASK - 1 ))

echo "[Task $TASK_ID] model=${MODEL} reps=${START_REP}-${END_REP}"

for REP in $(seq "$START_REP" "$END_REP"); do
  echo "  -> rep=${REP}: O=50 (simulate 50, analyze all 50) → ${OUTPUT_BASE_50}"
  srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    python -u main.py run-hdp \
      --K "$K" --O "$O50" --O-sim "$O50" \
      --S "$S" --n-obs "$N_OBS" --N-sources "$N_SOURCES" \
      --seed "$SEED" \
      --output-base "$OUTPUT_BASE_50" \
      --model-type "$MODEL" \
      --rep "$REP" --n-reps 1 \
      --use-vectorized-model \
      --num-chains 4 --n-threads 2 --parallel-chains \
      --num-warmup "$NUM_WARMUP" --num-samples "$NUM_SAMPLES"

  echo "  -> rep=${REP}: O=20 (simulate 50, analyze first 20) → ${OUTPUT_BASE_20}"
  srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    python -u main.py run-hdp \
      --K "$K" --O "$O20" --O-sim "$O50" --outcome-cap "$O20" \
      --S "$S" --n-obs "$N_OBS" --N-sources "$N_SOURCES" \
      --seed "$SEED" \
      --output-base "$OUTPUT_BASE_20" \
      --model-type "$MODEL" \
      --rep "$REP" --n-reps 1 \
      --use-vectorized-model \
      --num-chains 4 --n-threads 2 --parallel-chains \
      --num-warmup "$NUM_WARMUP" --num-samples "$NUM_SAMPLES"
done
