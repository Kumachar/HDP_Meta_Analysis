#!/usr/bin/env bash
#SBATCH --job-name=hdp_all_test
#SBATCH --account=fall2023-free-users
#SBATCH --partition=biostat-default        # 可改成 short
#SBATCH --constraint=AVX                   # 关键：只调度到支持 AVX 的节点
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8                  # 3 models × 3 reps = 9 tasks
#SBATCH --output=logs/hdp_all_%A_%a.out
#SBATCH --error=logs/hdp_all_%A_%a.err

set -euo pipefail
mkdir -p logs

echo "[INFO] $(date) host=$(hostname)"
echo "[INFO] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "[INFO] ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# -----------------------------
# Conda (adjust env name)
# -----------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate pymc_env || { echo "[ERROR] conda env missing"; exit 1; }
echo "[INFO] Python: $(which python)"

# -----------------------------
# CPU threading env
# -----------------------------
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset XLA_FLAGS

python - <<'PY'
import jax, os
print("[INFO] JAX backend:", jax.default_backend())
print("[INFO] Devices:", jax.devices())
print("[INFO] OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
PY

# -----------------------------
# Common knobs
# -----------------------------
K=6
O=20
S=3
N_SOURCES=3
N_OBS=80
SEED=20251027
TODAY=$(date +%F)
OUTDIR="results_${TODAY}_allmodels"

# -----------------------------
# Map array id -> (model, rep)
# 0-2: linear rep1-3
# 3-5: poisson rep1-3
# 6-8: logistic rep1-3
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
REP=$(( TASK_ID % 3 + 1 ))
MODEL_IDX=$(( TASK_ID / 3 ))
case "${MODEL_IDX}" in
  0) MODEL=linear ;;
  1) MODEL=poisson ;;
  2) MODEL=logistic ;;
  *) echo "[ERROR] bad array id ${TASK_ID}"; exit 2 ;;
esac

echo "[INFO] Running MODEL=${MODEL}, REP=${REP} → ${OUTDIR}/${MODEL}"

srun -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  python -u main.py run-hdp \
    --K "$K" --O "$O" --S "$S" --n-obs "$N_OBS" --N-sources "$N_SOURCES" \
    --seed "$SEED" \
    --output-base "$OUTDIR" \
    --model-type "$MODEL" \
    --n-reps 1 --rep "$REP" \
    --use-vectorized-model \
    --num-chains 1 \
    --n-threads "${SLURM_CPUS_PER_TASK}" \
    --num-warmup 10000 --num-samples 5000

echo "[DONE] ${MODEL}/rep${REP}"
