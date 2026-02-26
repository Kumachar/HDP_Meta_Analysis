#!/usr/bin/env bash
#SBATCH --job-name=hdp_run_one
#SBATCH --account=fall2023-free-users
#SBATCH --partition=biostat-default
#SBATCH --constraint=AVX
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@90
#SBATCH --output=logs/hdp_%j.out
#SBATCH --error=logs/hdp_%j.err

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
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda not found."
  exit 1
fi
conda activate "$HOME/.conda/envs/pymc_env" || conda activate pymc_env || { echo "[ERROR] conda env missing"; exit 1; }
echo "[INFO] Python: $(which python)"

# -----------------------------
# CPU threading env (headless)
# -----------------------------
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

# -----------------------------
# Shared defaults (tune as needed)
# -----------------------------
K=5
O=20
S=8
N_SOURCES=8
N_OBS=200
SEED=42
OUTPUT_BASE="results_sample_2025-10-20"
NUM_WARMUP=15000
NUM_SAMPLES=20000

# =============================
# MODE A: worker (runs exactly one MODEL,REP)
# =============================
if [[ -n "${MODEL:-}" && -n "${REP:-}" ]]; then
  echo "[WORKER] MODEL=${MODEL} REP=${REP}"

  python -u main.py run-hdp \
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
    --num-chains 4 \
    --n-threads 2 \
    --parallel-chains \
    --num-warmup "$NUM_WARMUP" \
    --num-samples "$NUM_SAMPLES"

  echo "[DONE] MODEL=${MODEL} REP=${REP}"
  exit 0
fi

# =============================
# MODE B: launcher — submit one job per rep listed in reps.txt
# =============================
REPS_FILE="${1:-reps.txt}"
if [[ ! -f "$REPS_FILE" ]]; then
  echo "[ERROR] reps file not found: $REPS_FILE"
  echo "Example lines:"
  echo "  linear 9 19 20 49 50 94 99 100"
  echo "  logistic 40"
  echo "  poisson 11 20 65 70 73 74 79 80 99 100"
  exit 2
fi

submitted=0
while IFS= read -r line; do
  # skip blanks/comments
  [[ -z "${line// }" || "${line}" =~ ^# ]] && continue

  # first token = model, rest = reps (ignore any 'rep' literals)
  read -r model rest <<<"$line"
  [[ -z "${model}" ]] && continue

  reps=()
  for tok in $rest; do
    [[ "$tok" == "rep" ]] && continue
    [[ "$tok" =~ ^[0-9]+$ ]] && reps+=("$tok")
  done
  (( ${#reps[@]} == 0 )) && { echo "[WARN] no reps on line: $line"; continue; }

  for rep in "${reps[@]}"; do
    echo "[SUBMIT] model=${model} rep=${rep}"
    sbatch \
      --job-name="hdp_${model}_rep${rep}" \
      --time=3:00:00 \
      --cpus-per-task=8 \
      --mem=4G \
      --partition=biostat-default \
      --constraint=AVX \
      --requeue \
      --output="logs/${model}_rep${rep}_%j.out" \
      --error="logs/${model}_rep${rep}_%j.err" \
      --export=ALL,MODEL="${model}",REP="${rep}" \
      "$0"
    ((submitted+=1))
  done
done < "$REPS_FILE"

echo "[LAUNCHER] submitted ${submitted} jobs at 3:00:00 each"
