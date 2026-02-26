#!/usr/bin/env bash
#SBATCH --job-name=plot_ci_src
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:20:00
#SBATCH --output=logs/plot_ci_src_%j.out
#SBATCH --error=logs/plot_ci_src_%j.err

set -euo pipefail
mkdir -p logs

# ---- Activate conda (adjust env name if needed) ----
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate pymc_env || { echo "[ERR] conda env missing"; exit 1; }
echo "[INFO] Python: $(which python)"

# ---- Args ----
RESULTS_ROOT="${1:?Usage: sbatch plot_ci_truth_per_source.sbatch RESULTS_ROOT [\"family1 family2 ...\"] [forest_subdir]}"
FAMILIES="${2:-linear poisson logistic}"
FOREST_SUBDIR="${3:-beta_so_forest_per_source}"

OUT_DIR="${RESULTS_ROOT}/summary_figs_ci/${FOREST_SUBDIR}"
mkdir -p "${OUT_DIR}"

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] FAMILIES=${FAMILIES}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

# ---- Loop over families and reps ----
for fam in ${FAMILIES}; do
  echo "[INFO] Processing family: ${fam}"

  rep_dirs=("${RESULTS_ROOT}/${fam}/data"/rep*)
  if [ ${#rep_dirs[@]} -eq 0 ]; then
    echo "[WARN] No rep* directories found for ${fam}"
    continue
  fi

  for rep_dir in "${rep_dirs[@]}"; do
    rep=${rep_dir##*rep}
    echo "[INFO] Plotting CI and truth for ${fam}, rep ${rep}"

    python -u plot_ci_truth.py \
      --results-root "${RESULTS_ROOT}" \
      --families "${fam}" \
      --rep "${rep}" \
      --out-dir "${OUT_DIR}" || echo "[WARN] Failed for ${fam}, rep ${rep}"
  done
done

echo "[DONE] All per-source forest plots completed."