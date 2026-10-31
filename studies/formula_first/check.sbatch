#!/bin/bash
#SBATCH --job-name=formula-check
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SNAPSHOT:?}" "${STUDY:?}" "${SIF:?}"
EMPTY_DIR=$(mktemp -d "$STUDY/check-${SLURM_JOB_ID}-XXXXXX")
cd "$EMPTY_DIR"
export APPTAINERENV_PYTHONPATH="$SNAPSHOT/src"
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
for script in "$SNAPSHOT"/studies/formula_first/*.sh "$SNAPSHOT"/studies/formula_first/*.sbatch; do
  bash -n "$script"
done
apptainer exec "$SIF" python -m pytest "$SNAPSHOT/tests/test_formula_prior.py" \
  -q -o cache_dir="$EMPTY_DIR/cache" --junitxml="$STUDY/check-${SLURM_JOB_ID}.xml"
apptainer exec "$SIF" python "$SNAPSHOT/studies/formula_first/prepare.py" \
  "$DATASET_PATH" "$SPLIT" "$STUDY/formula.json"
export APPTAINER_PWD="$EMPTY_DIR"
bash "$SNAPSHOT/studies/formula_first/submit.sh"
