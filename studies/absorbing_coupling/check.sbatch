#!/bin/bash
#SBATCH --job-name=absorbing-check
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SNAPSHOT:?}" "${SIF:?}" "${STUDY:?}"
EMPTY_DIR=$(mktemp -d "$STUDY/empty-check-${SLURM_JOB_ID}-XXXXXX")
cd "$EMPTY_DIR"
export APPTAINERENV_PYTHONPATH="$SNAPSHOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for SCRIPT in check.sbatch run.sbatch evaluate.sbatch submit.sh; do
  bash -n "$SNAPSHOT/studies/absorbing_coupling/$SCRIPT"
done
apptainer exec "$SIF" python -m pytest "$SNAPSHOT/tests/test_absorbing_types.py" \
  "$SNAPSHOT/tests/test_type_readout.py" "$SNAPSHOT/tests/test_absorbing_study.py" \
  --junitxml="$STUDY/check-${SLURM_JOB_ID}.xml" -o cache_dir="$EMPTY_DIR/pytest-cache" -q
