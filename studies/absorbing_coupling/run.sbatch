#!/bin/bash
#SBATCH --job-name=absorbing-pilot
#SBATCH --partition=gpu-5h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SNAPSHOT:?}" "${SIF:?}" "${STUDY:?}" "${TASKS:?}"
export SOURCE=/home/lsztuk/gpff/runs/default/26-10-27/absorbing-lw0.25
export DATASET=/home/lsztuk/gpff/data/qm9/b36d9cd218f5998d/qm9_filtered.db
export ASSETS=/home/lsztuk/gpff/data/qm9/b36d9cd218f5998d/assets/qm9_smiles_values.txt
export EMPTY_DIR
EMPTY_DIR=$(mktemp -d "$STUDY/empty-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}-XXXXXX")
cd "$EMPTY_DIR"
export APPTAINERENV_PYTHONPATH="$SNAPSHOT/src"
export APPTAINERENV_OMP_NUM_THREADS=4 APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_WANDB_MODE=offline
apptainer exec --nv "$SIF" python "$SNAPSHOT/studies/absorbing_coupling/execute_task.py" "$TASKS"
