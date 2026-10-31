#!/bin/bash
#SBATCH --job-name=absorbing-eval
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SNAPSHOT:?}" "${STUDY:?}" "${EVALUATION_LIST:?}"
REFERENCE_PARTITION=${REFERENCE_PARTITION:-val}
G=/home/lsztuk/gpff
SIF=$G/containers/gpff-py313-torch214-gcc.sif
ECODE=$G/frozen_code/cf5d247-eval-tiers/src
ASSETS=$G/repro/assets_qm9_b36d9cd218f5998d
SMILES=$G/data/qm9/b36d9cd218f5998d/assets/qm9_smiles_values.txt
DATASET=$G/data/qm9/b36d9cd218f5998d/qm9_filtered.db
SOURCE=$G/runs/default/26-10-27/absorbing-lw0.25
DIRECTORY=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$EVALUATION_LIST")
EMPTY_DIR=$(mktemp -d "$STUDY/empty-eval-${SLURM_JOB_ID}-XXXXXX")
cd "$EMPTY_DIR"
export APPTAINERENV_OMP_NUM_THREADS=4 APPTAINERENV_MKL_NUM_THREADS=4
if [[ -f $DIRECTORY/conditionals.jsonl ]]; then
  [[ ! -f $DIRECTORY/conditional_table.json ]] || exit 0
  APPTAINERENV_PYTHONPATH="$SNAPSHOT/src" apptainer exec "$SIF" python -m gpff.analysis.absorbing_report "$DIRECTORY" --conditionals
  exit
fi
# Refuse incomplete generations, without rebuilding or deleting their files.
[[ -f $DIRECTORY/generation.json ]] || { echo "Incomplete generation: $DIRECTORY" >&2; exit 1; }
[[ ! -f $DIRECTORY/study_evaluation/metrics.json ]] || exit 0
if [[ ! -f $DIRECTORY/evaluation/results.json ]]; then
APPTAINERENV_PYTHONPATH="$ECODE" apptainer exec "$SIF" gpff_prepare_metadata \
  --samples_db_path "$DIRECTORY/samples.db" --device cpu --structure_workers 4 \
  --targets rdkit_valid failure n_fragments ring_counts elongation nfe smiles element_counts bond_lengths bond_angles
APPTAINERENV_PYTHONPATH="$ECODE" apptainer exec "$SIF" gpff_calculate_metrics \
  --samples_db_path "$DIRECTORY/samples.db" \
  --metrics rdkit_validity failures uniqueness novelty vun elongation_w1 bond_length_w1 bond_angle_w1 \
  --split_path "$SOURCE/frozen/split.npz" --reference_smiles_path "$SMILES" \
  --reference_elongation_path "$G/data/qm9/b36d9cd218f5998d/assets/qm9_elongation_values.npy" \
  --reference_bond_lengths_path "$ASSETS/qm9_bond_lengths.npz" \
  --reference_bond_angles_path "$ASSETS/qm9_bond_angles.npz"
fi
APPTAINERENV_PYTHONPATH="$SNAPSHOT/src" apptainer exec "$SIF" python -m gpff.analysis.absorbing_report \
  "$DIRECTORY" --dataset "$DATASET" --split "$SOURCE/frozen/split.npz" --assets "$SMILES" --reference-partition "$REFERENCE_PARTITION"
