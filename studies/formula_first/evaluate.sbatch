#!/bin/bash
#SBATCH --job-name=formula-pilot-eval
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SNAPSHOT:?}" "${STUDY:?}" "${QUALIFIED_JOB:?}"
source "$SNAPSHOT/studies/formula_first/common.sh"
DIRECTORY=$(cat "$STUDY/qualified-$QUALIFIED_JOB.txt")/pretrained
EMPTY_DIR=$(mktemp -d "$STUDY/eval-${SLURM_JOB_ID}-XXXXXX")
cd "$EMPTY_DIR"
export APPTAINERENV_PYTHONPATH="$G/frozen_code/cf5d247-eval-tiers/src"
export APPTAINERENV_OMP_NUM_THREADS=4 APPTAINERENV_MKL_NUM_THREADS=4
apptainer exec "$SIF" gpff_prepare_metadata --samples_db_path "$DIRECTORY/samples.db" \
  --device cpu --structure_workers 4 \
  --targets rdkit_valid failure n_fragments ring_counts elongation nfe smiles element_counts bond_lengths bond_angles
apptainer exec "$SIF" gpff_calculate_metrics --samples_db_path "$DIRECTORY/samples.db" \
  --metrics rdkit_validity failures uniqueness novelty vun elongation_w1 bond_length_w1 bond_angle_w1 \
  --split_path "$SPLIT" \
  --reference_smiles_path "$G/data/qm9/b36d9cd218f5998d/assets/qm9_smiles_values.txt" \
  --reference_elongation_path "$G/data/qm9/b36d9cd218f5998d/assets/qm9_elongation_values.npy" \
  --reference_bond_lengths_path "$G/repro/assets_qm9_b36d9cd218f5998d/qm9_bond_lengths.npz" \
  --reference_bond_angles_path "$G/repro/assets_qm9_b36d9cd218f5998d/qm9_bond_angles.npz"
