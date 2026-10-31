#!/bin/bash
#SBATCH --job-name=sob-prep
#SBATCH --partition=gpu-9m
#SBATCH --time=00:09:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/sob_prep_%j.out
# SOB999: its own QM9 reference set, the pairing check, and whether its model runs in this stack
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
cd "$HERE"
SOB=$GPFF/runs/stefaan_original_baseline/seed999/frozen
SOB_SAMPLES=$SOB/repro/best_model
py export_reference.py $GPFF/data/qm9/b36d9cd218f5998d/qm9_filtered.db $SOB/split.npz sets/sob_qm9_test.db
py pair_check.py sets/sob_qm9_test.db $(for s in dd16 sdd16 sheun16 sdd256 sheun256; do echo $SOB_SAMPLES/$s/seed42069/samples.db; done)
MODEL=$SOB/best_model py refine_dd.py $SOB/best_model $SOB_SAMPLES/sdd16/seed42069/samples.db pilot/sob_dd_test.db --f_max 0.05 --max_steps 200 --limit 256
