#!/bin/bash
#SBATCH --job-name=mace-bench
#SBATCH --partition=gpu-9m
#SBATCH --time=00:09:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/mace_bench_%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
cd "$HERE"
nvidia-smi --query-gpu=name --format=csv,noheader
S=$SAMPLES/sdd16/seed42069/samples.db
py_mace mace_relax.py $MACE_MODEL $S pilot/mace_sdd16.db --limit 1024 --batch_size 512
py_mace mace_ase_check.py $MACE_MODEL $S pilot/mace_sdd16.db --n 16
py_mace mace_relax.py $MACE_MODEL sets/qm9_test.db pilot/mace_qm9.db --check 64
