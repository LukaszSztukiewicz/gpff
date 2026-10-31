#!/bin/bash
#SBATCH --job-name=dd-damping-probe
#SBATCH --partition=gpu-9m
#SBATCH --time=00:09:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/damping_%j.out
# sbatch probe.sbatch <structures.db> [damping_probe.py options]
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_code
cd "$HERE"
py damping_probe.py "$MODEL" "$@"
