#!/bin/bash
#SBATCH --job-name=uff-refine-dd
#SBATCH --partition=gpu-2h
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/refine_%j.out
# sbatch refine.sbatch <name> <samples.db> <out.db> [refine_dd.py options]
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_code
NAME=$1 IN=$2 OUT=$3; shift 3
echo "name=$NAME in=$IN out=$OUT args=$* host=$(hostname) code=$CODE"
py "$HERE/refine_dd.py" "$MODEL" "$IN" "$OUT" "$@"
