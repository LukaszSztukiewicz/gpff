#!/bin/bash
#SBATCH --job-name=gpff-speed-summary
#SBATCH --partition=cpu-9m
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:09:00
#SBATCH --output=/home/lsztuk/gpff/scratch/speedup-audit-20261029/summary-%j.out
set -euo pipefail
source /home/lsztuk/gpff/env.sh
study=$GPFF/scratch/speedup-audit-20261029
cd "$study/empty"
export APPTAINERENV_PYTHONPATH=$GPFF/frozen_code/speedup-audit-20261029/8991e5f/src
apptainer exec "$SIF" python "$study/summarize.py"
