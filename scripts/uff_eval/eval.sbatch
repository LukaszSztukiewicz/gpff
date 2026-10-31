#!/bin/bash
#SBATCH --job-name=uff-eval
#SBATCH --partition=cpu-5h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/eval_%j.out
# sbatch eval.sbatch <out_dir> <ref>=<db> <name>=<db> ... [--limit N] [--reuse] [--records_only]
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_code
echo "args=$* host=$(hostname) code=$CODE"
py "$HERE/relax_eval.py" --method uff --n_jobs $SLURM_CPUS_PER_TASK "$@"
