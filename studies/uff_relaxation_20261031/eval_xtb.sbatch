#!/bin/bash
#SBATCH --job-name=xtb-eval
#SBATCH --partition=cpu-5h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/eval_xtb_%j.out
# sbatch eval_xtb.sbatch <out_dir> <ref>=<db> <name>=<db> ... [--limit N] [--reuse] [--records_only]
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_code
echo "args=$* host=$(hostname) code=$CODE xtb=$XTB"
py_xtb "$HERE/relax_eval.py" --method xtb --n_jobs $SLURM_CPUS_PER_TASK "$@"
