#!/bin/bash
#SBATCH --job-name=mace-relax
#SBATCH --partition=gpu-2h
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/mace_%j.out
# sbatch mace.sbatch <out_dir> <name>=<db> ... [mace_relax.py options]; writes <out_dir>/<name>.db
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
OUT=$1; shift
mkdir -p "$OUT"
SETS=(); OPTS=()
for a in "$@"; do if [[ $a == *=* && $a != --* ]]; then SETS+=("$a"); else OPTS+=("$a"); fi; done
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) model=$MACE_MODEL"
for s in "${SETS[@]}"; do
  echo "== ${s%%=*}"
  py_mace "$HERE/mace_relax.py" "$MACE_MODEL" "${s#*=}" "$OUT/${s%%=*}.db" "${OPTS[@]}"
done
