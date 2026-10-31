#!/bin/bash
# GFN2-xTB version of the study, on the same sets as submit_full.sh (whose DD refinements must exist).
# Run from this folder on the hydra login node: bash -l submit_xtb.sh
set -euo pipefail
source env.sh
OUT=full
mkdir -p $OUT/eval_xtb

RAW="qm9=sets/qm9_test.db"
for s in dd16 sdd16 sheun16 sdd256 sheun256; do RAW="$RAW $s=$SAMPLES/$s/seed42069/samples.db"; done
DD="sdd16_dd=$OUT/refined/sdd16_dd.db sheun16_dd=$OUT/refined/sheun16_dd.db"

# two halves in parallel, then the summary over all sets from their records
a=$(sbatch --parsable eval_xtb.sbatch $OUT/eval_xtb $RAW --records_only)
b=$(sbatch --parsable eval_xtb.sbatch $OUT/eval_xtb $DD --records_only)
echo "xtb raw sets: $a, refined sets: $b"
echo "xtb summary: $(sbatch --parsable --dependency=afterok:$a:$b --cpus-per-task=2 eval_xtb.sbatch $OUT/eval_xtb $RAW $DD --reuse)"
