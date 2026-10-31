#!/bin/bash
# Full study: DD refinement of sdd16/sheun16 (GPU), then UFF evaluation of every set (CPU).
# Run from this folder on the hydra login node: bash -l submit_full.sh
set -euo pipefail
source env.sh
OUT=full
mkdir -p $OUT/refined $OUT/eval
F_MAX=0.05 MAX_STEPS=200

RAW="qm9=sets/qm9_test.db"
for s in dd16 sdd16 sheun16 sdd256 sheun256; do RAW="$RAW $s=$SAMPLES/$s/seed42069/samples.db"; done

refine_ids=""
for s in sdd16 sheun16; do
  id=$(sbatch --parsable refine.sbatch ${s}_dd $SAMPLES/$s/seed42069/samples.db $OUT/refined/${s}_dd.db \
       --f_max $F_MAX --max_steps $MAX_STEPS --trace $OUT/refined/${s}_dd_trace.npy)
  refine_ids="$refine_ids:$id"
  echo "refine $s: $id"
done

raw_id=$(sbatch --parsable eval.sbatch $OUT/eval $RAW)
echo "eval raw sets: $raw_id"
# after the refinements and the raw-set job, whose records it reuses
echo "eval all sets: $(sbatch --parsable --dependency=afterok$refine_ids:$raw_id eval.sbatch $OUT/eval $RAW \
  sdd16_dd=$OUT/refined/sdd16_dd.db sheun16_dd=$OUT/refined/sheun16_dd.db --reuse)"
