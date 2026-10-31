#!/bin/bash
# MACE part of submit_bonds_mace.sh again, after the first try ran out of memory on 40 GB GPUs at
# batch 512: batch 256, two sets per job. Run from this folder on the hydra login node.
set -euo pipefail
source env.sh
SOB=$GPFF/runs/stefaan_original_baseline/seed999/frozen
SOB_SAMPLES=$SOB/repro/best_model
fm() { case $1 in qm9) echo "qm9=sets/qm9_test.db";; *_dd) echo "$1=full/refined/$1.db";; *) echo "$1=$SAMPLES/$1/seed42069/samples.db";; esac; }
sb() { case $1 in qm9) echo "qm9=sets/sob_qm9_test.db";; *_dd) echo "$1=sob/refined/$1.db";; *) echo "$1=$SOB_SAMPLES/$1/seed42069/samples.db";; esac; }
ALL="qm9 dd16 sdd16 sheun16 sdd256 sheun256 sdd16_dd sheun16_dd"
fm_ids=""; sb_ids=""
for pair in "qm9 dd16" "sdd16 sheun16" "sdd256 sheun256" "sdd16_dd sheun16_dd"; do
  fm_ids="$fm_ids:$(sbatch --parsable mace.sbatch full/mace $(for s in $pair; do fm $s; done) --batch_size 256)"
  sb_ids="$sb_ids:$(sbatch --parsable mace.sbatch sob/mace $(for s in $pair; do sb $s; done) --batch_size 256)"
done
e1=$(sbatch --parsable --dependency=afterok$fm_ids eval.sbatch full/eval_mace $(for s in $ALL; do fm $s; done) --method precomputed --relaxed_dir full/mace)
e2=$(sbatch --parsable --dependency=afterok$sb_ids eval.sbatch sob/eval_mace $(for s in $ALL; do sb $s; done) --method precomputed --relaxed_dir sob/mace)
echo "mace fm$fm_ids sob$sb_ids | eval $e1 $e2"
echo "${fm_ids//:/ } ${sb_ids//:/ } $e1 $e2 5117993 5117994" > logs/mace_jobs.txt
