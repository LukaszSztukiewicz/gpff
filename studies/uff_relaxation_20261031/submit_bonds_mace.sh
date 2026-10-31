#!/bin/bash
# Per-bond-type study with GFN2-xTB and MACE-OFF24(M) for fancy-monkey ep5000 (full/) and the paper
# baseline SOB999 (sob/); UFF only for fancy-monkey, from its existing records.
# Run from this folder on the hydra login node: bash -l submit_bonds_mace.sh
set -euo pipefail
source env.sh
SOB=$GPFF/runs/stefaan_original_baseline/seed999/frozen
SOB_SAMPLES=$SOB/repro/best_model
mkdir -p full/mace full/eval_mace sob/refined sob/mace sob/eval_xtb sob/eval_mace

sets() {  # <reference.db> <samples dir> <refined dir>: name=db for the 8 sets
  echo "qm9=$1"
  for s in dd16 sdd16 sheun16 sdd256 sheun256; do echo "$s=$2/$s/seed42069/samples.db"; done
  echo "sdd16_dd=$3/sdd16_dd.db sheun16_dd=$3/sheun16_dd.db"
}
FM=$(sets sets/qm9_test.db $SAMPLES full/refined | tr "\n" " ")
SB=$(sets sets/sob_qm9_test.db $SOB_SAMPLES sob/refined | tr "\n" " ")
read -r -a FM_A <<< "$(echo $FM | cut -d" " -f1-4)"; read -r -a FM_B <<< "$(echo $FM | cut -d" " -f5-8)"
read -r -a SB_A <<< "$(echo $SB | cut -d" " -f1-4)"; read -r -a SB_B <<< "$(echo $SB | cut -d" " -f5-6)"
read -r -a SB_C <<< "$(echo $SB | cut -d" " -f7-8)"; read -r -a SB_RAW <<< "$(echo $SB | cut -d" " -f1-6)"
read -r -a FM_ALL <<< "$FM"; read -r -a SB_ALL <<< "$SB"

# SOB999 DD refinements, same settings as fancy-monkey's
r1=$(MODEL=$SOB/best_model sbatch --parsable refine.sbatch sdd16_dd $SOB_SAMPLES/sdd16/seed42069/samples.db sob/refined/sdd16_dd.db --f_max 0.05 --max_steps 200)
r2=$(MODEL=$SOB/best_model sbatch --parsable refine.sbatch sheun16_dd $SOB_SAMPLES/sheun16/seed42069/samples.db sob/refined/sheun16_dd.db --f_max 0.05 --max_steps 200)

# MACE-OFF24(M) relaxations (GPU)
m1=$(sbatch --parsable mace.sbatch full/mace "${FM_A[@]}" --batch_size 512)
m2=$(sbatch --parsable mace.sbatch full/mace "${FM_B[@]}" --batch_size 512)
m3=$(sbatch --parsable mace.sbatch sob/mace "${SB_A[@]}" --batch_size 512)
m4=$(sbatch --parsable mace.sbatch sob/mace "${SB_B[@]}" --batch_size 512)
m5=$(sbatch --parsable --dependency=afterok:$r1:$r2 mace.sbatch sob/mace "${SB_C[@]}" --batch_size 512)

# xTB for SOB999 (CPU), raw sets now and the refinements after them
x1=$(sbatch --parsable eval_xtb.sbatch sob/eval_xtb "${SB_RAW[@]}" --records_only)
x2=$(sbatch --parsable --dependency=afterok:$r1:$r2 eval_xtb.sbatch sob/eval_xtb "${SB_C[@]}" --records_only)
x3=$(sbatch --parsable --dependency=afterok:$x1:$x2 --cpus-per-task=2 eval_xtb.sbatch sob/eval_xtb "${SB_ALL[@]}" --reuse)

# RDKit checks and bond types of the MACE relaxations; fancy-monkey's UFF and xTB summaries again,
# now with bond types, from their records
e1=$(sbatch --parsable --dependency=afterok:$m1:$m2 eval.sbatch full/eval_mace "${FM_ALL[@]}" --method precomputed --relaxed_dir full/mace)
e2=$(sbatch --parsable --dependency=afterok:$m3:$m4:$m5 eval.sbatch sob/eval_mace "${SB_ALL[@]}" --method precomputed --relaxed_dir sob/mace)
e3=$(sbatch --parsable --cpus-per-task=2 eval.sbatch full/eval_uff "${FM_ALL[@]}" --reuse)
e4=$(sbatch --parsable --cpus-per-task=2 eval_xtb.sbatch full/eval_xtb "${FM_ALL[@]}" --reuse)
echo "refine $r1 $r2 | mace $m1 $m2 $m3 $m4 $m5 | xtb $x1 $x2 $x3 | eval $e1 $e2 $e3 $e4"
echo "$r1 $r2 $m1 $m2 $m3 $m4 $m5 $x1 $x2 $x3 $e1 $e2 $e3 $e4" > logs/bonds_mace_jobs.txt
