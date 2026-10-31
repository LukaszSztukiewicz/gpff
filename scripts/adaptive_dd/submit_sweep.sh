#!/bin/bash
# Transport sweep (feat/dd-transport 39424e1) on fancy-monkey ep5000: generate + RDKit validity,
# uniqueness, novelty and elongation JS with sigma_oct21's fast_rdkit.sbatch (same protocol as the
# earlier ratio sweep). Run on the hydra login node: bash -l submit_sweep.sh
set -euo pipefail
G=/home/lsztuk/gpff; DS=$G/data/qm9/b36d9cd218f5998d; D=$G/scratch/adaptive_dd
export RUN=$G/runs/default/26-10-20/fancy-monkey/frozen MODELS=epoch_5000_model
export DATAPATH=$DS/qm9_filtered.db ASSETS=$DS/assets CONFIG_FILES=$D/configs_adaptive.txt
export GEN_SIF=$G/containers/gpff-py313-torch214.sif GEN_CODE=$G/frozen_code/39424e1-dd-transport/src
for configs in "tdd16_s10_r75 tdd16_s10_r90 tdd16_s03_r75 tdd16_s03_r90 tdd16_s01_r75 tdd16_s01_r90" \
               "tdd256_s10_r75 tdd256_s03_r75 tdd256_s01_r75" "tdd256_s10_r90 tdd256_s03_r90 tdd256_s01_r90"; do
  id=$(CONFIGS=$configs sbatch --parsable --partition=gpu-2h --time=01:59:00 \
       --output=$D/logs/fast_rdkit_%j.out $G/scratch/sigma_oct21/fast_rdkit.sbatch)
  echo "sweep $id: $configs" | tee -a $D/jobids.txt
done
