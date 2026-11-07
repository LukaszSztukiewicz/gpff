#!/bin/bash
# Adaptive DD (feat/adaptive-dd 4c93744) on fancy-monkey ep5000, same protocol as submit_sweep.sh.
# Run on the hydra login node: bash -l submit_adaptive.sh
set -euo pipefail
G=/home/lsztuk/gpff; DS=$G/data/qm9/b36d9cd218f5998d; D=$G/scratch/adaptive_dd
export RUN=$G/runs/default/26-10-20/fancy-monkey/frozen MODELS=epoch_5000_model
export DATAPATH=$DS/qm9_filtered.db ASSETS=$DS/assets CONFIG_FILES=$D/configs_adaptive.txt
export GEN_SIF=$G/containers/gpff-py313-torch214.sif GEN_CODE=$G/frozen_code/4c93744-adaptive-dd/src
for configs in "add16_s03 add16_s01 add16_s03_plain add32_s03 asdd16_s03" "add256_s03 add256_s01" "asdd256_s03"; do
  id=$(CONFIGS=$configs sbatch --parsable --partition=gpu-2h --time=01:59:00 \
       --output=$D/logs/fast_rdkit_%j.out $G/scratch/sigma_oct21/fast_rdkit.sbatch)
  echo "adaptive $id: $configs" | tee -a $D/jobids.txt
done
