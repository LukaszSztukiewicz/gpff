#!/bin/bash
# Adaptive DD with SDD's noise in the refine phase (4c93744), fancy-monkey ep5000, same protocol as
# submit_adaptive.sh. Run on the hydra login node: bash -l submit_refine_noise.sh
set -euo pipefail
G=/home/lsztuk/gpff; DS=$G/data/qm9/b36d9cd218f5998d; D=$G/scratch/adaptive_dd
export RUN=$G/runs/default/26-10-20/fancy-monkey/frozen MODELS=epoch_5000_model
export DATAPATH=$DS/qm9_filtered.db ASSETS=$DS/assets CONFIG_FILES=$D/configs_adaptive.txt
export GEN_SIF=$G/containers/gpff-py313-torch214.sif GEN_CODE=$G/frozen_code/4c93744-adaptive-dd/src
for configs in "add16_s03_n05 add16_s03_n10 add32_s03_n10" "add256_s03_n05 add256_s03_n10" "asdd256_s03_n10"; do
  id=$(CONFIGS=$configs sbatch --parsable --partition=gpu-2h --time=01:59:00 \
       --output=$D/logs/fast_rdkit_%j.out $G/scratch/sigma_oct21/fast_rdkit.sbatch)
  echo "refine-noise $id: $configs" | tee -a $D/jobids.txt
done
