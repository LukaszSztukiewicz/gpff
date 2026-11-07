#!/bin/bash
#SBATCH --job-name=add-smoke
#SBATCH --partition=gpu-9m
#SBATCH --time=00:09:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=/home/lsztuk/gpff/scratch/adaptive_dd/logs/smoke_%j.out
# 512 samples of adaptive DD at 16 steps on fancy-monkey ep5000, to see it run
set -euo pipefail
G=/home/lsztuk/gpff; F=$G/runs/default/26-10-20/fancy-monkey/frozen; CODE=$G/frozen_code/4c93744-adaptive-dd/src
TMP=/tmp/add-smoke-$SLURM_JOB_ID; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
apptainer exec --nv --env PYTHONPATH=$CODE $G/containers/gpff-py313-torch214.sif gpff_generate \
  sampler=adaptive_direct_denoising model_path=$F/epoch_5000_model split_path=$F/split.npz \
  datapath=$G/data/qm9/b36d9cd218f5998d/qm9_filtered.db data_workdir=$TMP n_samples=512 batch_size=128 \
  seed=42069 num_workers=4 results_dir=$TMP output_db=$TMP/samples.db 2>&1 | grep -E "n_samples|Error|Traceback|error" 
