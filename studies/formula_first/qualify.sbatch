#!/bin/bash
#SBATCH --job-name=formula-qualify
#SBATCH --partition=gpu-2h
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SNAPSHOT:?}" "${STUDY:?}"
source "$SNAPSHOT/studies/formula_first/common.sh"
EMPTY_DIR=$(mktemp -d "$STUDY/qualify-${SLURM_JOB_ID}-XXXXXX")
cd "$EMPTY_DIR"
export APPTAINERENV_PYTHONPATH="$SNAPSHOT/src"
export APPTAINERENV_OMP_NUM_THREADS=4 APPTAINERENV_MKL_NUM_THREADS=4
for entry in base:-0.7 local:-1.2 broad:-0.2; do
  name=${entry%%:*}
  mean=${entry#*:}
  apptainer exec --nv "$SIF" gpff_train "${TRAIN_ARGS[@]}" \
    "run.id=smoke-$name" "run.path=$EMPTY_DIR" "run.data_dir=$(dirname "$DATASET_PATH")" \
    run.ckpt_path=null "data.data_workdir=$EMPTY_DIR/data-$name" \
    data.num_workers=4 logger=csv print_config=false trainer.max_epochs=2 \
    trainer.limit_train_batches=4 trainer.limit_val_batches=2 trainer.limit_test_batches=2 \
    'callbacks.milestone_checkpoint.epochs=[1,2]' \
    "pseudo_pes.geometry_sampler.sigma_mean=$mean" > "$STUDY/smoke-$name-${SLURM_JOB_ID}.log" 2>&1
  apptainer exec --nv "$SIF" gpff_generate prior=formula \
    "prior.formula_path=$STUDY/formula.json" "model_path=$EMPTY_DIR/smoke-$name/milestones/epoch_2_model" \
    "split_path=$SPLIT" "datapath=$DATASET_PATH" n_samples=8 batch_size=8 sampler.n_steps=4 \
    "output_db=$EMPTY_DIR/generated-$name/samples.db" > "$STUDY/generate-$name-${SLURM_JOB_ID}.log" 2>&1
done
apptainer exec --nv "$SIF" gpff_generate prior=formula \
  "prior.formula_path=$STUDY/formula.json" \
  "model_path=$REFERENCE/frozen/epoch_5000_model" \
  "split_path=$SPLIT" "datapath=$DATASET_PATH" n_samples=128 batch_size=128 \
  sampler.n_steps=256 seed=42069 "output_db=$EMPTY_DIR/pretrained/samples.db" \
  > "$STUDY/pretrained-${SLURM_JOB_ID}.log" 2>&1
printf '%s\n' "$EMPTY_DIR" > "$STUDY/qualified-${SLURM_JOB_ID}.txt"
