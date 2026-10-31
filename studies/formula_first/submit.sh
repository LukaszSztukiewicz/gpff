#!/bin/bash
# Default dry-run must itself run inside a Slurm job: hy composes with Python.
set -euo pipefail
: "${SNAPSHOT:?}" "${STUDY:?}"
source "$SNAPSHOT/studies/formula_first/common.sh"
export DATA_FAMILY=qm9 DATASET=b36d9cd218f5998d IDS="$STUDY/jobids.txt"
for entry in base:-0.7 local:-1.2 broad:-0.2; do
  name=${entry%%:*}
  mean=${entry#*:}
  args=("${SNAPSHOT##*/}" --constraint=40gb --time=2-00:00:00
    "${TRAIN_ARGS[@]}" "run.id=formula-$name" "run.path=$STUDY/training"
    "pseudo_pes.geometry_sampler.sigma_mean=$mean")
  if [[ ${SUBMIT:-0} == 1 ]]; then
    : "${QUALIFIED_JOB:?Set the completed qualification job ID}"
    [[ -f $STUDY/qualified-$QUALIFIED_JOB.txt ]]
    [[ ! -e $STUDY/training/formula-$name ]]
    [[ ! -f $STUDY/jobids.txt ]] || ! grep -q "^formula-$name train=" "$STUDY/jobids.txt"
    env -u DRY "$G/bin/hy" train "${args[@]}"
  else
    : "${SLURM_JOB_ID:?Dry-run config composition requires Slurm}"
    DRY=1 "$G/bin/hy" train "${args[@]}" > "$STUDY/dry-$name-${SNAPSHOT##*/}.txt"
  fi
done
