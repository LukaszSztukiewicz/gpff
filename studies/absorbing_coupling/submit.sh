#!/bin/bash
# Run on Hydra. DRY=1 (default) previews; DRY=0 submits explicit task arrays.
set -euo pipefail
: "${SNAPSHOT:?absolute frozen snapshot}" "${STUDY:?absolute study directory}"
SIF=${SIF:-/home/lsztuk/gpff/containers/gpff-py313-torch214-gcc.sif}
TASKS=${1:?JSONL task file}
COUNT=$(wc -l < "$TASKS")
export SNAPSHOT STUDY SIF TASKS
ARGS=(--parsable --array="0-$((COUNT-1))%2" --output="$STUDY/logs/pilot-%A_%a.out" --export=ALL "$SNAPSHOT/studies/absorbing_coupling/run.sbatch")
if [[ ${DRY:-1} == 1 ]]; then
  printf 'sbatch %q ' "${ARGS[@]}"
  printf '\n'
  cat "$TASKS"
else
  JOB=$(sbatch "${ARGS[@]}")
  printf '%s tasks=%s snapshot=%s\n' "$JOB" "$TASKS" "$SNAPSHOT" | tee -a "$STUDY/jobids.txt"
fi
