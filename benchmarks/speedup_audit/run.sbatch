#!/bin/bash
#SBATCH --job-name=gpff-speed-audit
#SBATCH --partition=gpu-2d
#SBATCH --gres=gpu:1
#SBATCH --constraint=40gb|80gb|h100|h200
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/lsztuk/gpff/scratch/speedup-audit-20261029/%x-%j.out
set -euo pipefail
source /home/lsztuk/gpff/env.sh
study=$GPFF/scratch/speedup-audit-20261029
code=$GPFF/frozen_code/speedup-audit-20261029
mkdir -p "$study/empty-$SLURM_JOB_ID" "$study/results"
cd "$study/empty-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$study/inductor-$SLURM_JOB_ID"
export APPTAINERENV_TRITON_CACHE_DIR="$study/triton-$SLURM_JOB_ID"
export APPTAINERENV_CUBLAS_WORKSPACE_CONFIG=:4096:8
export APPTAINERENV_PYTHONPATH="$code/8991e5f/src"
nvidia-smi --query-gpu=name,driver_version,uuid,power.limit --format=csv
run() {
    local name=$1
    shift
    echo "AUDIT_START $name $(date -Is)"
    apptainer exec --nv "$SIF" python "$study/audit.py" "$@" --out "$study/results/$name-$SLURM_JOB_ID.json"
    echo "AUDIT_DONE $name $(date -Is)"
}
case ${1:-pilot} in
  pilot)
    run components components
    run neural128 neural --batch 128 --train
    run converged128 converged --batch 128
    run deferred16 deferred --steps 16
    ;;
  endpoints)
    for revision in 759d3d6 8991e5f 8991e5f 759d3d6; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      run "endpoint-$revision-$RANDOM" workload --batch 512 --train --serialize
    done
    ;;
  compiled)
    run compiled128 neural --batch 128 --compile --train
    for revision in 759d3d6 8991e5f; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      run "compiled-endpoint-$revision" workload --batch 512 --train --compile --serialize
    done
    ;;
  neural512)
    run neural512 neural --batch 512 --train
    run neural128w512 neural --batch 128 --width 512 --train
    ;;
  test-endpoints)
    for revision in 759d3d6 8991e5f 8991e5f 759d3d6; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      run "test-endpoint-$revision-$RANDOM" workload --batch 128 --train --serialize
    done
    ;;
  test-compiled)
    run test-compiled128 neural --batch 128 --compile --train
    export APPTAINERENV_PYTHONPATH="$code/8991e5f/src"
    run test-compiled-head workload --batch 128 --train --compile --serialize
    ;;
  per-commit)
    for revision in 759d3d6 ef74aa3 b24eda8 93cca41 4839f63 24a2ce2 d464b76 38cb9db 47efdc3 3c78267 831b100 a0e6c9b 697f0f6 3b17abc ea093f3 428dc6a 632660a 8991e5f; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      run "increment-$revision" workload --batch 128 --train --serialize
    done
    ;;
  extra)
    run extra extra
    run deferred16 deferred --steps 16
    run real-adaptive adaptive
    export APPTAINERENV_PYTHONPATH="$code/8991e5f/src"
    apptainer exec --nv "$SIF" python -m pytest "$code/8991e5f/tests/test_painn.py" "$code/8991e5f/tests/test_alignment.py" "$code/8991e5f/tests/test_callbacks.py" "$code/8991e5f/tests/test_neighbors.py" "$code/8991e5f/tests/test_sampler.py" "$code/8991e5f/tests/test_results.py" "$code/8991e5f/tests/test_priors.py" "$code/8991e5f/tests/test_shape_predictor.py" -q
    ;;
  pipeline)
    for revision in 759d3d6 8991e5f 8991e5f 759d3d6; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      apptainer exec --nv "$SIF" python "$study/pipeline.py" --out "$study/results/pipeline-$revision-$RANDOM-$SLURM_JOB_ID.json"
    done
    ;;
  deterministic)
    run deterministic-neural128 neural --batch 128 --train --deterministic
    for revision in 759d3d6 8991e5f; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      run "deterministic-endpoint-$revision" workload --batch 128 --train --serialize --deterministic
    done
    ;;
  long)
    for revision in 759d3d6 8991e5f; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      run "long-$revision" workload --steps 256 --serialize
    done
    ;;
  overlap-compiled)
    run deferred-compiled deferred --compile
    ;;
  pipeline512)
    for revision in 759d3d6 8991e5f 8991e5f 759d3d6; do
      export APPTAINERENV_PYTHONPATH="$code/$revision/src"
      apptainer exec --nv "$SIF" python "$study/pipeline.py" --batch 512 --workers 12 --out "$study/results/pipeline512-$revision-$RANDOM-$SLURM_JOB_ID.json"
    done
    ;;
  adaptive-check)
    run adaptive-precision adaptive
    ;;
  *) echo "Unknown mode $1" >&2; exit 2 ;;
esac
