# GPFF speedup audit, 2026-10-29

Branch: `perf/futher-speedup` at `8991e5f`; base/merge-base with
`fix/seeded-split`: `759d3d65ab4dbae0962c349fef73638f45e48e9d`.

All Python, tests, training and sampling execute on Hydra under Slurm inside
`gpff-py313-torch214-gcc.sif`. No packages are installed. Existing training
runs, the editable checkout, and dataset contents are not modified.

Remote study: `~/gpff/scratch/speedup-audit-20261029/`.
Read-only revision snapshots: `~/gpff/frozen_code/speedup-audit-20261029/<sha>/`.

## Measurements

- `audit.py components`: alternating before/after component timings. Historical
  changed modules are imported under distinct names with fixed HEAD dependencies.
  A historical dependency is explicitly substituted where it affects the test
  (padding and Atoms conversion). These isolate the changed operation, rather
  than represent full-revision application throughput.
- `audit.py neural`: actual historical PaiNN modules with identical weights,
  256/512 channels, 4 interactions, 600 radial basis functions, cutoff 150,
  float32, matmul precision `high`; both reused and fresh neighbor tensors.
  Manual training steps include forward, MSE, backward and Adam. They exclude
  data loading, Lightning, EMA and validation. Compilation is opt-in.
- `audit.py workload`: a separate process with each revision's complete
  `PYTHONPATH`, measuring forward/backward/Adam/EMA training and full direct
  denoising (16 steps, CPU Gaussian prior, transfer, neighbor construction,
  final CPU clone and optional relaxed-result conversion). The training batch
  is already resident on GPU; fresh neighbor tensor identities mimic incoming
  DataLoader batches. These are throughput tests with matched initialized
  model weights, not molecule-quality or convergence studies.
- `pipeline.py`: actual GPFF Lightning task, AdamW, weighted loss, metrics, EMA,
  CPU pseudo-PES transforms and multi-worker DataLoader. It repeats 512 real
  cached QM9 molecules to measure training throughput without writing dataset
  or split files. It excludes validation, checkpointing and experiment logging.
- `audit.py converged`: model prediction at controlled convergence fractions.
  `adaptive` additionally measures a real trained width-512 checkpoint.
- `audit.py deferred`: three consecutive batches including ASE SQLite writes,
  RDKit validity checks and the final queue flush. The checks use real QM9
  geometries, so this tests the pipeline mechanism with realistic valid
  structures rather than the pathological outputs of an untrained model.
- `extra`: full DD16 ablation of the bookkeeping commit and covariance
  precision diagnostics.

Inputs: 512 structures selected with NumPy RNG seed 999 from the training
split of the 123,569-row paper QM9 database `b36d9cd218f5998d`. Raw structures
are fetched with an explicitly read-only SQLite connection. Sampling uses
128 structures. Model seed: 999; sampling prior seed: 42069.

Every timed block synchronizes CUDA before and after wall-clock timing.
Components usually have seven alternating paired trials, with multiple calls
per trial; full workloads have seven training and five sampling trials.
Warm-up is excluded. First-call latency is recorded separately. Raw samples,
environment details, peak allocated memory and correctness diagnostics are
stored in `results/*.json`. `speedup` in the original component JSON is the
median paired ratio; report ratios can instead be recomputed from the raw
samples as the ratio of the two medians. They can differ for multimodal timing
distributions. Compilation first-call timing can reuse artifacts already
produced within the same Slurm job; it is not guaranteed to be an empty-cache
installation cost.

Results are conditional on the recorded GPU, batch size, precision, compile
mode, CPU and sampling options. A component speedup is not a full training or
generation speedup. Small cross-process differences need the paired ablation
to support attributing them to a commit. GPU clocks are not pinned. Main
cluster and test partition hardware results must not be mixed into one ratio.

## Reproduction

Submit from `~/gpff` on Hydra (adjust the partition and resources):

```sh
sbatch scratch/speedup-audit-20261029/run.sbatch pilot
sbatch scratch/speedup-audit-20261029/run.sbatch endpoints
sbatch scratch/speedup-audit-20261029/run.sbatch compiled
sbatch scratch/speedup-audit-20261029/run.sbatch neural512
```

For the 12 GB test GPU use `test-endpoints`, `test-compiled`, `per-commit`,
`extra`, `pipeline` or `deterministic`, with `--partition=gpu-test
--constraint= --time=00:15:00 --mem=24G` passed to `sbatch`.

The first pilot stopped at the deferred-pipeline test because the synthetic
metadata fixture lacked `converged`. Completed earlier measurements were
preserved; the fixture was corrected and the deferred test is rerun by
`extra`. No production source changes were made.
