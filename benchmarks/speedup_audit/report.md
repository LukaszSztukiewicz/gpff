# GPFF GPU speedup audit — 29 October 2026

Audited all 17 commits on `perf/futher-speedup`, from merge-base
`759d3d6` (`fix/seeded-split`) through `8991e5f`.
All execution took place on Hydra under Slurm, inside the existing
Python 3.13 / PyTorch 2.14.1+cu130 GCC container. Code was pinned in read-only
snapshots; production code, existing runs and datasets were left untouched.

The branch speeds up sampling substantially. The model's training computation
also improves, but this translates into only about **1.6% greater sustained
throughput in the complete Lightning pipelines tested here**. Several commits
accelerate optional sampling paths and provide no benefit to default Gaussian
sampling or to training steps.

**Measured complete-branch effects.** Times are per batch. Full sampler timings
include the CPU Gaussian prior, transfer, neighbor construction, denoising,
final CPU copy and conversion to relaxed results; they exclude dataset loading,
SQLite writes and RDKit progress checks. The latter are measured separately.

| Workload | GPU and settings | Base → branch | Throughput multiplier |
|---|---|---:|---:|
| GPU-resident training, forward/backward/Adam/EMA | H200 NVL, batch 512, width 256, eager | 40.75 → 35.55 ms | **1.146×** |
| GPU-resident training, warmed compilation | A100 80 GB PCIe, batch 512, width 256 | 38.38 → 29.31 ms | **1.310×** |
| Full Lightning training, including transforms, loader, weighted loss, metrics, AdamW and EMA | A100 80 GB PCIe, batch 512, 12 workers, eager | 192.35 → 189.30 ms | **1.016×** |
| Full Lightning training, smaller batch | RTX 4070, batch 128, 4 workers, eager | 142.62 → 140.42 ms | **1.016×** |
| Direct sampling, 16 NFE | H200 NVL, batch 128, width 256, eager | 217.55 → 75.64 ms | **2.876×** |
| Direct sampling, 256 NFE | H100 PCIe, batch 128, width 256, eager | 2209.71 → 1619.13 ms | **1.365×** |
| Direct sampling, 16 NFE | RTX 4070, batch 128, width 256, eager | 858.04 → 381.40 ms | **2.250×** |
| Direct sampling, deterministic algorithms enabled, 16 NFE | RTX 4070, batch 128, width 256, eager | 940.11 → 411.05 ms | **2.287×** |

Training-compute timings exclude the loader and Lightning. Full Lightning
timings exclude validation, checkpoints and external experiment logging, and
repeat 512 cached real QM9 structures. They measure the actual GPFF task and
transforms but are not complete training-epoch timings. Lightning figures use
total measured time divided by batches, rather than a median of alternating
fast/slow prefetch blocks. There were two independent processes per revision
in ABBA order for each Lightning comparison and the eager H200/RTX endpoints.
The eager endpoint figures pool their raw trials.

Peak allocated training memory also decreases: **11.42 → 10.33 GiB** on
the eager H200 benchmark and **4.89 → 3.78 GiB** on the compiled A100 benchmark.

The slower CPU and framework work in the full pipelines largely masks the
model-compute improvement. The measurements do not establish that every
existing training run will gain exactly 1.6%, or that its bottleneck is identical
to this cached-input benchmark.

**Each commit, compared with its parent.** Unless marked otherwise, component
figures below are paired measurements on an A100 80 GB PCIe with batch 128,
256 channels, 4 interactions, 600 RBFs and float32 matmul precision `high`.
Ratios in this report are the before median divided by the after median.
These operation-level multipliers must not be multiplied together to predict
whole-application speedups.

| Commit | Improvement and measured effect | Effect on GPU training / sampling |
|---|---|---|
| `ef74aa3` | Cache molecule layout: sigma estimation **0.538 → 0.142 ms, 3.77×** | Sampling improvement at every sigma-estimating step. No training-step effect. |
| `b24eda8` | Batch covariance/Cholesky: GPU **72.92 → 1.57 ms, 46.37×**; CPU **20.01 → 1.23 ms, 16.28×** | Large benefit to covariance/scaffold constraints; zero for default Gaussian/DD sampling. GPU precision caveat below. |
| `93cca41` | Vectorize fully connected neighbors: GPU-input initialization **55.97 → 0.758 ms, 73.84×** | Saves time once per sampling batch. Training's per-molecule neighbor transform is unchanged. CPU-input path also improves **17.73 → 0.965 ms, 18.37×**. |
| `4839f63` | Serialize batch keys once and omit model features: **141.98 → 8.71 ms, 16.31×** | Major sampling-output improvement; no model or training-step gain. |
| `24a2ce2` | Group GMM draws by molecule size: CPU **53.57 → 7.26 ms, 7.38×** | Helps shape-conditioned sampling initialization; default Gaussian prior gains nothing. Fitting on 512 structures: **391.22 → 361.99 ms, 1.08×** on this host. |
| `d464b76` | Multi-tensor EMA: **0.907 → 0.174 ms, 5.20×** | Accelerates the update, but its cost overlaps other queued GPU work: the RTX full-revision training step improves only **65.23 → 65.08 ms, about 0.23%**. No sampling effect. |
| `38cb9db` | Fold cutoff into filter multiplication and filter half the pairs: H200 training batch 512 **40.32 → 35.45 ms, 1.137×**; forward **16.02 → 13.88 ms, 1.154×** | Main eager GPU model improvement. A100 batch-128 training also improves **22.80 → 20.55 ms, 1.110×**. These gains include both changes in this commit. |
| `47efdc3` | Vectorize scaffold centering: CPU **7.24 → 0.635 ms, 11.41×**; GPU **19.64 → 0.935 ms, 21.00×** | Helps only scaffold-Gaussian prior initialization, once per sampling batch. No default-prior or training-step effect. |
| `3c78267` | Skip converged molecules: real trained width-512 adaptive sampling on RTX **1825.82 → 1051.45 ms, 1.736×** | Valuable when molecules converge at different steps. All-active prediction is about **2% slower** in the RTX control, so ordinary DD with `f_max=0` gets no convergence-skipping benefit. Fixed-seed outputs differ; see below. |
| `831b100` | Remove a clone and NaN host sync: isolated bookkeeping **0.0897 → 0.1025 ms, 0.875×** | **No measurable full-sampler gain**: RTX DD16 **377.64 → 377.55 ms, 1.000×**, with paired interval spanning no change. The new unconditional scatter adds work when no NaNs occur. |
| `a0e6c9b` | Cache the neighbor-pair symmetry check: repeated A100 forward **8.67 → 8.10 ms, 1.070×** | Useful for repeated sampling with the same neighbor tensors. Fresh-list forward and training remain **approximately 1.000×**; training changes tensor identities each batch. |
| `697f0f6` | Construct Atoms in one pass: **7.24 → 2.13 ms, 3.39×**; lighter GPU trajectory snapshot **1.081 → 0.407 ms, 2.65×** | Improves final conversion and, when enabled, per-step trajectory recording. No training-compute gain. |
| `3b17abc` | Skip identity atom-type alignment: 64 CPU training samples **7.00 → 5.27 ms, 1.33×** | Improves CPU preprocessing; no GPU-kernel change. The full training pipelines gain much less than this local ratio. |
| `ea093f3` | Keep half-pair filtering under compilation: A100 repeated forward **3.445 → 2.786 ms, 1.237×**; fresh-list training **9.706 → 9.480 ms, 1.024×** | Helps the compiled model. Fresh-list A100 inference is unchanged; RTX compiled training gains **1.153×**, repeated inference **1.430×**. The graph break/check overhead matters on faster hardware. |
| `428dc6a` | Expose generation's optional `torch.compile`: warmed RTX HEAD sampling **381.40 → 126.96 ms, 3.004×** | Strong warmed sampling benefit when explicitly enabled. **Zero default benefit**, because `compile: null` remains the default. Startup can make short jobs slower; see below. No new training optimization. |
| `632660a` | Overlap result writing/checking with the next batch: three-batch A100 eager pipeline **1627.43 → 1551.12 ms, 1.049×** | Genuine generation-pipeline gain. RTX eager test gains **1.155×**; compiled A100 40 GB test gains **1.062×**. Includes the last queue flush and uses real QM9 structures for validity checks. No training-step gain. |
| `8991e5f` | Single GPU→CPU copy: full-feature batch clone **9.33 → 2.17 ms, 4.294×** | Saves the final copy per denoising run. CPU-only clone has no convincing gain. Lighter trajectory snapshots have a smaller payload than this full-feature clone benchmark. |

The GMM fitting gain is hardware/workload dependent: the same 512-structure
fit was 2.28× faster on the RTX test host. Shape draws preserve the intended
distribution rather than the exact old random sequence. The branch's optional
paths have been timed explicitly; their improvements are not assumed to apply
to the default sampler.

The larger **width-512** model was also tested on H200 at batch 128:
`38cb9db` improves training **22.10 → 20.13 ms, 1.098×**, and inference
**8.32 → 7.64 ms, 1.089×**. `a0e6c9b` improves its repeated inference
**7.63 → 7.21 ms, 1.058×**, with no useful fresh-batch training gain.

**Compilation and startup.** The A100 training comparison compiles both
revisions through the model API, which training already supported at the
baseline. Compiling the baseline sampling model through that same API gives
293.95 ms versus 60.57 ms for HEAD's warmed DD16 sampler, but the baseline
generation CLI did not expose this option. This is an API comparison, not a
baseline CLI feature.

The direct HEAD eager/compiled sampling comparison on RTX is approximately
3.00× after warm-up. Its first compiled denoising call took **18.25 seconds**;
an earlier compiled half-pair forward in the same study took **32.68 seconds**.
Artifacts were reusable within a Slurm job, so these are observed first-call
latencies rather than guaranteed empty-cache compilation costs. At roughly
254 ms saved per warmed batch, 18.25 seconds requires about **72 batches,
or 9,200 molecules**, to amortize. A cold compile can require more. At this
16-NFE setting, 1,000-sample jobs therefore should not be assumed to benefit
overall. More denoising steps can make compilation easier to amortize.

**Numerical findings.** The covariance speedup is accompanied by a material
GPU precision difference at the default `matmul_precision=high`. On real QM9
positions the old/new transformed coordinates differed by up to **0.1066**
on A100 and **0.0921** on RTX. Against a float64 reference on RTX, the old
maximum error was **9.75e-5**, while the new error was **0.0921**. With
`highest`, the new error fell to approximately **1.0e-4**. On random Gaussian
positions the corresponding new errors were **1.50e-3** at `high` and
**5.96e-7** at `highest`. This supports reduced-precision batched matmuls as
the cause, particularly for ill-conditioned covariances. The commit is fast,
but should not be described as numerically equivalent on GPU at default precision.

The default GPU adaptive test is not bitwise repeatable. With skipping enabled
versus disabled, mean NFE was **17.0234 versus 17.0469** in the first trial and
the final maximum coordinate difference was **1.38**. However, the follow-up
also found **1.024** maximum coordinate difference and changed NFE when merely
repeating the control **without any skipping**. Consequently, the large
default-precision position difference cannot be attributed to this optimization
alone. Under **deterministic algorithms plus `matmul_precision=highest`**,
skip/no-skip NFE was identical (**17.0**), with maximum coordinate difference
only **4.77e-6**. The measured speedup was independently replicated at **1.744×**
and is much larger than the 0.14% variation in mean NFE. Sample-quality effects
were not measured.

The existing affected regression tests passed: **29 passed** on Hydra.
The only pytest warnings were attempts to create its cache in read-only
snapshots. CPU tests do not establish GPU equivalence under TF32.

**Evidence and reproduction.** `results/*.json` preserves all raw trials,
correctness diagnostics, GPU/software metadata and peak allocated memory.
`summary.csv` and `summary.json` recompute ratios and paired bootstrap intervals
from those trials. The complete chronological revision sweep is retained as
`increment-<sha>-5151132.json` files. The harness and exact workload definitions
are in `audit.py`, `pipeline.py`, `run.sbatch` and `README.md`.
The remote study is `hydra:~/gpff/scratch/speedup-audit-20261029/`.

Benchmarks use QM9, with fixed inputs, seeds and architecture. GPU clocks were
not pinned. The ratios are measured results for these workloads, not universal
constants or guarantees for GEOM-Drugs, other batch sizes, or full training runs.


**Chronological full-revision sweep.** This is the same RTX 4070 throughout,
with batch 128, width 256, eager execution, 16 direct-denoising steps,
Gaussian initialization, no force convergence threshold, and relaxed-result
conversion. Every row uses the entire historical revision in a fresh process.
It excludes loader/Lightning overhead from training and excludes database
writing/checking from sampling. Therefore GMM/scaffold, CPU training alignment,
compilation and deferred-writing benefits are intentionally inactive here.
Small differences between adjacent process timings are not causal evidence;
the paired tests above provide the attribution.

| Revision | Training step, ms | Sampling batch, ms | Sampling speedup vs previous revision |
|---|---:|---:|---:|
| `759d3d6` | 65.23 | 858.36 | base |
| `ef74aa3` | 65.33 | 844.34 | 1.017× |
| `b24eda8` | 65.34 | 829.64 | 1.018× |
| `93cca41` | 65.23 | 722.89 | 1.148× |
| `4839f63` | 65.24 | 481.72 | 1.501× |
| `24a2ce2` | 65.23 | 481.50 | 1.000× |
| `d464b76` | 65.08 | 480.84 | 1.001× |
| `38cb9db` | 56.30 | 420.26 | 1.144× |
| `47efdc3` | 56.30 | 421.36 | 0.997× |
| `3c78267` | 56.30 | 421.89 | 0.999× |
| `831b100` | 56.30 | 421.59 | 1.001× |
| `a0e6c9b` | 56.30 | 409.96 | 1.028× |
| `697f0f6` | 56.28 | 397.22 | 1.032× |
| `3b17abc` | 56.30 | 398.43 | 0.997× |
| `ea093f3` | 56.29 | 397.13 | 1.003× |
| `428dc6a` | 56.29 | 397.49 | 0.999× |
| `632660a` | 56.29 | 398.11 | 0.998× |
| `8991e5f` | 56.33 | 382.28 | 1.041× |
