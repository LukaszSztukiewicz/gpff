# Native PaiNN stabilization

These are architectural hypotheses for GPFF, with independent switches for
ablation. Better molecular generation quality requires matched training and
generation experiments. Existing `experiment=gpff` behavior and old exported
models remain supported. The candidate `gpff_stable` recipe uses constant LR,
no warmup, no gradient clipping, no force saturation, and no skipping of exploding
batches. Optimizer settings and the training distribution/objective are unchanged.

## What transfers from EDM and EDM2

Read: [EDM, Karras et al., NeurIPS 2022](https://arxiv.org/abs/2206.00364),
especially section 5 and equations 7–8, and its direct successor
[EDM2, Karras et al., CVPR 2024](https://arxiv.org/abs/2312.02696),
especially sections 2.2–2.5 and 3. The EDM2 TeX source was also read on hydra
because the browser could not load its large PDF.

EDM separates the raw network, denoiser parameterization, loss weights, and
training noise distribution. EDM2 addresses drifting activation **and weight**
magnitudes, and therefore drifting effective update sizes. It also studies EMA.
Its complete recipe includes explicit LR decay; this implementation investigates
the architectural mechanisms at constant LR.

GPFF predicts `F=2(x0-x)` without sigma conditioning. Blindly adding EDM's
sigma-dependent input/output coefficients would change that inference contract.
Likewise, EDM2's learned sigma-dependent loss weights would change the optimal
time-agnostic predictor: different noise levels compete for one function of x.
Neither is included. The checked-in GPFF sigma weights are capped at 1; rare
arbitrarily large `1/sigma²` weights are not the default model's root problem.

## Implemented mechanisms

| Option | Mechanism addressed | Design |
| --- | --- | --- |
| `globals.weight_norm=true` | Linear maps, affine offsets, and relative update sizes drift | Normalize each augmented affine row `[W,b]` when used; keep its stored RMS at one after optimizer steps. Compensate SiLU's Gaussian RMS attenuation. Covers trunk, radial filters, and readout. |
| `globals.normalized_products=true` | Vector magnitudes feed back quadratically/cubically through interactions, mixing, and readout gates | Use smooth direction coordinates in tensor-product operands. The mixing MLP still receives the raw invariant magnitude. |
| `globals.radial_norm=true` | Gaussian basis inputs have sparse active energy, inconsistent with nominal dense fan-in | Normalize their RMS before applying the cutoff, preserving the cutoff envelope. |
| `globals.residual_scaling=true` | Every extra block adds a full-size update to the residual stream | Allocate a variance budget across all `2L` updates and use a variance-normalized residual sum. |
| `globals.prenorm=rms` | Mean subtraction can discard scalar information; learned norm gains can restore drift | Mean-preserving, gain-free, per-atom soft RMS for scalars and vectors. Independent of batch size. |
| `globals.degree_norm=mean` | `sqrt(degree)` assumes uncorrelated messages, although same-element neighbors can add coherently | Optional mean aggregation. Constant messages no longer grow with coordination. Ablate separately: random messages may become too small, and count information is reduced. |
| `globals.zero_net_force=true` | Capacity is spent on a translation mode absent from centered-noise targets | Orthogonally project each molecule's force prediction onto the zero-sum subspace. |
| `globals.output_gain=1.0` | Normalized internal maps should not determine physical output units | A learned scalar readout gain, initialized to one. No output saturation. |

`weight_norm` normalizes weights and biases jointly: otherwise a bias could bypass
the normalized map. Effective rows have L2 norm one; raw rows have RMS one so
adding channels does not automatically shrink the relative size of an Adam update.
This is **not** a proof of width-independent functional updates or a spectral-norm
bound. No parameters change during forward/backward. Projection occurs once per
optimizer step in `AtomisticTask`, after Adam and before EMA sees the model.
A custom training loop must call `project_weights(model)` after `optimizer.step()`.
Whole-model pickle export remains supported; PyTorch parametrizations were avoided
because they do not support this export path.

The problematic mixing term is `gate(q, |V|) * (V dot W)`. Scaling vectors by `a`
can scale the gate by `a` and the dot product by `a²`. With
`h(v)=v/sqrt(1+|v|²)`, the new terms are:

```
scalar update = gate(q, |V|) * (h(V) dot h(W))
vector update = gate(q, |V|) * h(W)
```

The contraction is a correlation-like invariant, removing quadratic magnitude
amplification. This changes the tensor-product parameterization, not a computed
gradient or force. `h` is equivariant, smooth with identity Jacobian at zero,
and retains magnitude information through a strictly monotone radial coordinate.
It avoids the singular derivative of `v/|v|`. The uncompressed invariant magnitude
also remains available to the mixing MLP. This is a GPFF-specific proposal, not
an EDM2 result.

For depth, with `a=1/sqrt(2L)`, each sub-block uses
`h_next=(h+a*delta_h)/sqrt(1+a*a)`. This controls expected accumulation for
comparable, uncorrelated branches; it does not bound correlated growth or guarantee
a stable Jacobian at arbitrary LR. The product and affine changes address separate
amplification paths.

When targets sum to zero and weights are constant across atoms of a molecule,
subtracting the prediction mean cannot increase its squared error. This applies
to the default mean-free VE noise and sigma weighting. It is not a zero-torque
constraint; disable it for non-centered noise or unrelated target fields.
Combining it with the older bounded head changes that head's per-atom bound.

## Use and ablate

Snapshot the uncommitted worktree as usual. On hydra, from an empty working
directory with the snapshot in `APPTAINERENV_PYTHONPATH`, inspect the configuration:

```sh
apptainer exec "$SIF" gpff_train experiment=gpff_stable --cfg job --resolve
```

The candidate starts at LR `1e-3`, batch 128, width 256, depth 4. For an 8-block,
512-channel model, override `globals.n_interactions=8 globals.n_atom_basis=512`.
These settings are not proven at full training duration. The site's `train.sbatch`
explicitly sets batch 512: pass `data.batch_size=128` again when using that launcher,
as well as `experiment=gpff_stable`.
Start fresh runs when changing parameterization flags; resume each run with its
original flags and frozen snapshot. Matching tensor shapes do not make optimizer
states interchangeable between normalized and original parameterizations.

Suggested ablation order, with identical split, noise seeds, fixed validation
noise, constant LR, and no skipped steps:

1. Existing prenorm + degree normalization, without a bounded force head.
2. Add affine weight normalization alone.
3. Add radial normalization, then depth scaling.
4. Add normalized products; switch `prenorm=true` to `rms`.
5. Compare `degree_norm=true` with `degree_norm=mean`; hold zero-force projection
   identical across the comparison.

Use `globals.<option>=false` for ablations. For trunk-only/head-only product tests,
override `model.representation.normalized_products` or
`model.output_modules.0.normalized_products` separately.

Sweep batch 32/128, width 256/512, depth 4/8/12, and LR `3e-4/1e-3/3e-3` in stages,
not a full grid initially. Compare both equal examples seen and equal wall time.
Log force loss by noise range, raw/EMA validation losses, parameter and activation
RMS by depth, gradient norms, finite-step fraction, throughput, and memory.
Then evaluate generation on identical initial noise with short and long direct
denoising budgets. Molecular validity, stability, diversity, and distribution
metrics decide success; low training loss alone does not.

EMA already exists. For fair comparisons across batch sizes, match its sample
horizon using `decay_new=decay_old**(effective_batch_new/effective_batch_old)` or
report the discrepancy. No EMA modification is part of this implementation.
EDM2 post-hoc EMA remains a separate quality study rather than a repair for the
model's internal dynamics.

## Validation

All Python, tests, and training checks run on hydra through Slurm and the existing
container. The study folder `~/gpff/scratch/edm-stabilization-20261030` contains job
scripts and the paired QM9 smoke driver. Measured results follow. A short stress
check cannot establish generalization or generation quality.

- Snapshot `138cc47-edm-native-20261030-v2`, CPU job **5159712**: full suite,
  **82 passed, 17 CUDA cases skipped**.
- Same snapshot, GPU job **5159713**: full suite, **99 passed**, including real
  CUDA/Inductor forward, backward, and projected optimizer updates.
- Final code snapshot `138cc47-edm-native-20261030-v3`, CPU job **5159762**:
  **82 passed, 17 CUDA cases skipped**, after retaining the original `rsqrt`
  operation for the existing sqrt-degree option. The snapshot is ready for fresh
  ablations; the worktree remains uncommitted.
- Earlier GPU job 5159675 had one float32 gradient-comparison mismatch in the
  existing unnormalized PaiNN test (one element, absolute error `1.62e-5`). The
  subsequent full suite passed with unchanged tolerances; this is recorded rather
  than silently weakening that test.

### Paired QM9 stress check: job 5159698

Depth 8, batch 8, constant LR **0.003**, 50 AdamW updates, one initialization seed
(999), paired precomputed training noise/batches (seed 42), 128 training molecules,
and 32 distinct fixed-noise validation molecules. These are subsets of the existing
`b36d9cd218f5998d` split (98,855/12,356/12,358); no dataset was rebuilt or modified.
Initial validation losses were all approximately 1.588. The common metric is the
existing capped-sigma-weighted force MSE. No clipping, scheduler, warmup, bounded
head, skipped steps, or EMA was used. All variants completed all 50 updates.

| Architecture | Width | Final validation loss | Maximum training loss |
| --- | ---: | ---: | ---: |
| Existing prenorm + sqrt-degree | 256 | 1.3140 | 338.75 |
| Native recipe | 256 | **0.9756** | 2.074 |
| Native, mean aggregation | 256 | 1.0395 | 2.198 |
| Native, original tensor products | 256 | 1.0468 | 2.049 |
| Existing prenorm + sqrt-degree | 512 | 108.0352 | 321,595,424 |
| Native recipe | 512 | **0.8698** | 1.989 |
| Native, mean aggregation | 512 | 1.0186 | 2.145 |
| Native, original tensor products | 512 | 1.0555 | 2.669 |

This supports retaining normalized products and sqrt-degree in the initial recipe;
mean aggregation is an optional hypothesis for more correlated/high-degree graphs,
not the QM9 recommendation. The native-vs-control comparison changes several
components, so it cannot assign the whole gain to any single one. Gradient norms
are recorded in the raw log but are not comparable across parameterizations.

Eager training was about **50% slower per step** here (roughly 3.6–3.7 s vs 2.4–2.5 s
for 50 steps, including final validation). Peak allocated GPU memory was
0.66 vs 0.58 GB at width 256 and 1.52 vs 1.31 GB at width 512. CUDA compilation
correctness was tested separately; compiled throughput was not benchmarked.
This very small, single-seed pilot establishes neither full-data convergence nor
generalization/generation quality. The encouraging result is learning without the
control's large spikes at the same high LR and small batch.

Raw metrics: `~/gpff/logs/gpff-edm-native-smoke-5159698.out`.
Reproducer: `~/gpff/scratch/edm-stabilization-20261030/smoke.py` and `smoke.sbatch`.
