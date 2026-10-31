# Post-hoc EMA for GPFF

This implements Section 3 and Appendix C of Karras et al., *Analyzing and
Improving the Training Dynamics of Diffusion Models*
([paper](https://arxiv.org/abs/2312.02696)). The local paper source is
`papers/edm2/{paper,supplemental-body,supplemental-figures}.tex` in the main
checkout. The equations are implemented independently, with no new dependency.

## Method and relationship to EDM2

For training weights θ(t), the normalized averaging response is

```
p(tau; t, gamma) = (gamma + 1) / t * (tau / t)^gamma,  0 <= tau <= t
sigma_rel = sqrt((gamma + 1) / ((gamma + 2)^2 * (gamma + 3)))
```

The initialization has zero weight. The absolute averaging width grows with
training time, while `sigma_rel` stays fixed. A width of 0.10 means a standard
deviation of 10% of elapsed training time; it is neither a last-10% window nor
a decay of 0.90. It is unrelated to GPFF's VE noise sigma.

After update t, `ema = beta * ema + (1 - beta) * theta`, where
`beta = (1 - 1/t)^(gamma + 1)`. The first update has beta=0. Solving the paper's
cubic converts sigma_rel to gamma. Defaults 0.05 and 0.10 give approximately
16.97 and 6.94, respectively.

Each stored average has a profile p_i. Reconstruction solves `A x = b`, with
`A_ij = integral(p_i p_j)` and `b_i = integral(p_i p_target)`, using analytic
inner products. The model is `sum_i x_i * theta_i`. Coefficients can be negative;
clipping would change the least-squares fit. Their sum is normalized to one as
in the [official implementation](https://github.com/NVlabs/edm2/blob/main/training/phema.py);
the printed Algorithm 3 omits this normalization. The solve and parameter
accumulation use float64; exported models use float32.

This contribution is independent of EDM2's magnitude-preserving U-Net, loss
weighting, noise conditioning, and sampler. GPFF can use it with its existing
time-agnostic force predictor and direct denoising. Weight averaging preserves
the architecture's equivariance. It is not an average of nonlinear models'
predictions and does not guarantee better molecular generation.

## Use on hydra

Run inside the usual container in a Slurm allocation, with the frozen code's
`src` on `APPTAINERENV_PYTHONPATH`. The module command needs no reinstallation.

```bash
# Practical GPFF default: paper profiles and cadence, float32 storage.
gpff_train experiment=gpff +callbacks=posthoc_ema

# Closest storage configuration to EDM2: float16 every 4096 optimizer updates.
gpff_train experiment=gpff +callbacks=posthoc_ema \
  callbacks.posthoc_ema.snapshot_dtype=float16

python -m gpff.cli.posthoc_ema \
  --input /path/to/run/posthoc_ema --sigma-rel 0.15 \
  --output /path/to/new/ema_015_model

# Earlier endpoint; snapshots after step 8192 are excluded.
python -m gpff.cli.posthoc_ema \
  --input /path/to/run/posthoc_ema --step 8192 --sigma-rel 0.05 \
  --output /path/to/new/step8192_ema005_model

gpff_generate model_path=/path/to/new/ema_015_model \
  split_path=/path/to/run/split.npz datapath=/path/to/matching/qm9_filtered.db
```

The callback observes raw weights alongside the existing conventional 0.995
EMA. Validation, best-model selection, scheduling, and milestones keep their
existing behavior. This reproduces the paper's averaging/synthesis, not its
entire training protocol.

The clock counts optimizer updates, excluding accumulation microbatches and
skipped updates. It equals EDM2's image clock up to a constant for a fixed
effective batch size. A batch-size change retains an update-based interpretation
rather than a molecule-based one. Snapshot filenames use this clock; Lightning's
global step and zero-based epoch are also recorded.

Each snapshot stores both parameter averages, current buffers, widths, clock,
and run identity. One `model.pt` stores the pickled architecture and
postprocessors. Buffers are copied from the requested endpoint, never averaged,
matching EDM2. Only rank zero writes files. Supported training uses one optimizer
and full replicated parameters (single device / DDP), not sharded training.
Keep the usual float32 training weights; float16 here is a storage option.

Lightning checkpoints include full precision EMA state and clock. Resume using
`hy resume` with the same callback config. Old conventional-EMA checkpoints
cannot initialize these profiles. Rewinding behind snapshots already saved
requires a new directory to avoid mixing divergent continuations. To relocate
a continuation, retain its earlier snapshots and `model.pt`/`run_id` for the
full reconstruction history. Training can resume from callback state alone,
but missing historical snapshots weaken the reconstruction. Normal training end
also saves the final snapshot.

Reconstructed outputs are inference models, not optimizer-resume checkpoints.
The adjacent JSON records source paths, widths, signed coefficients, run identity,
endpoint, and relative L2 profile error. That error measures agreement of the
averaging kernels, not molecular quality. Unsupported endpoints, mixed runs,
incompatible tensor layouts, and duplicate source profiles are rejected. As
elsewhere in GPFF, only load trusted pickled models.

## Cadence and evaluation

EDM2 saved pairs every 4096 updates and used 160–512 source averages per run.
Its accurate reconstruction over widths 0.015–0.25 depends on that history;
it is not guaranteed for short GPFF runs. Set
`callbacks.posthoc_ema.every_n_steps=<N>` to collect enough snapshots. Narrow
target widths need frequent recent snapshots. Inspect the JSON profile error.
Two online averages cost two extra parameter copies; each snapshot costs two
parameter copies plus buffers on disk, in addition to the one-time template.

Reconstruct a coarse grid (e.g. 0.015, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20,
0.25), then refine near the best validation result. Compare against conventional
EMA using the same dataset/split, molecule inputs, initial noise, and sampling
budget. Select on validation data, then report the chosen model on test data.
Measure molecular validity and structural quality, not just denoising loss.
The paper shows that architecture, training duration, guidance, and the quality
metric affect the preferred width; GPFF's sampler and NFE should therefore be
fixed or tuned jointly with EMA.

## GPFF extensions to consider (not implemented)

* **Epoch-based saving:** retain the same math, save at epoch intervals, and
  record the exact optimizer clock. This aligns with milestone studies;
  reconstruction already supports irregular source times.
* **Separate representation/head widths:** use the same snapshots to tune
  PaiNN and the force head separately. The paper's per-tensor experiment
  motivates this for heterogeneous architectures, but its stabilized model
  preferred a shared optimum. Start with a global width to limit overfitting.
* **Molecule-count clock:** use actual global molecules processed per update
  and `beta = (1 - delta_t/t)^(gamma+1)` for variable molecular batch sizes.
  An atom-count clock instead weights larger molecules more and needs separate
  justification.
* **Existing conventional-EMA milestones:** the paper allows other source
  profiles, but these require the actual torch-ema warmup/decay and update
  history, and a different Gram matrix. Sparse conventional-EMA milestones
  cannot be treated as power-profile snapshots. This implementation does not
  make that approximation.

## Validation

Validated on hydra, in the py313-torch214-gcc container under CPU Slurm jobs
on 2026-10-30, using frozen snapshot `138cc47-posthoc-ema-v2`:

* Job 5159714: **25 passed** across `tests/test_posthoc_ema.py` and
  `tests/test_callbacks.py`. Checks include numerical integration, direct
  averaging of a complete weight history, exact source-profile reconstruction,
  bit-identical resume, conventional-EMA coexistence, accumulation and skipped
  batches, float16 storage, Hydra composition, and real PaiNN export/reload
  through schnetpack (also with `model.compile(backend="eager")`).
* Job 5159727: the actual `gpff_train` entry point completed two tiny QM9
  epochs (four optimizer updates), validation/test, milestone saves, and
  CLI reconstruction at width 0.15. The isolated run is
  `~/gpff/scratch/posthoc-ema-validation/smoke-v2` on hydra. Its profile error
  is about 0.106: four updates exercise the plumbing, not reconstruction
  accuracy at realistic training lengths.

These checks validate implementation behavior; a molecular-quality EMA sweep
has not been run.
