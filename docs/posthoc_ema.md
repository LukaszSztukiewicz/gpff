# Post-hoc EMA for GPFF

This implements Section 3 and Appendix C of Karras et al., *Analyzing and
Improving the Training Dynamics of Diffusion Models*
([paper](https://arxiv.org/abs/2312.02696)). The local paper source is
`papers/edm2/{paper,supplemental-body,supplemental-figures}.tex` in the main
checkout. The equations are implemented independently, with no new dependency.

## Method and relationship to EDM2

For training weights $\theta(t)$, the normalized averaging response is

$$p(\tau; t, \gamma) = \frac{\gamma + 1}{t} \left(\frac{\tau}{t}\right)^{\gamma}, \quad 0 \le \tau \le t, \qquad \sigma_\text{rel} = \sqrt{\frac{\gamma + 1}{(\gamma + 2)^2 (\gamma + 3)}}$$

The initialization has zero weight. The absolute averaging width grows with
training time, while `sigma_rel` stays fixed. A width of 0.10 means a standard
deviation of 10% of elapsed training time; it is neither a last-10% window nor
a decay of 0.90. It is unrelated to GPFF's VE noise sigma.

After update $t$, $\hat\theta \leftarrow \beta \hat\theta + (1 - \beta) \theta$, where
$\beta = (1 - 1/t)^{\gamma + 1}$. The first update has $\beta = 0$. Solving the paper's
cubic converts sigma_rel to $\gamma$. Defaults 0.05 and 0.10 give approximately
16.97 and 6.94, respectively.

The two running averages are kept in float64. Late in training an update moves
them by $1 - \beta \approx (\gamma + 1)/t$ of their difference to the weights:
about $2 \times 10^{-6}$ at QM9 epoch 5000 (3.9M updates). That is below
float32 resolution for differences under a few percent of a weight, so float32
averages would stop following the weights. Snapshots are stored in float32.

Each stored average has a profile $p_i$. Reconstruction solves $Ax = b$, with
$A_{ij} = \int p_i p_j \, d\tau$ and $b_i = \int p_i p_\text{target} \, d\tau$, using analytic
inner products. The model is $\sum_i x_i \hat\theta_i$. Coefficients can be negative;
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
The milestones and the post-hoc EMA go in one override, since Hydra rejects two
`+callbacks=` options; with `hy train`, pass `'+callbacks=[milestones,posthoc_ema]'`
in place of `+callbacks=milestones`.

```bash
# Default: paper profiles, a snapshot every 16384 optimizer updates and at the
# end of each milestone epoch, float32 storage.
gpff_train experiment=gpff '+callbacks=[milestones,posthoc_ema]'

# Closest storage configuration to EDM2: float16 every 4096 optimizer updates.
gpff_train experiment=gpff '+callbacks=[milestones,posthoc_ema]' \
  callbacks.posthoc_ema.every_n_steps=4096 callbacks.posthoc_ema.snapshot_dtype=float16

# Width 0.15 at the latest snapshot.
python -m gpff.cli.posthoc_ema \
  --input /path/to/run/posthoc_ema --sigma-rel 0.15 \
  --output /path/to/new/ema_015_model

# Width 0.05 at the end of epoch 2000, comparable with milestones/epoch_2000_model.
python -m gpff.cli.posthoc_ema \
  --input /path/to/run/posthoc_ema --epoch 2000 --sigma-rel 0.05 \
  --output /path/to/new/epoch2000_ema005_model

# Any saved optimizer update; snapshots after update 16384 are excluded.
python -m gpff.cli.posthoc_ema \
  --input /path/to/run/posthoc_ema --step 16384 --sigma-rel 0.05 \
  --output /path/to/new/step16384_ema005_model

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
rather than a molecule-based one. Snapshots are saved every `every_n_steps`
updates and at the end of each completed epoch in `epochs`, by default the
milestone epochs when the milestones callback is on. Snapshot filenames use
the update clock; Lightning's global step and zero-based epoch, and for an
epoch-end snapshot the completed epoch (`epoch_end`), are also recorded.
Skipped batches shift the clock, so `--epoch` finds a snapshot by its recorded
epoch rather than computing its step.

Each snapshot stores both parameter averages, current buffers, widths, clock,
and run identity. One `model.pt` stores the pickled architecture and
postprocessors. Buffers are copied from the requested endpoint, never averaged,
matching EDM2. Only rank zero writes files. Supported training uses one optimizer
and full replicated parameters (single device / DDP), not sharded training.
Keep the usual float32 training weights; float16 here is a storage option.

Lightning checkpoints include the float64 averages and clock. Resume using
`hy resume` with the same callback config. Old conventional-EMA checkpoints
cannot initialize these profiles. A resume continues from its checkpoint's
clock, which is often earlier than the last snapshot: `hy resume`'s `latest` is
the newest milestone or best checkpoint, not `last.ckpt`. Snapshots after that
clock averaged the abandoned continuation, so the resumed run moves them to
`posthoc_ema/abandoned/after_<update>_<id>/` and logs a warning. Nothing is
deleted, and reconstruction reads only the top-level snapshots. A new run
refuses a directory holding another run's snapshots. Training can resume from
callback state alone, but missing historical snapshots weaken the
reconstruction. Normal training end also saves the final snapshot.

Reconstructed outputs are inference models, not optimizer-resume checkpoints.
The adjacent JSON records source paths, widths, signed coefficients, run identity,
endpoint update and epoch, and relative L2 profile error. That error measures
agreement of the averaging kernels, not molecular quality. Unsupported endpoints,
mixed runs, incompatible tensor layouts, and duplicate source profiles are
rejected. As elsewhere in GPFF, only load trusted pickled models.

## Cadence and evaluation

EDM2 saved pairs every 4096 updates and used 160–512 source averages per run.
GPFF's default is every 16384 updates, about 21 epochs on QM9 (773 updates per
epoch at batch size 128). For the default QM9 model (5.06M parameters, 38.6 MiB
per float32 snapshot) the relative profile error at epoch 5000 is at most 0.3%
for widths 0.015–0.25, from 236 snapshots (8.9 GiB) instead of 944 (36 GiB) at
4096. Earlier endpoints have fewer snapshots: at epoch 1000 the error is 3.2%
for width 0.015 and 2.1% for 0.25 (0.2% and 0.4% at 4096); at epoch 100, widths
below 0.05 are not reconstructed accurately at either cadence (14% for 0.015 at
4096). Lower `callbacks.posthoc_ema.every_n_steps` for short runs or for narrow
widths at early endpoints, and inspect the JSON profile error. Narrow widths
also have large signed coefficients (absolute sum about 100 for 0.015), which
amplify float16 storage rounding; keep float32 storage for them.
Two float64 online averages occupy four float32 parameter copies on the GPU
and in each Lightning checkpoint (81 MB for the default QM9 model).

Reconstruct a coarse grid (e.g. 0.015, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20,
0.25), then refine near the best validation result. Compare against conventional
EMA at the same milestone epochs (`--epoch`), using the same dataset/split,
molecule inputs, initial noise, and sampling budget. Select on validation data,
then report the chosen model on test data.
Measure molecular validity and structural quality, not just denoising loss.
The paper shows that architecture, training duration, guidance, and the quality
metric affect the preferred width; GPFF's sampler and NFE should therefore be
fixed or tuned jointly with EMA.

## GPFF extensions to consider (not implemented)

* **Separate representation/head widths:** use the same snapshots to tune
  PaiNN and the force head separately. The paper's per-tensor experiment
  motivates this for heterogeneous architectures, but its stabilized model
  preferred a shared optimum. Start with a global width to limit overfitting.
* **Molecule-count clock:** use actual global molecules processed per update
  and $\beta = (1 - \Delta t/t)^{\gamma + 1}$ for variable molecular batch sizes.
  An atom-count clock instead weights larger molecules more and needs separate
  justification.
* **Existing conventional-EMA milestones:** the paper allows other source
  profiles, but these require the actual torch-ema warmup/decay and update
  history, and a different Gram matrix. Sparse conventional-EMA milestones
  cannot be treated as power-profile snapshots. This implementation does not
  make that approximation.

## Validation

Validated on hydra, in the py313-torch214-gcc container under CPU Slurm jobs
on 2026-11-08, using frozen snapshot `01018b1-posthoc-ema` (on dev-wip
`fe0e489`):

* Job 5195209: **39 passed** across `tests/test_posthoc_ema.py`,
  `tests/test_callbacks.py`, `tests/test_configs.py` and
  `tests/test_checkpoint.py`. Checks include numerical integration, direct
  averaging of a complete weight history, exact source-profile reconstruction,
  bit-identical resume, a resume from an earlier checkpoint (later snapshots set
  aside, same reconstruction as an uninterrupted run), epoch-end snapshots and
  `--epoch`, float64 averages that keep moving at 3M updates,
  conventional-EMA coexistence, accumulation and skipped batches, float16
  storage, Hydra composition with the milestone epochs, and real PaiNN
  export/reload through schnetpack (also with `model.compile(backend="eager")`).
* Job 5195210: the actual `gpff_train` entry point with
  `'+callbacks=[milestones,posthoc_ema]'`, milestones at epochs 1 and 2 and a
  snapshot every 3 updates, trained three tiny QM9 epochs (6 updates). It then
  resumed from the epoch-1 milestone to four epochs, which moved the 3 later
  snapshots to `abandoned/`. CLI reconstruction at `--epoch 2` gave update 4
  both before and after the resume. The run is
  `~/gpff/scratch/posthoc-ema-review/smoke.UdIIbZ/smoke` on hydra. Its profile
  errors (0.10–0.15 at width 0.15) only reflect the few updates.

Numerical checks of the float32/float64 averages and of reconstruction versus
cadence are in `~/gpff/scratch/posthoc-ema-review/check.py`.

These checks validate implementation behavior; a molecular-quality EMA sweep
has not been run.
