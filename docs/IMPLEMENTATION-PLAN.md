# GPFF — Remediation and Implementation Plan

Derived from a source audit of the initial public release (`04252cb`). It
proposes concrete patches, in dependency order, for defects that
[`SPEC.md`](SPEC.md) does not already record.

## Relationship to SPEC.md

`SPEC.md` §11 is an accurate and unusually complete inventory of the codebase's
known gaps. This plan does **not** restate it. Three categories:

* **New here.** Defects `SPEC.md` §11 does not mention. These are the substance
  of the plan (Workstreams A–D, F).
* **Already in §11.** Scheduled below with an ID and an owner, but the analysis
  is §11's, not this document's. Marked *(SPEC §11.x)*.
* **Documented as deliberate.** Two items an earlier read of this audit
  mis-filed as bugs, which §2.2 and §5.1 explicitly justify:
  * `norm_epsilon = 1.0` in [`painn.py`](../src/gpff/model/painn.py) — §5.1
    states this is *not* a stability epsilon but an intentional soft direction
    vector. **Retracted as a defect.** It survives here only as an ablation
    (E3), because its effect at bonding distances is large (`‖dir_ij‖ = d/(d+1)`
    ≈ 0.5–0.6 at 1–2 Å) and no released config exposes the knob.
  * `SigmaWeights` clipped to `[0, 1]` — §2.2 states the asymmetry is
    deliberate ("`σ ≤ 1` → plain force matching"). **Retracted as a defect.**
    It survives as E1, because the audit quantified the consequence and the
    number is larger than the prose suggests.

## Taxonomy

Every patch carries a **blast radius**, which determines what has to be re-run:

| tag | meaning |
|---|---|
| `SAFE` | No change to any released numerical path. Ship freely. |
| `CONFIG` | Changes defaults or config resolution; no checkpoint impact. |
| `SAMPLE` | Changes sampled outputs. Re-run generation + evaluation. |
| `TRAIN` | Changes the learned function. Requires retraining and an A/B. |
| `EVAL` | Changes reported metric values. Re-run phase 2, and say so in any comparison table. |

---

## Sequencing

```
A (invariant violations)  ──┐
B (broken interfaces)     ──┼──► T (tests)  ──► E (measurement / ablations)
C (reproducibility)       ──┤
D (evaluation integrity)  ──┘
F (latent + hygiene)  ─ anytime
G (performance)       ─ after T
```

A, B, C and D are independent and can land in parallel. **T should land with
A** — there is currently no `tests/` directory, and A2/A3 are exactly the class
of defect a three-line regression test would have caught. E depends on T
because its whole purpose is to produce trustworthy A/B numbers.

| ID | Title | Radius | Effort |
|---|---|---|---|
| A1 | `f_max` corrupts the molecules it converges | `SAMPLE` | S |
| A2 | Network activations leak into the batch and get serialized | `SAMPLE`* | S |
| A3 | Adaptive samplers write positions outside `update_batch` | `SAMPLE` | S |
| A4 | Missing `σ` floors in the adaptive samplers | `SAMPLE` | S |
| A5 | Stale `t` in `AdaptiveStochasticHeun`; absent in `AdaptiveAncestral` | `SAMPLE` | S |
| B1 | Diffusion baselines cannot be sampled | `CONFIG` | M |
| B2 | Time-aware checkpoints fail with a bare `KeyError` | `CONFIG` | S |
| B3 | `wandb_upload_id=None` cannot disable uploads | `SAFE` | S |
| B4 | `gpff_generate` rebuilds QM9 on a mistyped path | `SAFE` | S |
| C1 | Resume is defeated by the run-directory layout | `CONFIG` | S |
| C2 | The training seed is never persisted | `CONFIG` | S |
| C3 | `val/loss` is logged without `batch_size` | `SAFE` | S |
| C4 | `predict_without_postprocessing` is not exception-safe | `SAFE` | S |
| D1 | Reference distributions are trusted, not recomputed | `EVAL` | M |
| D2 | Novelty depends on three artifacts staying index-aligned | `EVAL` | M |
| D3 | The RDKit timeout cannot interrupt the call it guards | `EVAL` | M |
| E1 | Quantify and ablate the `SigmaWeights` clip | `TRAIN` | L |
| E2 | Determine whether the t-aware baselines use `t` at all | `TRAIN` | M |
| E3 | Ablate `norm_epsilon` | `TRAIN` | M |
| E4 | Implement `mean_free_noise` *(SPEC §11.2)* | `TRAIN` | S |
| F1–F9 | Latent defects and hygiene | mixed | S |
| G1 | Filter-net cost | `TRAIN` | M |

\* A2 changes stored trajectory metadata and memory profile, not sampled geometries.

---

## Workstream A — invariant violations and silent corruption

`SPEC.md` §13 invariant 6 reads: *"Position updates go through `update_batch`,
so converged molecules stay frozen."* A1–A3 are three places where that
invariant is violated. A1 is the one that causes real damage.

### A1 — `f_max` corrupts the molecules it converges  `SAMPLE`

**Files:** [`sample/direct_denoising.py`](../src/gpff/sample/direct_denoising.py)

`denoise_step` runs `preprocess_step` first, and `preprocess_step` writes
positions unconditionally — by rebinding in the base class (line 74) and
**in place** in the scaffold subclasses (line 225, 280):

```python
batch[properties.R] = self.add_noise(batch[properties.R], step)   # no mask
```

`update_batch` then preserves converged molecules by reading
`old_value = batch[key].clone()` — but `batch[R]` is now the *perturbed*
positions. A molecule that converged at an earlier step therefore does not
freeze: it keeps receiving `λ_k·ξ` every remaining step, with no model call to
undo it (`nfe` correctly stops incrementing, which is what hides it).

Magnitude, for convergence at step 25 of 50 with `stochastic_lambda = 1.0`
(`λ_j = (j−1)/n_steps`):

```
Σλ² = (1/2500)·Σ_{i=0}^{23} i² = 4324/2500 = 1.73
  →  1.32 Å per coordinate, ≈2.3 Å in 3D, applied to every atom
```

That is more than enough to destroy RDKit validity. The feature ships disabled
(`f_max: 0.0` in all four DD configs), which is consistent with it never having
worked — and it means the `avg nfe` reported by
[`cli/generate.py:174`](../src/gpff/cli/generate.py) is currently always exactly
`n_steps`. Any adaptive-NFE claim depends on this patch.

**Patch.** Snapshot the pre-perturbation positions and restore them as the
masking baseline:

```python
def denoise_step(self, batch, step, conditions=None, **kwargs):
    # preprocess_step writes batch[R] unconditionally (and in place, in the
    # scaffold subclasses), so update_batch would otherwise read the perturbed
    # positions as the "previous" value for converged molecules. SPEC §13.6.
    x_prev = batch[properties.R].clone()

    batch = self.preprocess_step(batch, step, conditions=conditions, **kwargs)

    predictions = self.calculator.calculate(batch, forces=True, x0=True, estimate_sigma=True)
    ...
    batch[properties.R] = x_prev          # restore the masking baseline
    batch = self.update_batch(
        batch,
        updates={properties.R: x_next, keys.sigma_curr: sigma_curr},
    )
```

The alternative — making `preprocess_step` mask-aware — is cleaner in principle
but touches three subclasses and the documented override contract (§12,
"Add a constrained direct-denoising variant"). Prefer the snapshot; it is local
and cannot be forgotten by a future subclass.

**Verification.** `T1` below. Also re-run a small generation with `f_max=0.3`
and confirm validity no longer degrades relative to `f_max=0`.

### A2 — Network activations leak into the batch and get serialized  `SAMPLE`

**Files:** [`sample/base.py`](../src/gpff/sample/base.py),
[`sample/results.py`](../src/gpff/sample/results.py),
[`cli/generate.py`](../src/gpff/cli/generate.py)

`PaiNN.forward` mutates its input dict (`painn.py:155-157`), and
`NeuralNetworkPotential.forward` returns a *filtered* copy via
`extract_outputs` while the caller's dict keeps everything. `Calculator.calculate`
passes the live `batch`, so after the first model call the batch permanently
carries `scalar_representation` `(N, 256)` and `vector_representation`
`(N, 3, 256)`.

Neither key starts with `_`, so both survive the underscore filter in
`batches_to_trajectories` (`results.py:74`) and are run through
`to_serializable` → `.tolist()`.

For `batch_size = 32` (≈576 atoms): `147,456 + 442,368 = 589,824` floats per
snapshot. With `store_trajectories: true` and 50 steps, `stepwise_atoms` holds
all of them before trimming — ≈29.5 M Python float objects, which at ~32 B per
element plus inner-list overhead is **>1 GB per sampled batch**, all discarded
moments later.

The tell is [`cli/generate.py:199-203`](../src/gpff/cli/generate.py):

```python
filtered_data = {k: v for k, v in data.items()
                 if k not in ("scalar_representation", "vector_representation")}
```

The filter exists — it is one stage too late. It runs at DB-write time, after
`batches_to_trajectories` has already materialized everything. The same waste
occurs on the default path: `batches_to_relaxed([final_batch])` serializes one
snapshot per sampling batch even with `store_trajectories: false`, and
`generate.py:168` then keeps exactly one field (`nfe`) out of it.

**Patch.** Drop the activations at the source, in `Calculator.calculate`, right
after the forward. This also removes them from every `TrajectoryRecorder`
snapshot (which clones tensors, ~2.4 MB × 50 per call):

```python
predictions = self.model(batch)

# The representation writes its activations back into the batch dict. Nothing
# downstream reads them, and they would otherwise be cloned by every trajectory
# snapshot and serialized to Python lists by batches_to_trajectories.
for _key in ("scalar_representation", "vector_representation"):
    batch.pop(_key, None)
```

Then add a defensive skip in `batches_to_trajectories` using a shared constant,
and delete the now-redundant filter in `generate.py`.

**Verification.** `T2`. Also assert `store_trajectories: true` completes for
`n_samples=64` inside a fixed RSS budget.

### A3 — Adaptive samplers write positions outside `update_batch`  `SAMPLE`

**Files:** [`sample/heun.py:254`](../src/gpff/sample/heun.py)

`AdaptiveStochasticHeunSampler` assigns `batch[properties.R] = x_hat` directly,
so `update_batch` at line 285 reads the *churned* positions as `old_value` for
converged molecules — the same shape of defect as A1.

It is currently harmless, and the reason is worth recording: EDM churn noise
scales as `√(σ̂² − σ²) ∝ σ`, so a converged (small-`σ`) molecule receives a
proportionally tiny kick. It becomes live the moment `sigma_tmin` is lowered
below the convergence threshold, which is **hard-coded as `0.01` at line 295**
while `sigma_tmin` is a user-facing config value that happens to equal it.

**Patch.** Route the churn through `update_batch`, and derive the convergence
threshold from `self.noise_schedule.sigma_min` instead of the literal `0.01`.

### A4 — Missing `σ` floors in the adaptive samplers  `SAMPLE`

**Files:** [`sample/heun.py:153,179`](../src/gpff/sample/heun.py),
[`sample/ancestral_sampling.py:117,120`](../src/gpff/sample/ancestral_sampling.py)

`AdaptiveHeunSampler` divides by an unclamped model-estimated `σ_curr`
(line 153) and by `σ_next` (line 179), which is exactly `0` on the last step and
whenever `EDMNoiseSchedule.sigma_next` floors it. Both produce `±inf`. The run
survives *only* because `update_convergence(..., 0.0)` fires first and
`update_batch`'s `torch.where` discards the poisoned branch — `torch.where`
selects elementwise, so a non-finite value in the unselected branch is dropped
rather than propagated.

`AdaptiveStochasticHeunSampler` already clamps with
`clamp_min(torch.finfo(...).eps)`; `AdaptiveHeunSampler` and
`AdaptiveAncestralSampler` do not.

**Patch.** Apply the same `clamp_min` in both, and delete the now-dead
`torch.where(sigma_next > 0, ...)` guard at `heun.py:302` (`sigma_next` is
already `clamp_min`'d at line 279, so the condition is always true). Note in the
docstring that the last Heun corrector divides by `eps` and is therefore
discarded by the convergence mask by design.

### A5 — Stale `t` in `AdaptiveStochasticHeun`, absent in `AdaptiveAncestral`  `SAMPLE`

**Files:** [`sample/heun.py:297`](../src/gpff/sample/heun.py),
[`sample/ancestral_sampling.py:82`](../src/gpff/sample/ancestral_sampling.py)

`AdaptiveStochasticHeunSampler`'s stage-2 call omits the `batch[keys.t]` update
that `AdaptiveHeunSampler` performs at lines 170-176, so with
`time_aware_model=true` the corrector evaluates the model at `t(σ̂)` while
integrating to `σ_next`. `AdaptiveAncestralSampler` never sets `keys.t` at all,
so `time_aware_model=true` is silently a no-op there.

**Patch.** Add the missing `get_time_tensor` blocks. Better: factor the
`if self.time_aware_model: batch[keys.t] = self.noise_schedule.get_time_tensor(...)`
idiom into a `DiffusionSampler._set_time(batch, sigma)` helper and call it at
every `calculate` site, so a new sampler cannot omit it.

---

## Workstream B — unreachable and broken interfaces

### B1 — Diffusion baselines cannot be sampled  `CONFIG`

**Files:** [`configs/calculator/default.yaml`](../src/gpff/configs/calculator/default.yaml),
[`sample/base.py`](../src/gpff/sample/base.py), new `configs/sampler/*.yaml`

Three independent blockers, none documented:

1. `noise_key: ${key:noise}` resolves to `keys.noise == keys.eps == "eps"`, but
   `dm_t_aware` / `dm_no_t` name the head's output `eps_pred`. `Calculator.calculate`
   does `predictions[self.noise_key]` → `KeyError: 'eps'`.
2. Every adaptive sampler and the default `direct_denoising` sampler hard-code
   `estimate_sigma=True`, which exists only in the force branch, so those raise
   `ValueError` (`base.py:157`) with a noise model.
3. `time_aware_model: false` in every sampler config, so a `dm_t_aware`
   checkpoint additionally hits B2.

The working invocation is currently
`sampler=heun sampler.calculator.noise_model=true sampler.calculator.noise_key=eps_pred sampler.time_aware_model=true`,
and nothing says so.

**Patch.**
* In `Calculator.calculate`, replace the bare lookup with a check that raises
  naming the available prediction keys:
  ```python
  if self.noise_key not in predictions:
      raise KeyError(
          f"Calculator.noise_key={self.noise_key!r} is not among the model's "
          f"outputs {sorted(predictions)}. For the dm_* baselines set "
          f"sampler.calculator.noise_key=eps_pred."
      )
  ```
* Add `configs/sampler/dm_heun.yaml` and `dm_ve_ddpm.yaml` presets wiring
  `noise_model: true`, `noise_key: eps_pred`, `time_aware_model: true`.
* Document in `README.md` that the adaptive samplers are force-model-only, and
  why (`σ̂ = std(F/2)` has no noise-model equivalent).

### B2 — Time-aware checkpoints fail with a bare `KeyError`  `CONFIG`

**Files:** [`model/heads.py:380`](../src/gpff/model/heads.py),
[`sample/direct_denoising.py:47`](../src/gpff/sample/direct_denoising.py)

`DirectDenoisingSampler.__init__` *forces* `time_aware_model=False` with a
warning saying it "will be ignored". It is not ignored: the head does
`inputs[self.time_key]` and raises `KeyError: 't'`. So `gpff_t_aware` cannot be
sampled with the default sampler at all, and the failure is opaque.

Worse inside `SamplerCallback`: the validation batch still carries a **stale
training-time `t`** from `PseudoPESTransform`, so it silently uses the wrong
value instead of raising.

**Patch.** In `TimeAwareEquivariant.forward`, replace the implicit lookup:

```python
if self.time_key not in inputs:
    raise RuntimeError(
        f"{type(self).__name__} needs inputs[{self.time_key!r}]. Set "
        f"sampler.time_aware_model=true, and use a sampler that supports it "
        f"(not DirectDenoisingSampler, which has no notion of a noise level)."
    )
```

and change `DirectDenoisingSampler`'s warning text to say the combination is
unsupported rather than ignored. `SamplerCallback` should `pop` `keys.t` from
the batch before denoising.

### B3 — `wandb_upload_id=None` cannot disable uploads  `SAFE`

**Files:** [`callbacks.py:242`](../src/gpff/callbacks.py)

`self.wandb_upload_id = wandb_upload_id or "wandb_upload_id"` turns `None` into
a truthy string, so the `if self.wandb_upload_id is None: return` guards at
lines 251 and 289 are unreachable. Artifacts upload under the literal name
`"wandb_upload_id.model"`. (`upload_artifact` no-ops without a `WandbLogger`,
which is what hides it.)

**Patch.** `self.wandb_upload_id = wandb_upload_id` and let the existing guards
work. Also guard `trainer.datamodule.split_file` being `None` before
`artifact.add_file`.

### B4 — `gpff_generate` rebuilds QM9 on a mistyped path  `SAFE`

**Files:** [`cli/generate.py:91`](../src/gpff/cli/generate.py)

`qm9_valid.prepare_data()` treats a non-existent `datapath` as "download raw
QM9 and run the full RDKit validity filter over 130k molecules". A typo starts
a multi-hour build instead of raising.

**Patch.** Add an existence check for `config.datapath` next to the ones already
present for `model_path` (line 68) and `split_path` (line 80).

Related, worth a line in the same patch: `generate.py` builds `QM9Valid` without
`n_atoms_allowed` or `indices_path`. `subset()` composes correctly upstream
(`atoms.py:98-107`), so `properties.idx` really is a filtered-DB row index and
`PrecomputedScaffoldConditionGenerator`'s docstring is accurate — but a model
trained with `n_atoms_allowed` set has split indices addressing the
*size-filtered* subset, while generation addresses the full dataset. Different
molecules, silently. Either propagate the field or refuse to run when the split
length disagrees with the dataset length.

---

## Workstream C — reproducibility and operations

### C1 — Resume is defeated by the run-directory layout  `CONFIG`

**Files:** [`configs/run/default_run.yaml`](../src/gpff/configs/run/default_run.yaml)

Three mechanisms exist specifically to make resume bit-exact:
`DeterministicDataSeed`, the EMA `collected_params` restore (which is correct —
verified against upstream's `store`/`copy_to`/`restore` cycle), and the
`config.yaml` / `last.ckpt` auto-detection in `train.py:48-70`. All three are
reached only if the process lands in the *same* directory as the interrupted
run.

```yaml
path: ${run.work_dir}/runs/${project}/${now:%y-%m-%d}   # ← launch date
id:   ${petname:1}                                      # ← fresh random name
# train.yaml:  hydra.run.dir: ${run.path}/${run.id},  hydra.job.chdir: True
```

`${now:…}` resolves at composition time, so a cluster job requeued after
midnight — the normal case for long runs — chdir's into a new, empty directory
under the new date. `config.yaml` is absent, `ckpt_path` stays `None`, and
training **silently restarts from scratch**, orphaning the previous day's
`checkpoints/last.ckpt`.

**Patch.** Drop the date from the run path:

```yaml
path: ${run.work_dir}/runs/${project}
```

Anyone who wants date bucketing can set `run.path` explicitly, and it then
belongs to the *identity* of the run rather than to its launch time. Add a
migration note to `README.md`; existing runs stay where they are.

Optionally, make the failure loud instead of silent: if `run.ckpt_path is None`
and a `checkpoints/last.ckpt` exists anywhere under `${run.work_dir}/runs/${project}`
matching `${run.id}`, log a warning naming it.

### C2 — The training seed is never persisted  `CONFIG`

**Files:** [`cli/train.py:48-88`](../src/gpff/cli/train.py)

`config.yaml` is written at line 72, but `config.seed` is only generated at
line 86. With no explicit `seed=`, the randomly drawn seed is never recorded,
and a resume draws a new one — defeating `DeterministicDataSeed`, whose
docstring says it exists "so that checkpoint-resumed training reproduces an
uninterrupted run bit-for-bit."

**Patch.** Move the seed block above the resume-detection / `config.yaml` write.
Two lines, and it makes C1's mechanisms actually load-bearing.

### C3 — `val/loss` is logged without `batch_size`  `SAFE`

**Files:** [`task.py:312`](../src/gpff/task.py) (and 281, 336)

Upstream passes `batch_size=len(batch["_idx"])` (`schnetpack/task.py:206`); the
override drops it. Lightning then infers the batch size from an ambiguous
collection, which for a SchNetPack batch may pick the atom count rather than the
molecule count — skewing the epoch average that `ModelCheckpoint(monitor="val/loss")`
and `ReduceLROnPlateau` both key off.

**Patch.** Restore `batch_size=len(batch[properties.idx])` on all three
`self.log` calls.

### C4 — `predict_without_postprocessing` is not exception-safe  `SAFE`

**Files:** [`task.py:175-189`](../src/gpff/task.py)

It swaps `self.model.postprocessors` and restores it after the forward with no
`try/finally`. Any exception in the forward leaves the model permanently holding
only the forced postprocessors. Moot today — no released config defines a
postprocessor — but the mechanism exists precisely for future ones.

**Patch.** Wrap in `try/finally`.

While here: `setup()` registers `forced_postprocessors` as a second
`nn.ModuleList` on the same module, so those submodules appear twice in
`state_dict()`. Prefer a plain Python list attribute (`object.__setattr__`) or a
name-based lookup, so checkpoints stay clean.

---

## Workstream D — evaluation integrity

### D1 — Reference distributions are trusted, not recomputed  `EVAL`

**Files:** [`cli/reference_assets.py:50-56`](../src/gpff/cli/reference_assets.py),
[`cli/prepare_metadata.py`](../src/gpff/cli/prepare_metadata.py)

`prepare_metadata.predict_property` runs `best_model_gap` / `best_model_energy_U0`
on the *generated* molecules. `calculate_metrics` compares that against
`assets/qm9_*_values.npy`, which `reference_assets.py` reads verbatim out of the
reference DB's `metadata` — nothing in this repo ever runs the predictors over
the reference set. Unless that metadata was populated externally by the same
predictors, `js_divergence_energy_U0` and `js_divergence_gap` conflate
**predictor bias with generation quality**, and any systematic offset in the
property model inflates them.

The asymmetry shows in the code's own choices: SMILES are *recomputed* by the
same function on both sides (`reference_assets.py:70` vs
`prepare_metadata.py:171`), while the three distributions are trusted.
`elongation` is the clearest case — the observed side is `compute_elongation`
(max pairwise distance), the reference side is whatever `elongation_values`
happens to hold.

**Patch.**
* Add `--recompute` to `gpff_reference_assets` which, for `elongation_values`,
  walks the DB with `compute_elongation`; and for `gap_values` /
  `energy_U0_values`, runs `predict_property` against the same checkpoints used
  for the samples. Make it the default and keep the metadata path behind an
  explicit `--from-metadata`.
* Record provenance in a sidecar `assets/<prefix>_manifest.json`: source DB
  path, row count, DB file SHA-256, the model paths used, and the gpff commit.
* Have `calculate_metrics` read that manifest and print it in the header, so a
  results table can be traced to the assets that produced it.

### D2 — Novelty depends on three artifacts staying index-aligned  `EVAL`

**Files:** [`cli/calculate_metrics.py:138-150`](../src/gpff/cli/calculate_metrics.py),
[`data/datasets/qm9.py`](../src/gpff/data/datasets/qm9.py)

`load_reference_smiles` indexes the SMILES file by split indices. That requires
the split `.npz`, `qm9_smiles_values.txt` and the filtered DB to describe the
same row ordering. The filter that produces the DB runs RDKit under a
5-second `signal.alarm` per molecule, so its output is machine- and
load-dependent. `KNOWN_FILTERED_SIZES = [121651, 123569]` (`qm9.py:23`) is
direct evidence the count has already varied. Rebuild the DB elsewhere and
novelty silently reads the wrong SMILES per index, with no error.

**Patch.** Extend the D1 manifest to carry the filtered-DB SHA-256 and row
count, write the same fields into the DB metadata in `build_filtered_qm9`, and
have `calculate_metrics` refuse to proceed when
`len(reference_smiles_file) != row_count`, or when the manifest's DB hash does
not match the one recorded in the split file's directory. Fail fast beats a
plausible-looking wrong number.

### D3 — The RDKit timeout cannot interrupt the call it guards  `EVAL`

**Files:** [`analysis/structure.py:47-139`](../src/gpff/analysis/structure.py)

CPython runs signal handlers between bytecodes, so `signal.alarm` cannot
interrupt a long-running C++ call — and `rdDetermineBonds.DetermineBonds`
(line 70) is exactly that, a single C++ call, on precisely the pathological
generated geometries where a hang would occur. The `--timeout` flag therefore
gives false confidence. There is also a narrow window where the alarm fires
*inside* the `finally` block, after `except` has produced a return value,
letting `TimeoutException` escape and kill a 130k-molecule filtering run.

`get_posebusters_validity` has no timeout at all, re-instantiates
`PoseBusters(config="mol")` per molecule, and calls `.bust()` **outside** its
`try`, so a raise there escapes rather than returning `False`.

**Patch.** Move per-structure analysis behind a process-pool with a real
timeout:

```python
with ProcessPoolExecutor(max_workers=1) as pool:
    try:
        result = pool.submit(_analyze, atoms).result(timeout=timeout)
    except TimeoutError:
        result = _INVALID
```

A worker-per-molecule is too costly for 130k structures; use a persistent pool
with a recycle-on-timeout policy. If that is more surgery than wanted, the
minimum honest change is to (a) hoist the `PoseBusters` instance out of the
per-molecule loop, (b) move `.bust()` inside the `try`, and (c) change the
`--timeout` help text to state that it bounds Python-level work only.

---

## Workstream E — measurement tasks for documented design choices

These are **not** patches. Each is a question the code cannot answer by
inspection, where the audit produced a number large enough to be worth
resolving. All are `TRAIN`.

### E1 — Quantify and ablate the `SigmaWeights` clip

§2.2 documents the clip as a deliberate asymmetry ("`σ ≤ 1` → plain force
matching"). The audit's contribution is its size.

With `w = min(1, 1/σ²)` and target power `‖F‖² ≈ 12σ²`, the loss-contribution
density is `12·q(σ)·min(σ², 1)` — it rises as `σ²` up to `σ = 1`, then flattens.
Using `E[e^{2u}·1{u<a}] = e^{2m+2s²}·Φ((a−m−2s²)/s)` with `q = lognormal(−0.7, 1.2)`:

| regime | share of loss mass, clipped | unclipped (`1/σ²`) |
|---|---|---|
| σ < 0.1 | 0.15 % | 9.1 % |
| σ < 0.2 (sub-bond-length refinement) | **0.8 %** | **22.4 %** |
| σ > 10 (where the prior lives) | 1.4 % | 0.6 % |

A **28× under-representation** of the regime that decides bond lengths, and so
RDKit validity. Total mass splits 35 % below `σ = 1` / 65 % above, concentrated
in `σ ∈ [0.5, 3]`.

There is a second-order point that makes this a modelling decision rather than
an optimisation one. For a *time-conditional* model the Bayes-optimal predictor
at each `σ` is independent of `w`; `w` only sets emphasis. GPFF's flagship is
**time-agnostic**, so the optimum is a *weighted* conditional expectation,

```
F*(x) = 2 · ( E[w(σ)·x₀ | x] / E[w(σ) | x]  −  x )
```

and `w` tilts the `σ`-posterior itself. Changing the clip changes which function
is being regressed, not merely how well it is fit.

**Proposed experiment.** Add `sigma_min` to `VarianceExplodingGeometries`
(a lower bound in the existing rejection loop), which bounds `1/σ²` by
construction and lets the cap be raised honestly:

```
sigma_min = 0.05   →   truncates 2.8 % of draws,  cap = 1/σ_min² = 400
                   →   restores the σ < 0.2 share to ≈20 %
```

`0.05 Å` sits below thermal bond-length fluctuation, so nothing chemically
meaningful is truncated. Sweep `max_clipping ∈ {1 (current), 100, 400}` at fixed
`sigma_min = 0.05`, report RDKit and PoseBusters validity plus
`js_divergence_elongation`. One shortened training run per arm is enough to see
the direction.

### E2 — Determine whether the t-aware baselines use `t` at all

`TimeAwareEquivariant.forward` (`heads.py:380-383`) conditions by concatenating
one raw scalar to the 256-dim scalar features, and `t = σ/σ_max` is **linear**
in `σ` with `σ_max = 30`. For `σ ~ lognormal(−0.7, 1.2)`:

| statistic | σ | t = σ/30 |
|---|---|---|
| median | 0.497 | **0.0166** |
| mean | 1.020 | 0.034 |
| interquartile range | [0.221, 1.115] | **[0.0074, 0.0372]** |
| P(t > 0.1) | — | 6.7 % |

So the conditioning signal is one feature whose middle 50 % lives inside a
0.03-wide window near zero, appended to 256 features with O(1) activations, with
no embedding, no Fourier features and no log transform. EDM conditions on
`ln σ / 4`, which over the same range spans ≈[−0.38, 0.03] — roughly 15× more
dynamic range, on the correct scale.

Compounding it: `t` enters **only at the output head**. `model/nnp.yaml` is
`PairwiseDistances → PaiNN → TimeAwareEquivariant`, and PaiNN takes no time
input, so all four message-passing blocks are time-blind and only the final
3-layer gated MLP sees `t`.

If `gpff_t_aware` and `dm_t_aware` are effectively time-*agnostic*, then "GPFF
matches the time-aware baseline" is confounded. Since those baselines carry the
comparison, this should be settled before the clip or the backbone.

**Proposed diagnostic** (cheap, decisive, no retraining):

```
tests/diagnostics/time_sensitivity.py
  load a *_t_aware checkpoint
  for one held-out batch:
      L_true     = loss(batch)
      L_shuffled = loss(batch with keys.t permuted across molecules)
      L_zero     = loss(batch with keys.t := 0)
  report (L_shuffled − L_true)/L_true and (L_zero − L_true)/L_true
```

A ratio near zero means the head ignores `t`. If so, add a `time_embedding`
option to `TimeAwareEquivariant` (`identity` | `log` | `fourier`, defaulting to
`identity` so existing checkpoints are untouched), plus `*_t_aware_v2`
experiment configs using `log`, and re-run the baseline arm.

### E3 — Ablate `norm_epsilon`

§5.1 documents `norm_epsilon = 1.0` as an intentional soft direction vector, so
this is an ablation, not a fix. Two observations that argue for running it:

* The effect at chemically relevant distances is large: `‖dir_ij‖ = d/(d+1)` is
  0.50 at 1 Å and 0.60 at 1.5 Å, so every equivariant message is damped by
  roughly half exactly in the bonding range.
* The stated rationale ("at `d_ij → 0` it damps the direction instead of blowing
  it up") is already covered upstream: `d_ij = sqrt(Σr² + 1e-8)` floors `d` at
  `1e-4`, and self-pairs are excluded by `AllToAllNeighborList`. That explains
  wanting *some* floor; it does not select `1.0` over `1e-2`.

**Patch (config only, `CONFIG`).** Expose it —
`configs/model/representation/painn.yaml` gains `norm_epsilon: 1.0` with a
comment pointing at §5.1 — so the released value is explicit and the ablation is
a one-line override. Do **not** change the default: existing checkpoints carry
the value in their pickled `__dict__` and are unaffected either way, but a
silent default change would make new runs incomparable to published ones.

**Experiment.** `norm_epsilon ∈ {1.0, 0.1, 0.0}`, same seed, same budget.

### E4 — Implement `mean_free_noise`  *(SPEC §11.2)*

Already recorded in §11.2, including the recommendation. Two additions from the
audit.

**The size of the effect.** `x₀` is centered (`pseudo_pes.py:63`) but the noise
is not, so the target `F = −2σε` carries a centre-of-mass component `−2σε̄`.
PaiNN consumes only `Rij` (verified: `PairwiseDistances` →
`R[idx_j] − R[idx_i] + offsets`, and both heads read only the resulting
representations), so it is **strictly translation-invariant** and that component
is structurally unpredictable. It is an irreducible loss floor of `1/N` of
target power — 5.6 % at N = 18, **20 % at N = 5** — and because the loss reduces
with a flat per-atom `torch.mean`, small molecules carry the higher floor and
dominate the residual.

**Corroboration that it was intended.** `align_permutation` already solves the
Hungarian assignment **in the centered frame** (`alignment.py:428-430`) while
returning the uncentered `x_t`. And `align_rotation=true` would incidentally fix
it, since ASE's `minimize_rotation_and_translation` returns positions on the
*target's* centroid.

**Scope note.** Only the training-side change is required. Because the model
cannot see the centre of mass, a non-mean-free prior causes no distribution
shift at sampling — the COM simply stays wherever the prior put it. Centering
`GaussianPrior` and `DirectDenoisingSampler.add_noise` per molecule is
worthwhile for clean trajectories, but it is cosmetic, and it should land as a
separate commit so the A/B isolates the training change.

---

## Workstream F — latent defects and hygiene

| ID | Item | Radius |
|---|---|---|
| F1 | `SigmaWeights` returns a **per-atom** `energy_weight` `(N,1)` while `VEPseudoCalculator.get_energy` returns a per-molecule label that collates to `(B,)`. Any energy head with `weights_key=energy_weight` misbroadcasts. `NormWeights` returns the per-molecule shape. Latent (no released config uses an energy head) — which also means the energy weight is pure dead compute today. | `SAFE` |
| F2 | `PseudoPESTransform` stores `keys.noise` **before** `align_permutation` / `align_rotation` transform `x_t`, so `keys.noise ≠ (x_t − x₀)/σ` afterwards. `energy`/`forces` *are* recomputed post-transform (lines 93-94). Harmless only because the `dm_*` experiments that train on `keys.noise` set `align_permutation: false` — the invariant is not enforced anywhere. Recompute `noise` after alignment, or assert. | `TRAIN`(dm only) |
| F3 | `align_permutation`'s type mask is transposed: `cost_matrix[i,j]` is `[target, positions]`, `mismatch[a,b]` is `[positions, target]` (`alignment.py:434-438`). Currently harmless because `align_numbers` has already made `numbers == target_numbers`, so the mask is symmetric. Fix the indexing and add the assertion that makes it safe. | `SAFE` |
| F4 | `batch_match_principal_variances` has no cap on `scale = sqrt(target/λ.clamp(min=1e-8))` (`alignment.py:353`). The comment at 334-339 describes elongated targets pushing molecules non-finite and guards it with an `isfinite` check on the *covariance* — downstream of the cause. A floor relative to `λ.sum()`, or a cap on `scale`, addresses it at the source. (The exactly-planar case is self-limiting: a zero coordinate cannot be inflated.) | `SAMPLE` |
| F5 | `ScaffoldGaussianPrior` re-centers each molecule on its scaffold centroid *after* the conditions were computed from the original positions (`generate.py:150`), so pairing it with `sampler=shape_scaffold_direct_denoising` translates the free atoms to a centre from a different frame. `ScaffoldPrior` does not re-center and is fine. Refuse the combination, or recompute `free_center` post-prior. | `SAMPLE` |
| F6 | Nested `scaffold_ids` arrive from Hydra as `ListConfig` (`generate.py:116` instantiates without `_convert_`). `isinstance(ListConfig, Sequence)` is True so detection works, but `torch.as_tensor(ListConfig)` is incidental. Pass `_convert_="all"`. | `SAFE` |
| F7 | `EDMNoiseSchedule.sigma_next` returns NaN for non-integer `rho` (`(base + δ)**rho` on a negative base; fine for `5.0`, NaN for e.g. `3.5`), and NaN survives the `torch.where` on line 122. The abstract signature says `sigma_target` while the implementation says `sigma_curr` (`schedules.py:38` vs `105`). `get_t` returns negatives below `sigma_min`. | `SAFE` |
| F8 | `AtomisticTask` divergence fails open: `skip_exploding_batches: True` with `exit_after_n_skipped_train_batches` unset means a diverged run drops every batch indefinitely. Set a default. | `CONFIG` |
| F9 | Already in `SPEC.md` §11 — schedule but do not re-analyse: 3-vs-4-tuple arity in `VariancePreservingGeometries` / `MixedGeometries`; `callbacks/wandb_cleanup.yaml`; `generative_transform/diffusion.yaml`; inert `ModelOutput.warmup_steps`; `callbacks/sampler.yaml` interpolations; `generate.yaml` help listing non-existent samplers; `from token import OP`; `configs/__init__.py`. | mixed |

Two hygiene items worth pairing with F9 because they share a cause — configs
that reference symbols nobody imports: `model/representation/{schnet,so3net,field_schnet}.yaml`
are offered but none emit `vector_representation`, which every gpff head
requires. Either delete them or add the missing head wiring.

---

## Workstream G — performance

### G1 — The filter net dominates the model  `TRAIN`

§5.1 already notes 600 RBFs is large and suggests `shared_filters: true`. The
audit adds the ratio, which makes the case sharper.

Per batch (128 molecules × 18 atoms → 306 pairs each = 39,168 pairs), with
`filter_net = Dense(600, 4·256·3)`:

| | FLOPs |
|---|---|
| filter net | `39168 × 600 × 3072 × 2` ≈ **144 G** |
| all four interaction + mixing blocks | `4 × 2304 × ~262k × 2` ≈ **4.8 G** |

**≈30×.** The `filters` tensor alone is `39168 × 3072` fp32 = **482 MB**, held
live across the interaction loop because `torch.split` returns views.

The resolution choice is *correct* — `GaussianRBF` width is `150/599 = 0.250 Å`,
essentially identical to standard PaiNN's `5/19 = 0.263 Å`. The cost is the
problem: 30× was paid for coverage of a range where nearly all pair mass sits
below ~50 Å.

**Options, cheapest first.** (a) `shared_filters: true`, as §5.1 suggests — 4× on
the output width. (b) A log-spaced basis: `ExpGaussianRBF` already exists in
`nn/radial.py` with a config nothing selects, and would give identical
short-range resolution at a fraction of the count. (c) Reduce `cutoff` below
`5·σ_max` — but note invariant §13.11 couples `sigma_max`, `cutoff` and `n_rbf`,
so this is a three-way change.

Any of these is `TRAIN`; run it as an ablation against the released
hyperparameters, and only after E1/E2 have settled the objective.

### G2 — Vectorise the per-molecule Python loops  `SAFE`

Already in §11.5. `Calculator._compute_std_per_molecule` is the one on the
sampling hot path — it forces a device sync at every step of every sampler,
twice per Heun step. `scatter_reduce` with a mean-and-second-moment pass
replaces it exactly (note `torch.std` uses `correction=1`; preserve that or
document the change).

---

## T — Test plan

There is no `tests/` directory. A1 and A3 are exactly the class of defect a
short regression test catches, and E1–E4 need a trustworthy baseline. Propose
`tests/` with `pytest`, CPU-only, no downloads — all of the following run
against a synthetic 5-molecule batch and a 2-interaction toy PaiNN.

| ID | Test | Catches |
|---|---|---|
| T1 | Freeze a molecule via `f_max`, run to completion, assert its positions are **bit-identical** to the step at which it converged. | A1, A3 |
| T2 | After `denoise`, assert `set(batch) & {"scalar_representation", "vector_representation"} == set()`, and that `batches_to_trajectories` output contains neither key. | A2 |
| T3 | Round-trip the pseudo-PES identities: for random `(x₀, σ)`, assert `x₀ == x_t + F/2` and `std(F/2) ≈ σ` to tolerance. Guards SPEC §13.1. | forward-process changes |
| T4 | Assert every `Geometries` subclass's `get_noisy_geometry` returns a 4-tuple. | SPEC §11.1 arity |
| T5 | Run each sampler in `configs/sampler/` for 3 steps against the toy model; assert finite positions and `nfe ≤ n_steps × calls_per_step`. | A4, A5, B1 |
| T6 | Instantiate every YAML under `configs/` with `hydra.utils.instantiate` (`_target_` resolution only, no training). | F9, B1 |
| T7 | Rotate and translate an input batch; assert head outputs rotate equivariantly and are translation-invariant. | E3, and `TimeConditionedGatedEquivariant` if it is ever revived |
| T8 | `check_uniqueness` / `check_novelty` / `calculate_js_divergence` on hand-built inputs including empty SMILES and NaN. | D1, D3 |

T6 is the highest value-per-line: it would have caught `wandb_cleanup.yaml`,
`diffusion.yaml` and B1 at authoring time.

---

## Checkpoint compatibility

| change | existing checkpoints | notes |
|---|---|---|
| A1–A5, B1–B4, C1–C4, D1–D3, F1–F9, G2 | **load and run unchanged** | A1–A5 change *sampled outputs*; re-run generation and evaluation, and do not mix old and new samples in one table |
| E1 (clip / `sigma_min`) | load, but incomparable | retrain per arm |
| E2 (`time_embedding`) | load — default `identity` preserves current behaviour | new arms need retraining |
| E3 (`norm_epsilon` config exposure) | load unchanged | value is pickled in the module `__dict__`; do **not** change the class default |
| E4 (`mean_free_noise`) | load, but incomparable | retrain; land the sampler-side centering separately |
| G1 | load unchanged | new arms need retraining |

One caveat for E3: a gpff `PaiNN` pickled *before* `norm_epsilon` existed would
raise `AttributeError` in `forward`. `prepare_metadata._patch_legacy_painn_attrs`
handles this pattern for upstream PaiNN attributes but is not applied on the
`gpff_generate` path. If any such checkpoint exists, extend that helper to
`norm_epsilon: 1.0` and call it from `generate.py` too.

---

## Open questions for the authors

1. **Was `f_max` ever exercised?** If any reported NFE figure came from a run
   with `f_max > 0`, A1 means the reported structures drifted after
   convergence. If it was never used, the adaptive-NFE story rests entirely on
   the adaptive Heun / ancestral samplers, and A4/A5 apply there instead.
2. **Do the `*_t_aware` checkpoints respond to `t`?** E2 is a 20-minute
   diagnostic and determines whether the central baseline comparison holds.
3. **How were `energy_U0_values` and `gap_values` produced?** If by the same
   predictors used on the samples, D1 reduces to recording provenance. If by
   DFT, the two JS divergences need recomputing.
4. **Was `mean_free_noise: true` in the released config an intent that was never
   implemented, or a leftover?** §11.2 reads as the former. It changes the
   irreducible loss floor by `1/N`, so it is worth confirming before retraining.
5. **Is the `σ = 30` prior deliberate given the training distribution?**
   `P(σ > 25) ≈ 3 × 10⁻⁵` under `lognormal(−0.7, 1.2)`, so the first ~15 of 50
   EDM steps operate where under 1.5 % of training samples live. The default
   `direct_denoising` sampler is robust to this by construction (it ignores `σ`
   and re-noises at scale ~1); the Heun and ancestral samplers are not. Worth a
   sentence in the paper either way.
