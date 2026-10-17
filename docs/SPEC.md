# GPFF — Design and Architecture Specification

Reference description of the `gpff` codebase as it stands at the initial public
release (`04252cb`). It is written for two audiences: someone extending the
package (new sampler, prior, dataset, head, metric) and someone revisiting the
architectural decisions behind it. It documents what the code *does*, the
contracts it relies on, and where those contracts are thin.

Everything here was read off the source. Where the code and the docs/config
comments disagree, the source wins and the disagreement is called out in
§11.

---

## Table of contents

1. [What GPFF is](#1-what-gpff-is)
2. [The pseudo-PES formulation](#2-the-pseudo-pes-formulation)
3. [Repository layout](#3-repository-layout)
4. [Data pipeline](#4-data-pipeline)
5. [Model](#5-model)
6. [Training](#6-training)
7. [Sampling](#7-sampling)
8. [Conditioning: shape and scaffold](#8-conditioning-shape-and-scaffold)
9. [Evaluation](#9-evaluation)
10. [Configuration system](#10-configuration-system)
11. [Known gaps, dead code, and sharp edges](#11-known-gaps-dead-code-and-sharp-edges)
12. [Extension recipes](#12-extension-recipes)
13. [Invariants worth preserving](#13-invariants-worth-preserving)

---

## 1. What GPFF is

GPFF (Generative Pseudo-Force Field) generates 3D molecular structures by
training an equivariant **force field** rather than a noise predictor.

The standard denoising-diffusion recipe trains `ε_θ(x_t, t)` — a network that,
*given the noise level `t`*, predicts the noise that was added. GPFF drops the
`t` input. It trains `F_θ(x)`, a plain force predictor, on a synthetic
("pseudo") potential energy surface whose minimum is the clean structure. Two
properties follow, and they are the whole point of the design:

1. **One forward pass yields the clean structure directly.**
   `x̂₀ = x + F_θ(x)/2` — no schedule, no `t`, no ODE integration required.
2. **The force magnitude *is* the noise level.**
   `σ̂ = std(F_θ(x)/2)` per molecule. The sampler can ask the model *where it
   is* on the trajectory instead of being told, which enables per-molecule
   adaptive step sizes, per-molecule convergence, and early exit.

Everything downstream — the direct-denoising sampler, the adaptive Heun and
ancestral samplers, the NFE bookkeeping, the shape and scaffold conditioning —
is built on those two identities.

The package is a thin, opinionated layer on top of **SchNetPack** (data module,
`AtomisticTask`, `NeuralNetworkPotential`, transforms, `Atomwise`) and
**PyTorch Lightning** + **Hydra**. `morered` supplies exactly three symbols
(`AllToAllNeighborList`, `Diffuse`, `linear_decay`), which is why it is
installed `--no-deps`.

**Scope of the released code:** QM9 only, positions only. Atom types and atom
counts are *not* generated — they are taken from a test-split molecule and held
fixed while the positions are resampled from a prior and denoised. This is the
single most important scoping fact about the repository and it shapes the
sampling CLI (`n_samples` cannot exceed the test-split size).

---

## 2. The pseudo-PES formulation

### 2.1 Forward process and labels

For a clean structure `x₀ ∈ ℝ^{N×3}` (centred: the transform subtracts the
centroid) the training sample is built in
[`src/gpff/data/transform/pseudo_pes.py:47`](../src/gpff/data/transform/pseudo_pes.py):

```
σ  ~  LogNormal(μ = −0.7, s = 1.2),  rejection-resampled until σ ≤ σ_max
t  =  σ / σ_max                                  (per-atom, shape (N,1))
ε  ~  N(0, I_{N×3})
x_t = x₀ + σ ε
```

The pseudo-energy and pseudo-force
([`src/gpff/pseudo_pes/labels.py:28`](../src/gpff/pseudo_pes/labels.py)) are:

```
E(x_t)  =  Σ_{i,a} (x_t − x₀)²_{ia}                 (one scalar per structure)
F(x_t)  =  −∇_{x_t} E  =  2 (x₀ − x_t)  =  −2 σ ε   (shape (N,3))
```

so the pseudo-PES is a single isotropic harmonic well centred on the training
structure. The `2` factor is what puts the `/2` in `x̂₀ = x + F/2`.

`σ` is sampled log-normally rather than uniformly, so training concentrates
around `σ ≈ e^{−0.7} ≈ 0.5 Å` (where chemistry is decided) while still covering
the full range up to `σ_max = 30 Å` (where the prior lives). Rejection sampling
is a `while` loop — with the default parameters the acceptance rate is ~99.9%,
but it is unbounded in principle.

### 2.2 Loss weighting

`SigmaWeights` ([`src/gpff/pseudo_pes/weights.py:51`](../src/gpff/pseudo_pes/weights.py))
emits, clipped to `[0, 1]`:

```
w_E = clip(1/σ⁴)      w_F = clip(1/σ²)
```

and the loss is `mean( w_F ⊙ (F_θ − F)² )`
([`src/gpff/task.py:22`](../src/gpff/task.py)).

Because `‖F‖ ≈ 2σ√3` per atom, the `1/σ²` weight exactly cancels the target's
scale. The clipping is a deliberate asymmetry:

* `σ > 1`: weight `1/σ²` → the objective is **equivalent to ε-matching**.
* `σ ≤ 1`: weight clipped to `1` → the objective is **plain force matching**,
  so the near-clean regime is *not* up-weighted the way an ε-parameterisation
  would up-weight it.

`NormWeights` is the alternative (weight by the observed label magnitude rather
than by `σ`). It is wired in the config tree but not used by any released
experiment.

### 2.3 Permutation alignment — the non-obvious ingredient

At `σ = 30 Å` the noised structure is, for all practical purposes, an isotropic
Gaussian cloud that has forgotten the molecule. The *index* pairing between
`x_t[i]` and `x₀[i]` is then arbitrary, and the naive label
`2(x₀[i] − x_t[i])` asks the network to send each cloud atom to an
essentially random slot — a target that is not a function of the input.

`align_permutation`
([`src/gpff/utils/alignment.py:390`](../src/gpff/utils/alignment.py)) fixes
this with a Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on
squared distances between the centred clouds, with mismatched atomic numbers
given cost `1e12` so **only same-element atoms can be paired**. `x_t` is then
reordered so `x_t[i]` is the cloud atom assigned to `x₀[i]`; the atomic-number
array is permuted alongside, and (because of the element mask) comes back
unchanged.

The result is that the pseudo-force field points every noised atom at its
*nearest compatible* clean atom, which is a smooth, learnable, permutation-
invariant target. `experiment=gpff_no_permutation` is the ablation that turns
it off.

Cost: an `O(N³)` assignment per sample per epoch, run in the dataloader
workers on CPU. For QM9 (`N ≤ 29`) this is cheap; for larger molecules it is
the first thing that will bite.

`align_rotation` (Kabsch, via `ase.build.minimize_rotation_and_translation`) is
available and defaults to `false` in every released experiment.

### 2.4 Prediction conversions

`Calculator.calculate`
([`src/gpff/sample/base.py:130`](../src/gpff/sample/base.py)) is the single
place where a raw model output becomes whatever a sampler asked for. It
supports both parameterisations behind one interface:

| requested | force model (`noise_model: false`) | noise model (`noise_model: true`) |
|---|---|---|
| `forces` | `F` | `−2σ ε̂` |
| `noise`  | `−F / (2σ)` | `ε̂` |
| `score`  | `F / (2σ²)` | `−ε̂ / σ` |
| `x_0`    | `x + F/2` | `x − σ ε̂` |
| `sigma_curr` (`estimate_sigma`) | `std(F/2)` per molecule | — (unsupported) |

Two consequences worth internalising:

* **Only the force model can estimate `σ`.** Every adaptive sampler therefore
  requires `noise_model: false`. A noise model must be driven by the schedule.
* **`σ` is required** whenever a conversion divides by it. The method raises
  up front if a sampler asks for `score`/`noise`/`x_0` without either an
  explicit `sigma_t` or `estimate_sigma`.

`_compute_std_per_molecule` loops over `idx_m.unique()` in Python. It is the
hottest non-model code in the sampling loop and an obvious vectorisation
target (`scatter_reduce` over the flattened tensor, as `RemoveVariance`
already does).

---

## 3. Repository layout

```
src/gpff/
├── __init__.py          matred→gpff module aliases for unpickling old checkpoints
├── keys.py              string constants for batch dict keys
├── task.py              ModelOutput (weighted loss) + AtomisticTask (skip-NaN, EMA-safe)
├── callbacks.py         EMA resume fix, deterministic seeding, sampling-during-training,
│                        checkpointing + W&B artifact upload
├── cli/                 one module per console entry point
│   ├── __init__.py      banner, torch.load shim, OmegaConf resolvers
│   ├── train.py             gpff_train
│   ├── generate.py          gpff_generate
│   ├── prepare_metadata.py  gpff_prepare_metadata   (phase 1, GPU)
│   ├── calculate_metrics.py gpff_calculate_metrics  (phase 2, CPU)
│   └── reference_assets.py  gpff_reference_assets
├── pseudo_pes/          the forward process and its labels
│   ├── geometries.py    σ/x_t sampling  (VE, VP, Mixed)
│   ├── labels.py        pseudo energy + analytic pseudo force
│   └── weights.py       per-sample loss weights
├── data/
│   ├── datasets/qm9.py  QM9Valid + build_filtered_qm9
│   └── transform/       PseudoPESTransform, Remove/AddVariance
├── model/
│   ├── painn.py         PaiNN fork (returns vector representation)
│   └── heads.py         GatedEquivariant, TimeAwareEquivariant, Atomwise, DiffusionTime
├── nn/radial.py         ExpGaussianRBF
├── sample/              everything inference-side
│   ├── base.py          Calculator, Sampler, DiffusionSampler
│   ├── schedules.py     NoiseSchedule, EDMNoiseSchedule
│   ├── direct_denoising.py  DD samplers (plain / shape / scaffold / shape+scaffold)
│   ├── heun.py          Heun, StochasticHeun, Adaptive(+Stochastic)Heun
│   ├── ancestral_sampling.py  Ancestral, AdaptiveAncestral
│   ├── priors.py        Gaussian, Shape, Scaffold, ScaffoldGaussian
│   ├── conditions.py    condition generators (shape / scaffold / precomputed scaffold)
│   ├── shape_predictor.py   ShapeGMM (per-atom-count GMM over principal variances)
│   ├── observers.py     SamplerObserver, ProgressReporter, TrajectoryRecorder
│   └── results.py       batch snapshots → ase.Atoms + metadata
├── analysis/
│   ├── structure.py     per-structure: RDKit validity, SMILES, rings, elongation, PoseBusters
│   └── collection.py    per-set reductions: uniqueness, novelty, Tanimoto, JS divergence
├── utils/
│   ├── alignment.py     covariance/PCA/permutation/rotation helpers
│   ├── batching.py      batch ↔ ase.Atoms
│   ├── neighbors.py     compute_neighbors, AllToAllNeighborList
│   └── wandb.py         artifact upload / download helpers
└── configs/             Hydra config tree (see §10)
```

There are **no tests** in the repository.

---

## 4. Data pipeline

### 4.1 `QM9Valid`

[`src/gpff/data/datasets/qm9.py`](../src/gpff/data/datasets/qm9.py) subclasses
SchNetPack's `QM9` datamodule and adds two things.

**Filtering (`prepare_data`).** If the target database does not exist, the raw
QM9 (`remove_uncharacterized=True`) is downloaded next to it and
`build_filtered_qm9` writes a new database containing only molecules RDKit
accepts (neutral, radical-free, single fragment). Each row keeps its
`original_index` and its canonical SMILES. The write goes to a `_partial.db`
first and is `os.replace`d, so an interrupted build cannot leave a
plausible-looking half database.

A guard rejects an existing database that neither carries `filter_flags:
["valid"]` in its metadata nor has one of the two `KNOWN_FILTERED_SIZES`
(`121651`, `123569` — the second is the current recipe, the first predates
`remove_uncharacterized=True`).

**Size-restricted subsets (`setup`).** `n_atoms_allowed` selects only molecules
with an exact atom count, caching the index list in a pickle. Unused by the
released experiments.

Splits: `num_train: 0.8`, `num_val: 0.1`, test = remainder. The split file is
written by SchNetPack and is a **required input to sampling and to the novelty
metric** — it identifies which molecules the model saw.

### 4.2 Transform chain

Every released experiment uses the same three transforms, in this order:

```yaml
data.transforms:
  - ${generative_transform}                        # PseudoPESTransform
  - _target_: morered.transform.AllToAllNeighborList
  - _target_: schnetpack.transform.CastTo32
```

Order matters: the neighbour list must be built **after** the positions have
been replaced by `x_t`, and `CastTo32` must run last.

The neighbour list is **fully connected**. With `cutoff = 5 · σ_max = 150 Å`
and a molecule that never exceeds ~10 Å, a distance cutoff would be
meaningless at high `σ` — atoms in the `σ = 30` cloud are tens of Ångström
apart and still need to see each other. The same choice is repeated at
inference (`Calculator.update_neighbors` calls `compute_neighbors(...,
fully_connected=True)`), which is also why the list can be built **once** per
denoising run and reused as positions move.

### 4.3 What `PseudoPESTransform` writes into the batch

| key | shape | meaning |
|---|---|---|
| `properties.R` | `(N,3)` | the noised (and permuted) positions `x_t` |
| `properties.Z` | `(N,)` | atomic numbers, permuted alongside |
| `keys.x_0` | `(N,3)` | the centred clean structure |
| `keys.noise` (`"eps"`) | `(N,3)` | the sampled `ε` |
| `keys.sigma_curr` | `(N,1)` | `σ`, broadcast per atom |
| `keys.t` | `(N,1)` | `σ/σ_max`, broadcast per atom |
| `keys.energy` | `(1,)` | pseudo-energy label |
| `keys.forces` | `(N,3)` | pseudo-force label |
| `keys.energy_weight`, `keys.forces_weight` | `(N,1)` | loss weights (if a weight generator is set) |

The transform is the *only* place the forward process is defined. Swapping the
generative process means swapping this transform (or its three injected
components), nothing else.

---

## 5. Model

### 5.1 Backbone

[`src/gpff/model/painn.py`](../src/gpff/model/painn.py) is a fork of
SchNetPack's PaiNN. The differences are small and deliberate:

* `return_vector_representation` publishes `mu` as `"vector_representation"`,
  which the equivariant heads consume.
* `norm_epsilon` (default `1.0`) is added when normalising `r_ij` to a
  direction: `dir_ij = r_ij / (d_ij + norm_epsilon)`. This is **not** a
  numerical-stability epsilon — at `d_ij = 150 Å` it is negligible, at
  `d_ij → 0` it damps the direction vector instead of blowing it up. It makes
  `dir_ij` a *soft* direction whose magnitude grows with distance.
* `nuclear_embedding` / `electronic_embeddings` hooks are exposed.

Released hyperparameters (`experiment=gpff`):

| | |
|---|---|
| `n_atom_basis` | 256 |
| `n_interactions` | 4 |
| `cutoff` | `5 · σ_max` = **150 Å** (`CosineCutoff`) |
| `n_rbf` | `4 · cutoff` = **600** (`GaussianRBF`) |

600 radial basis functions is very large by atomistic-ML standards; it is
forced by the 150 Å range. The unshared filter net alone is
`Dense(600 → 4·256·3 = 3072)` ≈ 1.8 M parameters. `shared_filters: true` is the
lever if this becomes a problem.

`ExpGaussianRBF` ([`src/gpff/nn/radial.py`](../src/gpff/nn/radial.py)) is an
alternative basis that maps distance through `exp(−γ d)` before the Gaussians,
concentrating resolution at short range. Available as
`model/representation/radial_basis=exponential_gaussian`; not used by the
released experiments.

### 5.2 Heads

| head | used by | notes |
|---|---|---|
| `GatedEquivariant` | `gpff`, `gpff_no_permutation`, `dm_no_t` | vector output from `(l0, l1)`; time-conditioning code is commented out |
| `TimeAwareEquivariant` | `gpff_t_aware`, `dm_t_aware` | concatenates `t` onto `l0`; `time_head=null` means `t` is read from the batch |
| `Atomwise` | — | scalar head with optional time conditioning and `σ²` scaling |
| `DiffusionTime` | — | invariant `t` predictor (would let a model infer its own `t`) |
| `TimeConditionedGatedEquivariant` | — | not exported in `__all__` |

**Design decision worth flagging:** the model predicts forces **directly** from
an equivariant head, *not* as `−∇E`. The learned field is therefore not
guaranteed to be conservative. The pseudo-energy exists only as a label and as
the derivation of the force target; no released experiment attaches an energy
output at all. If a conservative field is wanted, the pieces are already
present (`Atomwise` for the energy, SchNetPack's `Forces` response module for
the derivative) — it is a config change plus a `ModelOutput` entry, not a code
change.

`DiffusionTime` is the unused counterpart of the same idea on the other side:
it would let a nominally time-aware model predict its own `t`, which is how
MoreRed handles the same problem. GPFF's answer is `σ̂ = std(F/2)` instead.

---

## 6. Training

### 6.1 Entry point

`gpff_train` → [`src/gpff/cli/train.py`](../src/gpff/cli/train.py). Hydra
`chdir: True`, so the run directory is
`runs/${project}/${date}/${run.id}` with `run.id` a random `petname`.

Flow: resume detection (`config.yaml` present → archive it as
`config.old.N.yaml`, pick up `checkpoints/last.ckpt`) → seed → instantiate
datamodule, model, task, callbacks, loggers, trainer → `fit` → `test` →
reload the best checkpoint and `save_model(model_path,
do_postprocessing=True)`.

Two artifacts come out of a run and both are needed later:

* `best_model` — the inference model (`gpff_generate model_path=…`)
* `split.npz` — the split (`gpff_generate split_path=…`, and the novelty
  reference in `gpff_calculate_metrics`)

### 6.2 `AtomisticTask` additions

[`src/gpff/task.py:105`](../src/gpff/task.py) extends SchNetPack's task with:

* **`predict_without_postprocessing`** — swaps `model.postprocessors` for the
  subset carrying `force_apply=True` during train/val/test, so that
  postprocessors which are part of the *parameterisation* (e.g. `AddVariance`)
  still run while inference-only ones do not.
* **Exploding-batch guard** — a train batch whose loss is NaN/Inf or exceeds
  `loss_threshold` (`1e3`) is skipped (returns `None`), counted, and logged as
  `train/skipped_nan`. `debug_folder` dumps the offending batch and weights;
  `exit_after_n_skipped_train_batches` turns a persistent problem into a hard
  failure instead of a silently degenerate run.
* **`ModelOutput` with `weights_key`** — the loss weight tensor travels through
  the *targets* dict, so `_filter_targets` pulls `weights_key` out of the batch
  alongside the target property.
* **`warmup_steps` on `ModelOutput`** — linearly blends from uniform weights to
  the real ones. Note: `ModelOutput.global_step` is initialised to `0` and
  **never incremented**, so with `warmup_steps > 0` the dampening is
  permanently `0` and the weights stay uniform forever. Every released config
  sets `warmup_steps: 0`, which is why this has not surfaced.

### 6.3 Reproducibility machinery

This is a deliberately engineered part of the package and easy to break.

* **`DeterministicDataSeed`** ([`src/gpff/callbacks.py:60`](../src/gpff/callbacks.py))
  reseeds `random` / `numpy` / `torch` with `base_seed + epoch` at every
  `on_train_epoch_start`, in the spirit of `DistributedSampler.set_epoch`.
  Lightning does not checkpoint RNG state, so without this a resumed run draws
  a different stream than an uninterrupted one and silently diverges. It also
  recreates the *first* epoch's dataloader iterator once, because PL 2.5.3
  builds it in `_FitLoop.setup_data` before the hook fires.

  This matters more here than in a typical trainer: the noise, the `σ` draw and
  the permutation alignment all happen in the dataloader workers, so the data
  stream *is* the forward process.

* **`ExponentialMovingAverage`** ([`src/gpff/callbacks.py:29`](../src/gpff/callbacks.py))
  fixes a resume bug in SchNetPack's EMA callback: checkpoints are written
  while the EMA average is `copy_to()`'d into the model, so a resumed run must
  first `restore()` the raw training weights (the EMA's `collected_params`)
  before continuing. It also suppresses the double-store during sanity
  checking.

* **`torch.load` shim** ([`src/gpff/cli/__init__.py:29`](../src/gpff/cli/__init__.py))
  forces `weights_only=False` globally. Torch ≥ 2.6 defaults to `True`, which
  cannot unpickle full Lightning checkpoints containing SchNetPack objects.
  This is a process-wide monkeypatch and it is only safe because the package
  loads nothing but its own checkpoints — a constraint the `gpff_generate`
  docstring reinforces by refusing any non-local model path.

* **`matred` aliases** ([`src/gpff/__init__.py`](../src/gpff/__init__.py))
  register `matred.*` → `gpff.*` in `sys.modules` so checkpoints pickled under
  the package's former name still resolve.

* **Dependency pinning** — `pyproject.toml` pins the whole numerical stack
  around `torch==2.8.0` with the comment that newer torch releases changed
  numerical results.

### 6.4 Released experiments

| experiment | predicts | `align_permutation` | time input |
|---|---|---|---|
| `gpff` | pseudo-forces | ✅ | none |
| `gpff_no_permutation` | pseudo-forces | ❌ | none |
| `gpff_t_aware` | pseudo-forces | ✅ | `t` concatenated |
| `dm_no_t` | `ε` | ❌ | none |
| `dm_t_aware` | `ε` | ❌ | `t` concatenated |

The four ablations isolate exactly two variables: the **parameterisation**
(force vs `ε`) and the **time input** (present vs absent), plus the permutation
alignment which only makes sense for the force parameterisation. `dm_*` also
drop the loss weights (`weight_generator: null`), since `ε`-matching is already
scale-free.

All five share the same backbone, the same `σ_max = 30`, the same prior
(`GaussianPrior(sigma=σ_max)`), and the same `lr = 1e-4` with AdamW +
`ReduceLROnPlateau(factor=0.5, patience=250)`.

---

## 7. Sampling

### 7.1 Entry point and loop

`gpff_generate` → [`src/gpff/cli/generate.py`](../src/gpff/cli/generate.py).
Hydra `chdir: False`, so paths are relative to the invocation directory.

```
load model (local file, required)      load split file (local, required)
                     ↓
        QM9Valid(datapath, split_file, load_properties=[])
                     ↓
   for each test batch (until n_samples written):
        conditions = condition_generator.from_batch(batch)   # from the DATA positions
        batch      = prior.sample_from_batch(batch, conditions)
        final      = sampler.denoise(batch, conditions, observers=[recorder?])
        atoms      = batches_to_relaxed([final])
        write atoms (+ nfe) to samples.db   [+ trajectories.db]
```

Ordering constraint: **conditions are built before the prior overwrites the
positions.** Scaffold and shape conditioning both read the original geometry.

The run config is written into the database's ASE metadata, so a samples
database is self-describing.

`n_samples` is capped by the test-split size — the pipeline resamples positions
for existing molecules, it does not generate compositions.

### 7.2 `Sampler.denoise` contract

[`src/gpff/sample/base.py:354`](../src/gpff/sample/base.py). Runs under
`@torch.no_grad()`.

```
conditions ← cast once (never mutated inside the loop)
batch[converged] ← zeros(n_molecules, bool)
batch[nfe]       ← zeros(n_molecules, int32)
batch ← _init_state(batch)                    # sampler-specific seeding hook
observers.on_start(...)
for step in n_steps … 1:
    batch ← cast_batch(batch)
    batch ← denoise_step(batch, step, conditions)
    batch ← check_convergence(batch, step)
    observers.on_step_end(batch, step)
    if all converged: break
observers.on_end(batch)
return clone_batch(batch)                     # CPU copy
```

Four cooperating mechanisms make the per-molecule adaptivity work:

* **`converged`** — a per-molecule boolean. `check_convergence` sets *all* of it
  at `step == 1` (so the loop always terminates on the last step) and freezes
  any molecule whose positions went NaN, which stops divergent molecules from
  burning further NFE.
* **`update_batch(batch, updates)`** — writes atom-wise updates but leaves atoms
  of converged molecules at their previous values. A key not yet present is
  written unconditionally.
* **`update_convergence(batch, value, threshold, reduce)`** — reduces a per-atom
  scalar to a per-molecule statistic (`mean` by default, `amax` for force norms)
  and ORs the threshold test into `converged`.
* **`nfe`** — incremented inside `Calculator.calculate` by `~converged`, so the
  cost metric counts only work actually spent on unfinished molecules.

Several samplers call `update_convergence` **mid-step, on purpose**, so a
molecule that converges now does not take this step's position update (direct
denoising) or does not pay for the second Heun stage. The comments in the source
mark these; do not "tidy" them into `check_convergence`.

**Observers** (`on_start` / `on_log` / `on_step_end` / `on_end`) are how logging
and trajectory recording stay out of the sampler bodies. `TrajectoryRecorder`
collects a CPU `clone_batch` per step; `ProgressReporter` prints and/or drives
tqdm. Per-call observers are merged with the sampler's own for the duration of
one `denoise`.

### 7.3 The samplers

#### DirectDenoisingSampler — the GPFF-native one

[`src/gpff/sample/direct_denoising.py:89`](../src/gpff/sample/direct_denoising.py).
Not an ODE/SDE integrator at all. Each step:

```
r  = noise_ratio(step) = (step − 1) / n_steps      # ~1 → 0
x ← x + λ · r · ξ,   ξ ~ N(0, I)                   # preprocess_step
F, x̂₀, σ̂ ← calculator.calculate(forces, x0, estimate_sigma)
if f_max > 0: converge molecules with max‖F‖ ≤ f_max     # mid-step
x ← x̂₀                                            # frozen molecules keep x
postprocess_step(...)                              # constraint enforcement
```

This is annealed iterative refinement: jump straight to the model's `x̂₀`, then
re-inject a shrinking amount of noise to let it escape. It only works because
the model is time-agnostic — there is no `t` to be inconsistent with.

The class defines **two override hooks** so that constrained variants are ~20
lines each:

* `preprocess_step` — perturbation and any pre-inference constraint
* `postprocess_step` — post-update constraint enforcement

| subclass | preprocess | postprocess |
|---|---|---|
| `ShapeDirectDenoisingSampler` | perturb, then blend towards target principal-variance *ratios* | — |
| `ScaffoldDirectDenoisingSampler` | perturb free atoms only | restore scaffold atoms to `x_prior` |
| `ShapeScaffoldDirectDenoisingSampler` | perturb + covariance/centre match, free atoms only | restore scaffold |

The shape blend uses
`(1 − ρ^p)·reshaped + ρ^p·current` with `ρ = denoising_ratio(step)` and
`p = prior_shape_denoising_power` (default 4). Early steps are dominated by the
reshaped positions; the exponent makes the constraint release quickly so the
final structure is chemistry-driven, not shape-driven.

#### Heun samplers

[`src/gpff/sample/heun.py`](../src/gpff/sample/heun.py). Standard EDM
second-order integration of the probability-flow ODE, with `x̂₀` supplied by the
calculator:

```
d₁ = (x − x̂₀)/σ_curr
x' = x + (σ_next − σ_curr) d₁
if σ_next > 0:  d₂ = (x' − x̂₀')/σ_next ;  x' = x + (σ_next − σ_curr)(d₁+d₂)/2
```

`StochasticHeunSampler` adds EDM churn: inflate `σ_curr → σ̂ = (1+γ)σ_curr`
within `[sigma_tmin, sigma_tmax]` and add the matching noise before the step.

`AdaptiveHeunSampler` / `AdaptiveStochasticHeunSampler` replace the schedule's
`σ` with the **model-estimated** `σ̂` and derive the step size from how far the
molecule actually moved in EDM-transformed `t` space
(`DiffusionSampler.adjust_step_size`, clamped below by `1/(n_steps−1)`), then
clamp `σ_next` to the schedule value at `step−1`. This is where `n_steps` (the
budget) and `n_steps_target` (the nominal schedule resolution) diverge:
`n_steps = 50, n_steps_target = 10` means "aim for 10-step-sized moves, but
allow up to 50 iterations for molecules that need them".

#### Ancestral samplers

[`src/gpff/sample/ancestral_sampling.py`](../src/gpff/sample/ancestral_sampling.py).
VE-DDPM updates. The adaptive variant additionally overrides
`check_convergence` to converge molecules whose `σ` fell below `1e-5`.

#### Noise schedule

`EDMNoiseSchedule` ([`src/gpff/sample/schedules.py:66`](../src/gpff/sample/schedules.py)):

```
σ_i = (σ_min^{1/ρ} + s_i (σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ,   s uniform on [0,1]
sigmas = [0.0, σ_0, …, σ_{n−1}]                            # index 0 is exactly 0
```

so `get_sigma(n_steps) = σ_max` and `get_sigma(0) = 0`, and the loop counting
down from `n_steps` to `1` lands on `σ_next = 0` at the final step.

`get_t(σ)` is the inverse in EDM-transformed space. `get_time_tensor` — the
time fed to a time-aware *model* — is `σ/σ_max`, matching the training
definition of `t`. Do not confuse the two: `get_t` is for step-size arithmetic,
`get_time_tensor` is for the network.

`sigma_next` snaps to exactly `0` below `0.95 σ_min` so the last step is clean.

### 7.4 Priors

| prior | positions written | requires |
|---|---|---|
| `GaussianPrior(sigma)` | `σ · N(0,I)` for all atoms | — |
| `ShapePrior` | Gaussian, rescaled to prescribed principal variances | `target_variances` |
| `ScaffoldPrior` | scaffold kept; free atoms matched to a target covariance + centre | `scaffold_mask`, `free_covariance`, `free_center` |
| `ScaffoldGaussianPrior(sigma)` | scaffold kept (molecule re-centred on it); free atoms isotropic | `scaffold_mask` |

The centring in `ScaffoldGaussianPrior` is load-bearing: free atoms are drawn
around the origin, and dataset structures are *not* centred, so without
translating each molecule onto its scaffold centroid the frozen scaffold and
the free-atom cloud would live in different frames. `ScaffoldPrior` needs no
such step because it places the free atoms in the scaffold's own frame by
construction.

---

## 8. Conditioning: shape and scaffold

### 8.1 The `conditions` contract

A single `conditions: Dict[str, Tensor]` is built once per batch by a
`ConditionGenerator` and passed to **both** the prior and every `denoise_step`.
Priors and samplers are pure consumers; they never compute conditioning
themselves. This is what lets a prior and a sampler agree on the same target
without either owning it.

| key | shape | ordering |
|---|---|---|
| `keys.target_variances` | `(n_molecules, 3)` descending | `idx_m.unique()` |
| `keys.scaffold_mask` | `(n_atoms,)` bool | atom order |
| `keys.free_covariance` | `(n_molecules, 3, 3)` | `idx_m.unique()` |
| `keys.free_center` | `(n_molecules, 3)` | `idx_m.unique()` |

**`idx_m.unique()` ordering is the invariant that ties the whole system
together.** Every per-molecule tensor — conditions, GMM samples, covariances —
is ordered by it, and helpers like `batch_apply_covariances` accept an explicit
`molecule_ids` argument precisely so that a *subset* of atoms (the free atoms)
can still index rows aligned with the full batch.

### 8.2 Shape conditioning

`ShapeGMM` ([`src/gpff/sample/shape_predictor.py`](../src/gpff/sample/shape_predictor.py))
fits **one Gaussian mixture per atom count** over the log of the three
principal-axis variances of the training structures. Sampling it answers "how
large and how elongated should a molecule with this many atoms be?".

Details that matter:

* Fitted in **log space** so samples are positive; a `variance_floor`
  (`1e-6`) is applied first because planar and linear molecules have a
  vanishing smallest variance that would log to `−inf`.
* Atom counts with fewer than two training structures are skipped with a
  warning; sampling that size later raises.
* Samples are **sorted descending** — the mixture cannot guarantee ordering,
  and every consumer assumes the descending convention.
* `load_or_fit(cache_path=…)` pickles the fitted object. `target_ratio` is
  applied at *sampling* time, so one cache serves every requested shape.

`target_ratio` redistributes the total variance while preserving its sum: the
molecule keeps the size the mixture predicted but is forced into the requested
elongation (`[1,1,1]` spherical, `[8,1,1]` a rod).

Two levels of enforcement:

* `prior=shape` alone — the shape is only set in the prior; denoising is free
  to walk away from it.
* `prior=shape sampler=shape_direct_denoising` — the shape is re-imposed at
  every step with the decaying blend of §7.3.

The sampler-side enforcement uses `preserve_spread=True`, i.e. only the
*ratios* are enforced and each molecule keeps its own total spread,
orientation, and centroid. The prior-side call does not.

### 8.3 Scaffold conditioning

Three generators, all emitting the same three keys:

* `ScaffoldConditionGenerator(scaffold_ids)` — one flat list applied to every
  molecule, or one list per molecule ordered like `idx_m.unique()`.
* `PrecomputedScaffoldConditionGenerator(scaffold_ids_path)` — a JSON
  `{"molecules": [{"dataset_idx": …, "scaffold_ids": […]}, …]}`. Molecules are
  keyed by `properties.idx` (the *dataset* index), so batching and ordering do
  not matter and each molecule gets its own scaffold. **Every sampled structure
  must be covered**, which is why the documented workflow is to pass a split
  file whose test split contains only those molecules.

Both derive `free_covariance` / `free_center` from the *data* positions, so
the resampled free atoms occupy the region they originally did around the
scaffold. Both validate: indices in range, no duplicates, not the whole
molecule (which would leave the free covariance undefined).

### 8.4 Numerical robustness in the geometry helpers

[`src/gpff/utils/alignment.py`](../src/gpff/utils/alignment.py) carries three
guards that exist because sampling actually hit them:

* **`_robust_symeig`** — near-degenerate covariances (repeated principal
  variances, common for symmetric molecules) make the accelerator's `eigh`
  fail to converge. Solves in float64 with trace-scaled jitter, retries on CPU,
  and falls back to `diag(cov)` with an identity basis rather than aborting a
  run.
* **Non-finite guard in `batch_match_principal_variances`** — a molecule whose
  denoising diverged has non-finite positions; it is skipped (and dropped later
  as invalid) instead of taking the whole batch down. Strongly elongated
  targets like `[0.9, 0.05, 0.05]` are what push molecules there.
* **Cholesky failure warning in `batch_apply_covariances`** — keeps the original
  positions for that one molecule.

The philosophy is consistent and worth preserving: **one pathological molecule
must never abort a 10 000-sample run.** It should be dropped by the validity
filter instead.

---

## 9. Evaluation

Evaluation is deliberately split into two phases with a cached intermediate,
because phase 1 is expensive and GPU-bound while phase 2 is cheap and gets
re-run whenever a metric definition or a reference set changes.

```
samples.db ──[gpff_prepare_metadata, GPU]──► evaluation/metadata/*.npy, smiles.txt
                                                        │
                                             [gpff_calculate_metrics, CPU]
                                                        ▼
                                              evaluation/results.json
```

The samples database is only ever **read**. Both phases default the evaluation
folder to `<samples.db dir>/evaluation`.

### 9.1 Phase 1 — `gpff_prepare_metadata`

Writes one file per quantity. `--targets` selects; a *pass* that produces
several arrays runs whole, but only targeted arrays are written, so untargeted
files already on disk survive.

| group | arrays | needs |
|---|---|---|
| RDKit pass | `rdkit_valid`, `ring_counts`, `elongation`, `nfe`, `smiles` | nothing but the DB |
| PoseBusters pass | `posebusters_valid` | nothing (much slower) |
| predicted | `gap` | `--gap_model_path` |
| predicted | `energy_U0` | `--energy_u0_model_path` + `--atomrefs_path` |

`missing_inputs` is checked **before any work starts** — an eight-hour
PoseBusters pass never runs only to fail on a missing model path.

Two compatibility shims live here and are worth knowing about:

* `_patch_legacy_painn_attrs` backfills `epsilon` and
  `return_vector_representation` onto unpickled PaiNN instances that predate
  SchNetPack reading those attributes in `forward`.
* `_ensure_spk_dataset_db_metadata` backfills `_property_unit_dict` and
  `_distance_unit` so a plain sampled database loads as an `ASEAtomsData`.

`_copy_db_to_workdir` mirrors the database into a local scratch directory
(hash-named, size/mtime-checked) to keep repeated reads off a network share.

`energy_U0` subtracts single-atom reference energies via `scatter_add` over
`idx_m` so the reported value is an atomisation-style energy comparable to the
reference distribution.

### 9.2 Phase 2 — `gpff_calculate_metrics`

Reduces the cached arrays. Results are **merged** into `results.json`, so
metrics computed in earlier runs are preserved.

| metric | over | needs |
|---|---|---|
| `rdkit_validity` | all molecules | `rdkit_valid` |
| `posebusters_validity` | all molecules | `posebusters_valid` |
| `uniqueness` | valid only | `smiles` |
| `novelty` | valid only | `smiles`, `--reference_smiles_path`, (`--split_path`) |
| `mean_tanimoto_similarity` | valid only | `smiles` |
| `js_divergence_{energy_U0,gap,elongation}` | valid only | the array + its reference `.npy` |

Conventions to preserve:

* Validity is over **all** molecules; everything else is over the **valid**
  subset. A metric with no valid molecules is `nan`, and `n_molecules` is
  always reported.
* Novelty defaults to `--novelty_reference train_val`: a molecule counts as
  novel if it is absent from the model's **train + val** splits (test was never
  seen). `full` compares against the whole reference set.
* SMILES are **isomeric** (stereochemistry kept), and empty for invalid
  molecules — so empty SMILES are never unique and never novel.
* Tanimoto uses the topological RDKit fingerprint (`maxPath=5`, 2048 bits),
  which is **stereo-blind** — deliberately a different notion of identity than
  the isomeric SMILES used for uniqueness.
* The reference data comes entirely from **files**, never a reference database.
  `gpff_reference_assets` regenerates those files from a reference QM9 database
  so the assets are reproducible. Reference SMILES keep one line per molecule
  *including empty lines for invalid ones*, so line positions still line up
  with split-file indices.

`missing_reference_inputs` again fails fast; a missing per-sample *array* is
caught as a `KeyError` inside `compute_metrics` and reported as "run
`gpff_prepare_metadata` for it first".

### 9.3 Validity definition

`analyze_with_rdkit` ([`src/gpff/analysis/structure.py`](../src/gpff/analysis/structure.py))
is the single definition of chemical validity used by the dataset filter, the
training-time sampler callback, the sampling progress log, and the metrics:

1. positions → XYZ → `Chem.MolFromXYZBlock`
2. `rdDetermineBonds.DetermineBonds(charge=0, allowChargedFragments=False, embedChiral=True)`
3. reject radicals; reject fragments (`"."` in the canonical SMILES)
4. valid → ring counts (3/4/5/6-membered) and elongation (max pairwise distance)

Wrapped in a `SIGALRM` timeout (default 5 s) because bond perception can hang
on pathological geometries, and in `BlockLogs()` because RDKit is loud. The
`SIGALRM` mechanism means **this function is main-thread, single-process only** —
it cannot be moved into a thread pool as written.

---

## 10. Configuration system

Hydra, config root `src/gpff/configs`, packaged via
`[tool.setuptools.package-data]`.

### 10.1 Groups

```
train.yaml         run/ trainer/ callbacks/ task/{optimizer,scheduler}/ logger/
                   model/{representation/{radial_basis/}} data/ prior/
                   generative_transform/{pseudo_pes/{geometry_sampler,
                                         energy_function,weight_generator}/}
                   experiment/ mode/

generate.yaml      sampler/{calculator/,noise_schedule/} prior/ condition_generator/
```

`train.yaml` sets `model`, `data`, `prior`, `generative_transform` and
`experiment` to `null` in its defaults — an experiment config is expected to
`override` them. Both entry points add `$PWD` and `$PWD/configs` to the Hydra
search path, so a user's own `experiment/my_exp.yaml` works without touching
the package.

### 10.2 Custom resolvers

Registered in [`src/gpff/cli/__init__.py`](../src/gpff/cli/__init__.py):

| resolver | use |
|---|---|
| `${petname:1}` | run id (cached — one id per process) |
| `${mul:a,b}` | `cutoff: ${mul:${globals.sigma_max}, 5}` |
| `${key:name}` | `${key:forces}` → `gpff.keys.forces`; keeps YAML in sync with `keys.py` |
| `${uuid:…}`, `${tmpdir:…}` | available, unused by released configs |

`${key:…}` is the mechanism that stops config strings and batch-dict keys from
drifting apart. Prefer it over a literal whenever a config names a batch key.

### 10.3 The `globals` block

Experiments funnel everything through `globals`, which the rest of the tree
interpolates:

```yaml
globals:
  sigma_max: 30
  cutoff: ${mul:${globals.sigma_max}, 5}      # 150 Å
  n_rbf:  ${mul:${globals.cutoff}, 4}         # 600
  n_atom_basis: 256
  n_interactions: 4
  align_permutation: true
  weights_key: ${key:forces_weight}
  lr: 1e-4
  model_path: "best_model"
```

Note the derived quantities: **changing `sigma_max` changes the cutoff and the
radial basis size**. That coupling is intentional (the receptive field must
cover the prior's spread) and is the reason `sigma_max` is not a free knob.

### 10.4 Sampler config shape

Every sampler config composes a calculator and, for schedule-driven samplers, a
noise schedule:

```yaml
defaults:
  - /calculator@calculator: default
  - /noise_schedule@noise_schedule: edm
  - _self_
_target_: gpff.sample.heun.AdaptiveHeunSampler
n_steps: 50
n_steps_target: 10
observers:
  - _target_: gpff.sample.observers.ProgressReporter
```

Because everything about *inference* belongs to the calculator, model-level
overrides are addressed through it:
`sampler.calculator.device=cpu`, `sampler.calculator.dtype=float64`,
`sampler.calculator.noise_model=true`. `noise_schedule.n_steps: ${..n_steps}`
keeps the schedule length tied to the sampler's budget.

`gpff_generate` instantiates the calculator first (injecting the loaded model)
and passes it into the sampler, so the sampler config's own `calculator` node
is overridden at runtime.

---

## 11. Known gaps, dead code, and sharp edges

Recorded so that a future reader can distinguish "load-bearing" from "left
over". Nothing here is currently breaking a released workflow.

### 11.1 Broken if used

* **`VariancePreservingGeometries` and `MixedGeometries` return 3 values**
  (`x_t, sigma_t, t`) while `Geometries.__call__` unpacks **4**
  (`x_t, sigma_t, t, noise`). Selecting
  `generative_transform/pseudo_pes/geometry_sampler=vp_geometries` or
  `=mixed_geometries` raises immediately. Only `VarianceExplodingGeometries` is
  on the working path. — [`src/gpff/pseudo_pes/geometries.py:22`](../src/gpff/pseudo_pes/geometries.py)
* **`callbacks/wandb_cleanup.yaml`** targets `gpff.callbacks.WandbCleanupCallback`,
  which does not exist.
* **`generative_transform/diffusion.yaml`** interpolates `${diffusion_process}`,
  which no config group provides.
* **`ModelOutput.warmup_steps`** is inert — `global_step` is never incremented
  (§6.2).

### 11.2 Silently ignored

* **`mean_free_noise: true`** in `ve_geometries.yaml` has no effect.
  `VarianceExplodingGeometries` stores the flag on the base class and never
  reads it; only the removed `…Old` variant used it. If mean-free noise is
  wanted (it usually is, for translation-invariant generation), it has to be
  implemented.
* **`GatedEquivariant`'s time-conditioning code is fully commented out**, so
  `include_time` and `time_head` are accepted and ignored. Use
  `TimeAwareEquivariant` for a time-aware force head.
* **`Calculator.update_neighbors` builds the list only if `idx_i` is absent.**
  Correct today (the generate-time datamodule attaches no neighbour transform,
  and a fully connected list stays valid as positions move). It would silently
  reuse a stale, cutoff-limited list if a neighbour-list transform were ever
  added to the sampling datamodule.
* **`AtomisticTask.n_skipped_val_batches`** is logged but never incremented (the
  validation-side skip is commented out).

### 11.3 Unreferenced code

Present, importable, not used by any released config or entry point:

| | |
|---|---|
| `data/transform/scaling.py` | `RemoveVariance` / `AddVariance` — a per-molecule scale-normalisation parameterisation |
| `model/heads.py` | `Atomwise`, `DiffusionTime`, `TimeConditionedGatedEquivariant` (last one not in `__all__`) |
| `pseudo_pes/geometries.py` | `VarianceExplodingGeometriesOld` |
| `pseudo_pes/labels.py` | `PseudoCalculator.get_forces` (autograd path; the transform uses `get_forces_analytic`) |
| `pseudo_pes/weights.py` | `NormWeights` |
| `nn/radial.py` | `ExpGaussianRBF` |
| `utils/wandb.py` | `load_latest_model`, `load_split`, `load_last_state_dict` |
| `configs/model/representation/` | `schnet.yaml`, `so3net.yaml`, `field_schnet.yaml` (the latter needs an undefined `globals.response_properties`) |
| `keys.py` | roughly half the constants (`eps_pred`, `sigma_T`, `sigma_i`, `sigma_t_plus_1`, `n_steps_estimate`, `covariance`, …) |

Also: `src/gpff/task.py:1` has a stray `from token import OP`;
`src/gpff/configs/__init__.py` does `from .data import *` / `from .experiment
import *` on YAML-only directories (works as namespace packages, imports
nothing); `src/gpff/__init__.py` lists `.paper`, `.paper.plotting` and
`.sample.pseudo_ff` among the `matred` aliases — modules that no longer exist,
skipped by the `try/except`; `pseudo_pes.py` and `ancestral_sampling.py` carry
`__main__` demo blocks referencing paths outside the repo.

### 11.4 Documentation drift

* `generate.yaml`'s Hydra help lists `sampler=ddim`, `adaptive_ddim`,
  `euler_maruyama` — none exist. The real set is `direct_denoising`,
  `shape_direct_denoising`, `scaffold_direct_denoising`,
  `shape_scaffold_direct_denoising`, `heun`, `stochastic_heun`,
  `adaptive_heun`, `adaptive_stochastic_heun`, `ve_ddpm`, `ve_adaptive_ddpm`.
* `README.md`'s CLI table omits `gpff_reference_assets` (it is in
  `pyproject.toml` and in `cli/__init__.py`'s docstring).
* `callbacks/sampler.yaml` interpolates `${sampler}` and `${prior}`, but
  `train.yaml` includes neither a `sampler` group nor that callback by default —
  using it requires adding both on the command line.

### 11.5 Performance characteristics

* Several hot paths loop over `idx_m.unique()` in Python:
  `Calculator._compute_std_per_molecule`, `batch_compute_principal_variances`,
  `batch_apply_covariances`, `batch_match_principal_variances`,
  `compute_neighbors`, `ScaffoldGaussianPrior`. All are vectorisable with
  `scatter_reduce` / batched `linalg`.
* The Hungarian alignment is `O(N³)` per training sample per epoch, on CPU
  workers.
* The fully connected graph is `O(N²)` edges with `n_rbf = 600`.
* PoseBusters is by far the slowest evaluation pass, which is why it is a
  separate opt-in target.

---

## 12. Extension recipes

### Add a sampler

1. Subclass `Sampler` (free-form) or `DiffusionSampler` (schedule + `σ` state).
2. Implement `denoise_step(batch, step, conditions=None, **kwargs)`. Use
   `self.calculator.calculate(...)` for every model call so NFE is counted.
3. Route all position writes through `self.update_batch(...)` so converged
   molecules stay frozen.
4. Seed any extra state in `_init_state`; add exit criteria in
   `check_convergence` (or mid-step via `update_convergence` when a converging
   molecule must skip the rest of *this* step).
5. Emit diagnostics with `self._log(step=…, …)` — never `print`.
6. Add `configs/sampler/<name>.yaml` composing
   `/calculator@calculator: default` (+ `/noise_schedule@noise_schedule: edm`).
7. Export it from the module's `__all__` (`sample/__init__.py` re-exports `*`).

If the sampler needs `σ`, decide up front whether it comes from the schedule
(`get_sigma(step)`, works with a noise model) or from the model
(`estimate_sigma=True`, force model only).

### Add a constrained direct-denoising variant

Subclass `DirectDenoisingSampler` and override `preprocess_step` and/or
`postprocess_step` only. Document the `conditions` keys you require in the
class docstring — that docstring is the contract, and the matching
`ConditionGenerator` docstring must list the same keys as *emitted*.

### Add a conditioning mode

1. `ConditionGenerator` subclass with `from_batch(batch) -> Dict[str, Tensor]`.
   Read the **data** positions; `gpff_generate` calls it before the prior runs.
2. New `keys.*` constants for anything new.
3. A `Prior` that consumes them, if the prior needs to change.
4. A sampler that consumes them, if the constraint must survive denoising.
5. Configs under `condition_generator/`, `prior/`, `sampler/`.
6. Order per-molecule tensors by `idx_m.unique()`; pass `molecule_ids`
   explicitly when operating on an atom subset.

### Change the forward process

Everything lives behind `PseudoPESTransform`'s three injected components:

* **`Geometries`** — how `(x_t, σ, t, ε)` are drawn. Return **four** values.
* **`PseudoCalculator`** — the pseudo-energy and its analytic force.
* **`LossWeights`** — per-sample weighting.

Changing the pseudo-PES away from the harmonic well changes the
`x̂₀ = x + F/2` and `σ̂ = std(F/2)` identities, so `Calculator.calculate` must be
updated in lockstep. Those identities are the coupling point between the
forward process and every sampler.

### Add a dataset

Subclass a SchNetPack `AtomsDataModule` the way `QM9Valid` does, add
`configs/data/<name>.yaml` inheriting from `custom`, and point an experiment at
it. `gpff_generate` currently hard-codes `QM9Valid` — generalising sampling to
another dataset means making that instantiation config-driven.

### Add a metric

1. A pure reduction in `analysis/collection.py` (per-set) or
   `analysis/structure.py` (per-structure).
2. If it needs a new per-sample array: add it to the right group constant in
   `prepare_metadata.py` (`RDKIT_KEYS` / `POSEBUSTERS_KEYS` / `PROPERTY_KEYS`)
   and produce it in the corresponding pass.
3. Add the name to `METRICS` in `calculate_metrics.py`, handle it in
   `compute_metrics`, and declare any reference input in
   `missing_reference_inputs` so the run still fails fast.

---

## 13. Invariants worth preserving

Short list of the things that will break quietly if violated.

1. **`x̂₀ = x + F/2` and `σ̂ = std(F/2)`** follow from `E = ‖x_t − x₀‖²`.
   Changing the pseudo-energy invalidates every sampler.
2. **Per-molecule tensors are ordered by `idx_m.unique()`.** Conditions, GMM
   samples, covariances, centres.
3. **Principal variances are descending.** Producers sort; consumers assume.
4. **Conditions are computed before the prior overwrites positions.**
5. **The neighbour list is fully connected** — in training (`AllToAllNeighborList`)
   and at inference (`compute_neighbors(fully_connected=True)`), built once per
   run. A distance cutoff is meaningless at `σ = 30 Å`.
6. **Position updates go through `update_batch`**, so converged molecules stay
   frozen; NFE is incremented only inside `Calculator.calculate`.
7. **One bad molecule must not abort a run.** Guard, warn, skip — let the
   validity filter drop it.
8. **The evaluation reads the samples database only**, and reference data comes
   from files. Phase 2 must stay cheap and re-runnable.
9. **Validity has exactly one definition** (`analyze_with_rdkit`), shared by the
   dataset filter, training callback, sampling log, and metrics.
10. **`transforms` order is `PseudoPESTransform → neighbour list → CastTo32`.**
11. **`sigma_max`, `cutoff` and `n_rbf` are coupled** through the `globals`
    interpolations; changing one means re-deriving the others.
12. **`${key:…}` is the bridge between YAML and `keys.py`.** Use it instead of
    duplicating a key string.

---

## Appendix A — batch dictionary keys

SchNetPack-owned (`schnetpack.properties`): `_idx`, `_idx_m`, `_n_atoms`,
`_positions` (`R`), `_atomic_numbers` (`Z`), `_cell`, `_pbc`, `_idx_i`,
`_idx_j`, `_Rij`, `_offsets`.

GPFF-owned (`gpff.keys`), grouped by who writes them:

| written by | keys |
|---|---|
| `PseudoPESTransform` | `x_0`, `eps`/`noise`, `sigma_curr`/`t`, `energy`, `forces`, `energy_weight`, `forces_weight` |
| `Sampler.denoise` | `converged`, `nfe` |
| `DiffusionSampler._init_state` | `sigma_curr`, `sigma_last`, `step_size` |
| adaptive samplers | `sigma_next`, `sigma_predicted` |
| `ScaffoldDirectDenoisingSampler._init_state` | `x_prior` |
| condition generators | `target_variances`, `scaffold_mask`, `free_covariance`, `free_center` |
| `Calculator.calculate` (returns, does not write) | `score`, `x_0`, `forces`, `noise`, `sigma_curr` |

Declared but unused: `eps_pred`, `sigma_t_plus_1`, `sigma_i`, `sigma_T`,
`sigma_initial`, `n_steps_estimate`, `covariance`, `variance`, `x_t`.

## Appendix B — outputs of a full workflow

```
runs/<project>/<date>/<run.id>/
├── config.yaml            resolved config (config.old.N.yaml on resume)
├── checkpoints/           last.ckpt + best epoch checkpoints
├── best_model             inference model  → gpff_generate model_path=
├── best_model.task        full Lightning task
├── split.npz              → gpff_generate split_path= , novelty reference
└── samples/               SamplerCallback output, if enabled

<results_dir>/
├── samples_<timestamp>.db          + run config in ASE metadata, nfe per row
├── trajectories_<timestamp>.db     if store_trajectories=true
└── evaluation/
    ├── metadata/{rdkit_valid,ring_counts,elongation,nfe,posebusters_valid,
    │             gap,energy_U0}.npy  +  smiles.txt
    └── results.json
```
