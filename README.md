# Generative Pseudo-Force Field (GPFF)

Reference implementation of the Generative Pseudo-Force Field (GPFF) model and
samplers for molecular structure generation on QM9.

## Layout

```
gpff/
├── pyproject.toml
├── LICENSE
└── src/
    └── gpff/                       # the importable package
        ├── cli/                    # one module per console entry point
        │   ├── __init__.py         # shared banner, torch.load shim, OmegaConf resolvers
        │   ├── train.py            # gpff_train
        │   ├── generate.py         # gpff_generate
        │   ├── prepare_metadata.py  # gpff_prepare_metadata
        │   └── calculate_metrics.py # gpff_calculate_metrics
        ├── callbacks.py
        ├── diagnostics.py          # force-field diagnostics every N training samples
        ├── keys.py
        ├── task.py
        ├── analysis/               # validity, metrics, DB metadata, force-field diagnostics
        ├── configs/                # Hydra config tree
        ├── data/                   # data modules, datasets (QM9), transforms
        ├── model/                  # network heads (PaiNN, gated equivariant outputs)
        ├── nn/                     # neural-network building blocks
        ├── pseudo_pes/             # Gaussian pseudo-PES used to train GPFF
        ├── sample/                 # samplers (DD, Heun, ancestral, adaptive), priors, shape predictors
        └── utils/                  # generic helpers
```

## Installation

```bash
git clone https://github.com/stefaanhessmann/gpff.git
cd gpff
pip install -e .
```

`pip install -e .` resolves all standard dependencies plus the pinned
`schnetpack` git URL declared in `pyproject.toml`.

Python 3.13 only (`>=3.13,<3.14`): the numerical stack is pinned as a set and
held to a single interpreter version.
CUDA-capable GPU recommended for training and large-scale sampling.

## Command-line interface

After install the entry points are on `PATH`:

| Command          | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| `gpff_train`     | Train a model. Wraps `gpff.cli.train:train()`.       |
| `gpff_generate`  | Sample molecules from a checkpoint. Wraps `gpff.cli.generate:sample_molecules()`. |
| `gpff_prepare_metadata`  | Compute the per-sample arrays of a sampled DB into `evaluation/metadata/`. Wraps `gpff.cli.prepare_metadata:main()`. |
| `gpff_calculate_metrics` | Reduce those arrays to `evaluation/results.json`. Wraps `gpff.cli.calculate_metrics:main()`. |
| `gpff_prepare_geom_revisited` | Build the GEOM-Drugs dataset of the MiDi split, cleaned as in GEOM-Drugs Revisited (below). Wraps `gpff.cli.prepare_geom_revisited:main()`. |

`gpfftrain` still works as a deprecated alias for `gpff_train`.

### GEOM-Drugs (Revisited)

`gpff_prepare_geom_revisited --source_dir <train/val/test_data.pickle of the MiDi split>
--data_root data/geom_drugs` writes one dataset folder: `geom_drugs.db` (every cleaned
conformer once, hydrogens included, with `smiles`, `mol_id` and `conf_rank`), the two split
files `split_all.npz` (all conformers) and `split_top5.npz` (the first 5 of each molecule),
and `manifest.json` (source files, cleaning, counts, sha256s). The molecules are cleaned by
the script of GEOM-Drugs Revisited (Nikitin et al. 2025), vendored unchanged. Train on a
view with its data config and the folder:

```bash
gpff_train experiment=gpff data=geom_drugs_revisited_top5 run.data_dir=data/geom_drugs/<id>
gpff_train experiment=gpff data=geom_drugs_revisited_all  run.data_dir=data/geom_drugs/<id>
```

Both configs set no split sizes, so on a folder without their split file (such as the older
TFDS-based GEOM-Drugs builds) they fail instead of drawing a random split. The source
pickles carry no energies: "first 5" is the order of the pickle, which the MiDi paper calls
the 5 lowest energy conformers.

Training logs performance metrics by default through the configured logger
(including W&B): epoch/train/validation seconds, molecules per second, examples
seen, CUDA allocated/reserved peaks in GiB, and batch-loss extrema/finite counts.
The optional learned output gain is also recorded. Metrics use `perf/`, `train/`,
and `model/` prefixes. Epoch timing includes validation and preceding callback
overhead; train timing subtracts validation, including repeated validation passes.
The first epoch includes compilation; compare later epochs for steady throughput.
Counts resume from checkpoints, and metrics are per device. Disable with
`~callbacks.performance_metrics`.

For post-hoc EMA tuning, train with `'+callbacks=[milestones,posthoc_ema]'` (one
override for both; Hydra rejects two `+callbacks=` options) and use
`python -m gpff.cli.posthoc_ema` to reconstruct inference models afterward.
See [the method, hydra workflow, and GPFF design notes](docs/posthoc_ema.md).

Two opt-in callbacks log diagnostics of a time-agnostic force model
(`experiment=gpff`, `gpff_no_permutation`) next to `val/loss`. They are built on
`gpff.diagnostics.EveryNSamples`, which counts the training molecules seen and
runs in the first validation after every `every_n_samples` of them rather than
every epoch (the default 5,000,000 is about 50 epochs of QM9, 98.9k training
molecules; scale it for other datasets). On the other epochs it costs one
integer addition per training batch and no device synchronization; the count
resumes from checkpoints.

Hydra takes one `+callbacks=` override, so to add them to e.g. the milestones
use the list form (quoted for the shell):
`'+callbacks=[milestones,denoising_error,sigma_calibration]'`.

- `denoising_error` logs the denoising error $e(\sigma)$ in units of
  the noise at fixed noise levels, `val/e_sigma_<sigma>`, and its integral over
  $\log \sigma$, `val/e_log_sigma_integral`. A perfect model approaches
  $1/(N-1)$ at small $\sigma$. It re-noises the validation molecules (aligned
  like the training transform; solved once, kept in host memory) and costs one
  forward pass per $\sigma$ and validation batch.
- `sigma_calibration` logs how well $\hat\sigma = \operatorname{std}(\hat F/2)$,
  the noise level the adaptive samplers use, matches the std of its target
  $x_0 - x_t$: `val/sigma_hat_ratio_<lo>_<hi>` (geometric mean of the ratio per
  noise level bin, 1 is calibrated) and `val/sigma_hat_log_ratio_mean` and
  `_rms`. It costs one extra forward pass per validation batch.

Both are only logged; checkpoints are still selected by `val/loss`.

Hydra configs live under `src/gpff/configs/`. Override with standard dotted
overrides. Everything about model inference — the model itself, `device`,
`dtype`, and whether the model predicts noise or forces — belongs to the
sampler's calculator, so those overrides are addressed as
`sampler.calculator.<key>` (e.g. `sampler.calculator.noise_model=true`):

```bash
gpff_train experiment=gpff data=qm9_valid trainer.max_epochs=300

# Runs are seeded from `seed` (default 999); vary it to repeat an experiment.
# The split stays the same: only data.splitting.seed (default 999, the split
# of the paper's baseline) changes it.
gpff_train -m experiment=gpff seed=0,1,2

# Resume an interrupted run from its last checkpoint by passing its id and path:
gpff_train experiment=gpff run.id=<run-name> run.path=runs/<project>/<date>

# The default force weight is 1/sigma^2 clipped at 1, which flattens below 1 Å, so
# the loss of nearly clean draws fades like sigma^2. Two weights that stay finite
# at sigma = 0 and keep a gradient there (sigma_floor, default 0.03, sets where they level off):
#   floored_sigma_weights: 1/(sigma^2 + sigma_floor^2)
#   soft_sigma_weights:    1/sigma^2 above sigma_knee (default 1), 1/(sigma * sigma_knee) below it
gpff_train experiment=gpff generative_transform/pseudo_pes/weight_generator=floored_sigma_weights
gpff_train experiment=gpff generative_transform/pseudo_pes/weight_generator=soft_sigma_weights

# Replace shares of the noise draws by sigma = 0 anchors (x_t = x_0, zero-force label) and by
# log-uniform draws from small_sigma_range (default [0.01, 0.1]). The weight has to be finite
# at sigma = 0: the two above are, and so is the default clipped at 1, but SigmaWeights
# without max_clipping is not:
gpff_train experiment=gpff generative_transform/pseudo_pes/weight_generator=soft_sigma_weights \
  pseudo_pes.geometry_sampler.anchor_fraction=0.05 pseudo_pes.geometry_sampler.small_sigma_fraction=0.2

# Also keep the model after 100, 200, 500, 1k, 2k, 5k, 10k and 15k epochs in
# milestones/ (epoch_<N>.ckpt for Lightning, epoch_<N>_model as the model_path
# for gpff_generate), regardless of early stopping and the best epoch. Off by
# default; the second line picks other epochs:
gpff_train experiment=gpff +callbacks=milestones
gpff_train experiment=gpff +callbacks=milestones 'callbacks.milestone_checkpoint.epochs=[50,100]'

# Train faster (all off by default): skip the QM9 property columns gpff does
# not use, serve the database from memory, keep the DataLoader workers between
# epochs (needs per-sample training noise, a different but equally valid noise
# stream) and compile the model. torch.compile needs a C and a C++ compiler at
# run time, found through CC/CXX or as gcc/g++ on PATH:
gpff_train experiment=gpff data.load_properties=[] data.cache_in_memory=true \
  data.persistent_workers=true callbacks.deterministic_data_seed.per_sample_train_noise=true \
  'compile={dynamic:true}'

# Control the update size of the weight matrices of the equivariant blocks
# (gpff.optim): AdamW's update divided by its RMS or spectral norm, or Muon's
# or SOAP's update. The learning rate then sets the size of each matrix's update, so it
# needs its own search; starting points are in configs/task/optimizer/block_adam.yaml.
# task.optimizer_args.block_update is rms, spectral, muon, normuon (Muon with every row normalized by its
# running RMS), muonh (Muon on a hypersphere, see below), soap (Adam in the eigenbasis of Shampoo's
# preconditioner, refreshed every
# task.optimizer_args.precondition_frequency steps; one_sided=true rotates only the smaller side) or adam (control);
# task.block_scope is all matrices of the blocks or only the vector ones (all, vector).
# task.block_slices=true steps the roles of the fused layers (2 or 3 outputs computed by one weight) on their own.
# task.optimizer_args.orthogonalization=polar_express gives muon and normuon Polar Express's orthogonalization
# (default newton_schulz).
# task.optimizer_args.weight_decay (default 0, off) decays those matrices only:
gpff_train experiment=gpff task/optimizer=block_adam task.optimizer_args.block_update=rms globals.lr=1e-5

# MuonH ("Hyperball Optimization"): Muon's direction with the norm of the weight it updates, on weights
# held at a fixed norm, so that globals.lr is the fraction of a weight's norm that a step moves it. The fixed
# norm is globals.weight_norm=true (EDM2's normalization: the rows of every linear layer are projected back
# onto their sphere after each step; the recipe gpff_stable sets it). The task refuses block_update=muonh
# without it, and weight_decay with it (the projection undoes the decay). The other block updates also run on
# weight-normalized layers; muon, normuon and rms keep their own update size there.
gpff_train experiment=gpff_stable task/optimizer=block_adam task.optimizer_args.block_update=muonh

# Gate the messages an atom receives from its neighbors by an invariant number that reads
# the scalars of both atoms and their distance (all its weights start at zero: a gate of one):
gpff_train experiment=gpff globals.neighbor_gate=true

# Sample with a local checkpoint; model_path and split_path are required and
# are the only way to point gpff_generate at a model:
gpff_generate \
  sampler=stochastic_heun \
  model_path=/path/to/best_model split_path=/path/to/split.npz \
  datapath=/path/to/qm9_filtered.db \
  n_samples=10000
# (add 'compile={dynamic:true}' to compile the model, as for training)

# The shape prior draws positions with a prescribed covariance per molecule, so
# it needs a condition generator to supply one variance vector per molecule:
gpff_generate \
  sampler=direct_denoising prior=shape condition_generator=shape \
  condition_generator.shape_predictor.cache_path=/path/to/shape_gmm.pt \
  model_path=/path/to/best_model split_path=/path/to/split.npz \
  datapath=/path/to/qm9_filtered.db n_samples=10000

# Use sampler=shape_direct_denoising to keep the sampled shape through
# denoising rather than only setting it in the prior:
gpff_generate \
  sampler=shape_direct_denoising prior=shape condition_generator=shape \
  condition_generator.shape_predictor.cache_path=/path/to/shape_gmm.pt \
  model_path=/path/to/best_model split_path=/path/to/split.npz \
  datapath=/path/to/qm9_filtered.db n_samples=10000

# Direct denoising jumps to the predicted clean positions, which is only accurate at
# low noise. sampler=adaptive_direct_denoising walks down an EDM grid of noise levels
# first (transport, per molecule from its own sigma estimate), then takes over-relaxed
# jumps without the model's net force and torque until the forces are below f_max.
# The options are in configs/sampler/direct_denoising.yaml; a jump optimizer moves the
# noise-free jumps (gpff.sample.jump_optimizers, measured in its docstring), e.g.
gpff_generate \
  sampler=direct_denoising sampler.n_steps=16 sampler.stochastic_lambda=0 \
  sampler.transport_sigma=1.0 sampler.transport_ratio=0.25 \
  'sampler.jump_optimizer={_target_:gpff.sample.jump_optimizers.MDMin}' \
  model_path=/path/to/best_model split_path=/path/to/split.npz \
  datapath=/path/to/qm9_filtered.db n_samples=10000

# Train on straight paths from each molecule to noise of its own size and shape
# instead of variance-exploding noise. Their noise is distributed like the shape
# prior, so sample such a model with prior=shape condition_generator=shape and a
# direct denoising sampler as above; the Heun and DDPM samplers assume
# variance-exploding noise.
gpff_train experiment=gpff generative_transform/pseudo_pes/geometry_sampler=shape_geometries

# Evaluate a sampled DB in two phases, both writing into an evaluation folder
# next to the database (unless --evaluation_dir says otherwise); the database
# itself is only read. No reference database is used — the reference
# distributions, SMILES and atom references all come from files in assets/.
#
# Phase 1 (GPU): the per-sample arrays the figure scripts need, into
# evaluation/metadata/ (.npy per quantity, smiles.txt). --targets picks which
# arrays (default: all); gap/energy_U0 are predicted and need their models.
gpff_prepare_metadata \
  --samples_db_path samples.db \
  --gap_model_path assets/best_model_gap \
  --energy_u0_model_path assets/best_model_energy_U0 \
  --atomrefs_path assets/qm9_energy_U0_atomrefs_eV.npy \
  --device cuda

# Phase 2 (CPU, cheap and re-runnable): reduce the cached arrays to
# evaluation/results.json. --metrics picks the metrics (default: all); each
# metric's inputs are checked up front — js_divergence_* need the reference
# distribution files, novelty needs the reference SMILES and, for train+val,
# the split file. Novelty is measured against train+val by default
# (--novelty_reference full compares against the whole reference set instead).
gpff_calculate_metrics \
  --samples_db_path samples.db \
  --reference_energy_U0_path assets/qm9_energy_U0_values.npy \
  --reference_gap_path assets/qm9_gap_values.npy \
  --reference_elongation_path assets/qm9_elongation_values.npy \
  --reference_smiles_path assets/qm9_smiles_values.txt \
  --split_path /path/to/split.npz

# Just the structure metrics, no reference data or models needed:
gpff_prepare_metadata --samples_db_path samples.db \
  --targets rdkit_valid posebusters_valid smiles
gpff_calculate_metrics --samples_db_path samples.db \
  --metrics rdkit_validity posebusters_validity uniqueness mean_tanimoto_similarity
```

## Citation

If you use this code, please cite:

Hessmann, S. S. P., Kahouli, K., Gugler, S., Plainer, M., Noé, F.,
Müller, K.-R., & Gebauer, N. W. A. (2026). *Generative Pseudo-Force Fields for
Molecular Generation.* arXiv:2605.19050.

```bibtex
@article{hessmann2026generative,
  title={Generative Pseudo-Force Fields for Molecular Generation},
  author={Hessmann, Stefaan Simon Pierre and Kahouli, Khaled and Gugler, Stefan and Plainer, Michael and No{\'e}, Frank and M{\"u}ller, Klaus-Robert and Gebauer, Niklas Wolf Andreas},
  journal={arXiv preprint arXiv:2605.19050},
  year={2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

