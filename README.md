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
        ├── keys.py
        ├── task.py
        ├── analysis/               # validity, metrics, DB metadata
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

`gpfftrain` still works as a deprecated alias for `gpff_train`.

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

# Sample with a local checkpoint; model_path and split_path are required and
# are the only way to point gpff_generate at a model:
gpff_generate \
  sampler=stochastic_heun \
  model_path=/path/to/best_model split_path=/path/to/split.npz \
  datapath=/path/to/qm9_filtered.db \
  n_samples=10000

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

