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
        │   ├── calculate_metrics.py # gpff_calculate_metrics
        │   └── report.py            # gpff_report
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
| `gpff_prepare_geom_revisited` | Build the GEOM-Drugs dataset of the MiDi split, cleaned as in GEOM-Drugs Revisited (below). Wraps `gpff.cli.prepare_geom_revisited:main()`. |
| `gpff_report` | Figures and an HTML page of evaluated sample sets. Wraps `gpff.cli.report:main()`. |

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

The folder has no `split.npz`, so `data=geom_drugs` (for the TFDS-based folders) fails on it
instead of reading the wrong conformers, and the other way round. The source pickles carry
no energies: "first 5" is the order of the pickle, which the MiDi paper calls the 5 lowest
energy conformers.

To diffuse atom types together with positions on this dataset, use
`experiment=gpff_types_geom_drugs`. It combines the categorical type head with
the scalable PaiNN graph and radial-basis options. The vocabulary is
H, B, C, N, O, F, Si, P, S, Cl, Br, I, Bi: all 13 elements recorded by the
canonical dataset `f71e7ef19df1bd58`, with hydrogens included. The default view
is `geom_drugs_revisited_top5`; use `data=geom_drugs_revisited_all` for all conformers.

```bash
gpff_train experiment=gpff_types_geom_drugs \
  run.data_dir=data/geom_drugs/f71e7ef19df1bd58 \
  model/representation/radial_basis=hybrid +callbacks=milestones
```

The corruption is Gaussian noise on scaled one-hot atom types, trained with
cross-entropy against the clean elements. The absorbing-state design in
`docs/absorbing_atom_types.md` remains a proposal. The model reads the noisy
types, and permutation alignment matches positions and types jointly.
The original `gpff_types` (MSE) and `gpff_types_categorical` (cross-entropy)
experiments retain the five-element QM9 vocabulary and also accept the scalable
radial-basis, filter and global-channel options.

Generation must use the training vocabulary in the same order and a type prior
of the same width. For the GEOM experiment, pass the following in addition to
the model, database and matching split paths to `gpff_generate`:

```bash
gpff_generate model_path=/path/to/best_model \
  datapath=/path/to/f71e7ef19df1bd58/geom_drugs.db \
  split_path=/path/to/f71e7ef19df1bd58/split_top5.npz \
  prior.n_types=13 '+sampler.atom_types=[1,5,6,7,8,9,14,15,16,17,35,53,83]'
```

Training logs performance metrics by default through the configured logger
(including W&B): epoch/train/validation seconds, molecules per second, examples
seen, CUDA allocated/reserved peaks in GiB, and batch-loss extrema/finite counts.
The optional learned output gain is also recorded. Metrics use `perf/`, `train/`,
and `model/` prefixes. Epoch timing includes validation and preceding callback
overhead; train timing subtracts validation, including repeated validation passes.
The first epoch includes compilation; compare later epochs for steady throughput.
Counts resume from checkpoints, and metrics are per device. Disable with
`~callbacks.performance_metrics`.

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

# Control the update size of the weight matrices of the equivariant blocks
# (gpff.optim): AdamW's update divided by its RMS or spectral norm, or Muon's
# update. The learning rate then sets the size of each matrix's update, so it
# needs its own search; starting points are in configs/task/optimizer/block_adam.yaml.
# task.optimizer_args.block_update is rms, spectral, muon, soap (Adam in the eigenbasis of Shampoo's
# preconditioner, refreshed every task.optimizer_args.precondition_frequency steps) or adam (control);
# task.block_scope is all matrices of the blocks or only the vector ones (all, vector).
# task.block_slices=true steps the roles of the fused layers (2 or 3 outputs computed by one weight) on their own.
# task.optimizer_args.one_sided=true makes soap rotate only the smaller side of every matrix (cheaper refreshes).
# task.optimizer_args.compiled=true compiles the update with torch.compile (same steps, fewer kernels).
# task.optimizer_args.weight_decay (default 0, off) decays those matrices only:
gpff_train experiment=gpff task/optimizer=block_adam task.optimizer_args.block_update=rms globals.lr=1e-5

# Scale to larger molecules (all off by default; experiment=gpff_scalable without an
# override is experiment=gpff). A radial basis of far fewer functions than the 4 per A of
# the cutoff: hybrid (0.25 A apart up to 6 A, then a ratio of 1.25 up to 300 A, 42 functions),
# log (32 functions from 0.5 A to 300 A) or uniform with globals.n_rbf and globals.rbf_cutoff.
# A two-layer MLP between the basis and the filters instead of one linear layer, and a
# global channel (a virtual node per molecule with its size and shape) in every interaction block:
gpff_train experiment=gpff_scalable model/representation/radial_basis=hybrid \
  globals.filter_hidden=64 globals.global_channel=true
# A neighbor graph built inside the model, on the GPU, from the current positions: in training
# and at every step of a sampler (gpff_generate builds no neighbor list for such a model).
# knn is the symmetric union of the k nearest neighbors; a pair weighs
# cosine_cutoff(d_ij / max(rho_i, rho_j)), rho_i the distance to the (k+1)-th neighbor, so a
# pair leaves the graph with weight 0. size_radius takes all pairs within
# graph_radius * R_g(x_t) / R_g,ref(N), R_g,ref(N) = c N^e (globals.rg_ref_coefficient/exponent):
gpff_train experiment=gpff_scalable_graph globals.knn_k=16
gpff_train experiment=gpff_scalable_graph model/representation/graph=size_radius

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
  --reference_bond_lengths_path assets/qm9_bond_lengths.npz \
  --reference_bond_angles_path assets/qm9_bond_angles.npz \
  --reference_smiles_path assets/qm9_smiles_values.txt \
  --split_path /path/to/split.npz

# Just the structure metrics, no reference data or models needed:
gpff_prepare_metadata --samples_db_path samples.db \
  --targets rdkit_valid posebusters_valid smiles
gpff_calculate_metrics --samples_db_path samples.db \
  --metrics rdkit_validity posebusters_validity uniqueness mean_tanimoto_similarity

# Bond lengths, each bond against the reference bonds of its own chemical
# environment (e.g. C(sp2)-C(sp3)); the reference bond table is built once with
# gpff_reference_assets --targets bond_lengths. Also writes the per-environment
# breakdown to evaluation/bond_length_w1.json:
gpff_prepare_metadata --samples_db_path samples.db --targets rdkit_valid bond_lengths
gpff_calculate_metrics --samples_db_path samples.db --metrics bond_length_w1 \
  --reference_bond_lengths_path assets/qm9_bond_lengths.npz

# Bond angles work the same way, in degrees, with the central atom's
# hybridization as the environment (e.g. C-C(sp3)-C):
gpff_prepare_metadata --samples_db_path samples.db --targets rdkit_valid bond_angles
gpff_calculate_metrics --samples_db_path samples.db --metrics bond_angle_w1 \
  --reference_bond_angles_path assets/qm9_bond_angles.npz

# What a relaxation has to move is the geometry the model got wrong, as opposed
# to the molecule it chose: relax the samples (uff, xtb or a MACE model on the
# GPU) and compare the two databases with the same metrics. mean_strain is the
# energy the relaxation released:
# (relax the reference molecules the same way and build the reference tables
# from those: every method pulls structures toward its own minima, UFF by some
# 40 mA even for real QM9 geometries)
gpff_relax --samples_db_path samples.db --output_db_path relaxed.db --method uff
# or, with mace-torch installed (pip install -e .[mace]), many molecules at a
# time on the GPU, which is the fastest and the most accurate of the three:
# gpff_relax ... --method mace --mace_model_path MACE-OFF24_medium.model
gpff_prepare_metadata --samples_db_path relaxed.db --targets rdkit_valid strain bond_lengths
gpff_calculate_metrics --samples_db_path relaxed.db --metrics mean_strain bond_length_w1 \
  --reference_bond_lengths_path assets/qm9_bond_lengths.npz

# Why molecules are invalid (failure_<cause> shares), valid+unique+novel, the
# elongation W1 and, for models that generate the atom types, the composition
# against the test molecules the samples came from (the reference element counts
# come from gpff_reference_assets --targets element_counts). Each call also notes
# in evaluation/provenance.json which code computed every array and metric:
gpff_prepare_metadata --samples_db_path samples.db \
  --targets rdkit_valid failure smiles elongation element_counts
gpff_calculate_metrics --samples_db_path samples.db \
  --metrics rdkit_validity failures vun elongation_w1 composition \
  --reference_smiles_path assets/qm9_smiles_values.txt \
  --reference_elongation_path assets/qm9_elongation_values.npy \
  --reference_element_counts_path assets/qm9_element_counts.npz \
  --split_path /path/to/split.npz

# Figures and an HTML page of evaluated models, each <run>/frozen/repro/<model>
# with <config>/seed<N>/evaluation/ folders; --control adds the paired
# differences of the others to it:
gpff_report runs/a/frozen/repro/epoch_5000_model runs/b/frozen/repro/epoch_5000_model \
  --control a:5000 --output_dir report/
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

