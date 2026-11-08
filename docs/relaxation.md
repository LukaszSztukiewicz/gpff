# Relaxation studies

Run these commands in the same container, code snapshot and dataset protocol
as the experiment. GPU commands require a GPU allocation. None of the commands
downloads models or installs packages.

With an existing container and a snapshot supplied through `PYTHONPATH`, the
new console scripts may not be registered in the container. Use
`python -m gpff.cli.relax_report`, `python -m gpff.cli.refine` and
`python -m gpff.cli.damping_probe` with the same arguments in that case.

## Paired physical relaxation

```sh
gpff_relax --samples_db_path samples.db --output_db_path relaxed.db --method uff
gpff_relax_report --samples_db_path samples.db --relaxed_db_path relaxed.db \
  --output_dir relaxation_report
```

UFF uses perceived bond orders first and falls back to connectivity only if
that setup fails. Each output row records `bond_orders`; its energies are not
interchangeable between those modes. UFF keeps its assigned graph, while xTB
and MACE may move a structure into a different perceived graph.

The paired report writes `summary.json`, `summary.md`, `pairs.npz` (one value
per source row, readable with `allow_pickle=False`) and input/code provenance.
It checks row count and atom order. New `gpff_relax` outputs also record a
source database hash and source row IDs; those are checked before pairing.
For older outputs the report explicitly labels pairing as `atom_order_only`:
matching formulas cannot prove row identity or the order of equal elements.
Canonical isomeric SMILES changes count stereochemical changes too.

Counts use all input rows, including failed setups. Strain, raw RMSD and
RMSD after optimal proper rotation/translation are summarized over finite
values from successful setups, with mean, median and p90. No atom permutation
is attempted. Strain is in kcal/mol, displacement in Angstroms. Cohorts are
`all`, `initially_valid`, `finally_valid`, `stayed_valid`, `same_identity`,
`gained` and `lost`. Each reports its size, successful setups, convergence,
and strain among converged setups separately. A failed setup contributes no
spurious zero to the statistics. Missing statistics are JSON `null`.

`gpff_calculate_metrics` retains its existing `mean_strain` definition:
validity is measured in the database being evaluated (after relaxation for
`relaxed.db`). Use the paired report to select an initial-validity cohort.

Relax reference molecules with the **same method and settings** and build
the relaxed reference assets from them before computing distribution scores.
Raw samples should still be evaluated against the raw reference. An
independent real-molecule set from the same dataset and protocol estimates
finite-sample variability; comparison of a reference with itself is zero by
construction. Record the exact database ID, split, code, model and settings.

## Distribution diagnostics

The existing `bond_length_w1.json` and `bond_angle_w1.json` breakdowns now
include `reference_std` and `w1_over_std` per environment. The normalized W1
is dimensionless; 1 means a distance of one reference standard deviation.
It supplements the original distances and uses their existing grouping,
minimum counts, label-version checks and reference weights.

The top-level `w1_over_std` averages normalized distances by reference counts.
Labels with reference std at most $10^{-12}$ are excluded from normalization
only. `reference_weight_normalized` and `observed_fraction_normalized` report
its coverage; the original W1 and its coverage still include those labels.
Do not compare normalized scores with substantially different coverage.

## Saved-sample DD refinement

```sh
gpff_refine --model_path /path/to/force_model --samples_db_path samples.db \
  --output_db_path refined.db --f_max 0.05 --max_steps 200 \
  --batch_size 128 --trace_path shifts.npz --device cuda
```

This uses the existing DD sampler with no noise. The force model must be
time-agnostic and predict the GPFF pseudo-force. It stops a molecule when its
largest atomic force norm is at most `f_max`. A DD update is $F/2$, so this
threshold bounds the proposed atomic step by `f_max / 2` Angstroms. The
terminal force check costs an NFE; reaching `max_steps` is not convergence.

Rows retain their source data and key-value pairs, plus `dd_nfe`,
`dd_converged`, `dd_failed`, and `refinement_source_id`; `nfe` includes both
generation and refinement. Stale physical-relaxation energies and status are
removed. Database metadata records the source/model hashes and settings.
Existing output or trace files are refused. The optional NPZ contains
`max_shift` (steps by molecules), `dd_nfe`, and `dd_converged`. Stopped or
unexecuted steps are NaN, so quantiles can use only molecules still active.

## Damping probe

```sh
gpff_damping_probe --model_path /path/to/force_model --samples_db_path samples.db \
  --output_dir damping_report --etas 1 0.5 1.5 --steps 60 --limit 512 --device cuda
```

For each damping factor, the probe starts from the same saved geometries and
iterates $x \leftarrow x + \eta F/2$ without stopping early. It saves
`traces.npz` (eta by step by molecule; final RMSDs are eta by molecule),
`summary.json` and input/code provenance. Per-step statistics include the
largest atomic displacement, cosine with the previous step, estimated sigma,
translation, and the largest displacement after subtracting the best
infinitesimal rigid rotation/translation. Cosines near -1 indicate alternating
steps, near +1 steady drift. Zero-norm steps have undefined cosine, saved as
NaN and excluded from summaries. Nonfinite trajectories abort the probe.

## MACE numerical validation

`tests/test_mace_relaxation.py` compares batched energies/forces with
`MACECalculator`, including an active subset of unequal-size molecules and
updated positions. It also compares final optimization energies/geometries
with ASE LBFGS. These checks require `mace-torch` and `GPFF_MACE_MODEL` pointing
to an existing compatible model. They use CUDA when available and otherwise
CPU. They do not download assets. An independent quadratic test exercises
per-molecule freezing and step-limit accounting without MACE installed.

The original scripts and reports are retained in
[`studies/uff_relaxation_20261031`](../studies/uff_relaxation_20261031/README.md),
with the original single-structure relaxation module saved alongside them.
