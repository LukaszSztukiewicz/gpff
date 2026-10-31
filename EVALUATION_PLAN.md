# Evaluation suite: implementation plan

Scope: the minimal version of [EVALUATION_SUITE.md](EVALUATION_SUITE.md). The fast tier
gains every free metric, each evaluation records which code computed it, and a report turns
the per-sample arrays into figures. The 2,236 existing sample databases are scored again
without regenerating them.

## Principles

- **Additive only.** No existing metric changes its definition, name or default. Old
  results.json values stay comparable, and the paper tier (`FULL=1`) is left as it is.
- **Keep the current shape.** Every new array goes through `gpff_prepare_metadata`, every
  new number through `gpff_calculate_metrics`, in their explicit if/elif style. No registry
  and no plugin layer.
- **Tiers are shell presets,** not Python. The fast tier is the target and metric lists in
  `slurm/fast_rdkit.sbatch`.
- **No new dependencies.** numpy, matplotlib and rdkit are pinned already, and the HTML is a
  plain string template.
- **One commit per step,** each with tests that run on hydra, and only when you say commit.
  Hydra scripts get a `.bkp-<date>` copy before an edit, as now.
- **Nothing is overwritten on hydra.** The backfill writes new files only and checks the old
  ones.

## Step 0: worktree

`~/STUDIES/gpff-eval-tiers` on a new branch `feat/eval-tiers` from `feat/evaluation-suite`
(1c0509e). The uncommitted trajectory-diagnostics work in `gpff-structural-metrics` is not
touched. The two branches will both add a pyproject entry point and a README section, a
trivial conflict at merge time.

`analysis/composition.py` exists only as an untracked file in `gpff-typed-readout`. Commit it
there, then `git checkout feat/typed-type-readout -- src/gpff/analysis/composition.py` takes
it over unchanged.

## Step 1: per-sample arrays (`structure.py`, `prepare_metadata.py`, `reference_assets.py`)

1. **`analyze_with_rdkit` returns `failure` and `n_fragments`** (~25 lines).
   - `failure` is the first of: `nonfinite` (positions checked up front), `timeout`,
     `bond_orders` (DetermineBonds raised), `radical`, `fragments`, `error`, else `valid`.
   - `n_fragments` is `len(Chem.GetMolFrags(mol))` after DetermineBonds, or -1 when no bonds
     were perceived.
   - The validity code path stays exactly as it is, so `rdkit_valid == (failure == "valid")`
     by construction. A test pins that, and `_invalid_analysis` gains both keys.
2. **New metadata targets in the RDKit pass** (~20 lines):
   - `failure.npy`: a fixed-width str array, so it loads without pickle.
   - `n_fragments.npy`.
   - `element_counts.npz`: one array per element present, one row per sample, read from the
     atoms in the same walk. `load_metadata` already loads any `.npz` as a dict, and the
     length check skips dicts.
3. **`gpff_reference_assets` target `element_counts`**: `<prefix>_element_counts.npz`, one
   row per database molecule in database order, as smiles_values.txt is (~10 lines).

Tests:
- Failure causes on built molecules: methane is valid, two methanes 10 Å apart are
  fragments, CH3 is a radical, a NaN position is nonfinite, and a timeout uses the existing
  monkeypatch.
- The invariant above.
- Element counts on a small DB in tmp_path.

## Step 2: metrics (`calculate_metrics.py`)

New entries in METRICS, defaults unchanged otherwise (~50 lines):

| metric | definition | needs |
|---|---|---|
| `failure_<cause>` | share of all samples per cause; with validity they sum to 1 | failure |
| `vun` | share of all samples that are valid, unique and novel | smiles, novelty reference |
| `elongation_w1` | W1 in Å of the valid samples' elongation to the reference | `--reference_elongation_path`, existing |
| `element_tv`, `size_w1`, `element_fraction_<X>`, `molecule_fraction_<X>`, `heteroatom_fraction` | `composition.composition_metrics` on all samples, against the first n test rows of the reference | element_counts, new `--reference_element_counts_path`, `--split_path` |

Composition takes the atomic numbers rebuilt from the count table (`np.repeat`), so
composition.py needs no change. For untyped runs every composition distance is 0 by
construction, which doubles as a check that rows line up with the split.

Tests:
- Failure shares plus validity equal 1.
- A composition table against itself gives TV 0 and size W1 0.
- `elongation_w1` of a set against itself is 0.

## Step 3: provenance (`cli/__init__.py`, both CLIs)

A helper `record_provenance(evaluation_dir, names, inputs)` merges into
`evaluation/provenance.json` one entry per array or metric written (~25 lines). Each entry
holds:
- the `gpff` package path, which names the frozen_code snapshot;
- the rdkit and posebusters versions;
- the md5 of every reference file and the split;
- n and the time.

It is a sidecar: results.json keeps its flat `{name: value}` shape, so `hy res`,
summarize.py and the study compare.py scripts are unaffected. Test: two calls merge, and a
later write of the same name replaces its entry.

## Step 4: report (`analysis/report.py`, `cli/report.py`)

The command, ~250 lines including the template:

```
gpff_report <model dir>... [--names A B ...] [--control A] [--seed_band 1.5] --output_dir DIR
```

A model dir is `<run>/frozen/repro/<model>`, as in paired_compare.py; repo code takes
paths, and `hy` maps names. The report reads `evaluation/{results.json, metadata/,
bond_length_w1.json, bond_angle_w1.json}` and writes PNGs and `index.html`, a results table
with Wilson CIs computed from the per-sample arrays.

| figure | from | shows |
|---|---|---|
| failure mix per config | failure.npy | where and why molecules are lost |
| validity by heavy-atom count, Wilson bands | rdkit_valid, element_counts | size dependence |
| bond-environment map: median shift × spread ratio, area = weight | bond_length_w1.json (+ angles) | which chemistry is off, in which direction |
| paired Δ vs control: config × metric, McNemar sd, seed band shaded | rdkit_valid, failure | A/B at a glance (C1) |
| metric vs epoch, one line per run (when names carry `:<epoch>`) | results.json | milestone curves (L1) |

**Pairing.** Same seed folder and identical element_counts per row; otherwise the
comparison is marked unpaired. McNemar's sd is `sqrt(n10 + n01) / n`, as in compare.py.

Test: a smoke test builds two fake evaluation dirs in tmp_path and checks that every figure
and index.html are written.

## Step 5: hydra rollout (each submit after your go-ahead)

1. **Snapshot.** Run `git archive` into `frozen_code/<sha>-eval-tiers` and run the tests
   there (gcc SIF, from an empty directory).
2. **Reference assets.** Build them in `repro/assets_qm9_b36d9cd218f5998d/`, since `data/`
   is read-only: bond_lengths, bond_angles and element_counts from the paper DB. This is a
   short CPU job of about 3 CPU-minutes for the RDKit walk. SMILES and elongation stay where
   they are.
3. **Parity gate.** Run the new prepare step on about 20 evaluated databases (QM9 and
   CrossDocked, several configs) into scratch dirs, and compare rdkit_valid and smiles with
   the stored arrays. The v1 arrays came from code_33d809d on the py312 SIF; v2 runs on
   py313 with rdkit 2025.9.6.
   - If they match, v2 only adds files.
   - If they don't, you decide whether to rescore validity everywhere with v2 or keep v1
     validity and add only the new arrays.
4. **`slurm/fast_rdkit.sbatch`** (~15 changed lines):
   - `EVAL_CODE` and `EVAL_SIF` override repro/env.sh's CODE and SIF.
   - The prepare targets add failure, n_fragments, element_counts, bond_lengths and
     bond_angles.
   - The default METRICS add the new metrics and bond_length_w1 / bond_angle_w1, with
     their reference arguments.
   - The "already evaluated" test reads provenance.json for `EVAL_CODE` instead of looking
     for the `rdkit_validity` key.
   - `submit_eval.sh` adds the failure shares to GEOM-Drugs' METRICS, since they need no
     reference.
5. **Backfill.** An array job in `scratch/eval_tiers/` runs over the list of existing
   `samples.db` (samples_h.db for heavy-atom data). It writes the new arrays and metrics
   only, checks rdkit_valid against the stored one, and logs mismatches. About 10
   CPU-hours, no GPU.
6. **`hy report <run>:<model>... [--control <run>:<model>]`** (~15 lines in bin/hy): it
   resolves names with find_run and runs gpff_report in the container from an empty
   directory. It writes to `<run>/frozen/report/` for one run, or to
   `scratch/<study>/report/` with `--out`. A laptop zsh function can rsync the folder and
   open it with `explorer.exe`.

Done when:
- `hy eval` on a new milestone produces the new metrics with provenance;
- every old fast evaluation has them too;
- `hy report` renders a comparison of two existing runs.

## Size

About 120 lines of library changes, 250 of report, 15 + 15 of hydra shell, and the tests.
New files: report.py, cli/report.py, the backfill sbatch and its list.

## Not in this plan (and what would trigger each)

| item | trigger |
|---|---|
| T0 pulse in training (`SamplerCallback`) | needs a new training snapshot; after the next model change |
| PB-fast with per-check columns | benchmark the mol config without "Energy ratio" first |
| MACE relaxation in the pipeline, NFE sweep config, σ-coverage and trajectory-by-outcome figures (T2) | a result that needs explaining |
| novelty split / recovery of the own test molecule | needs the row→test-index map, including the shape split |
| composition-standardized validity | the next typed-model comparison (~15 lines on element_counts) |
| milestone mode at 2.5k samples | needs n in the output path, or it collides with the 10k folders |
| EDM stability, torsions, FCD, LaTeX tables (T3) | paper writing |
| merging the five fast_rdkit copies, study.yaml | when a sixth study copies it again |
| replacing JS_MPD | kept for paper parity; the report shows elongation_w1 next to it |
