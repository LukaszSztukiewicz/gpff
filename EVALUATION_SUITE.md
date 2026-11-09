# GPFF evaluation suite: tiers, metric families, reports

Status: proposal, 2026-10-26. Nothing below is implemented unless marked *exists*.
Companions: [EVALUATION_METRICS.md](EVALUATION_METRICS.md) (literature survey) and
[EVALUATION_DIAGNOSTICS.md](EVALUATION_DIAGNOSTICS.md) (trajectory diagnostics, uncommitted
in this worktree).

## 1. What exists

Code, feat/evaluation-suite:

- `gpff_prepare_metadata` writes per-sample arrays to `evaluation/metadata/`: rdkit_valid,
  smiles, ring_counts, elongation, nfe, strain, bond_lengths.npz, bond_angles.npz,
  posebusters_valid, and gap/energy_U0 when property models are given.
- `gpff_calculate_metrics` reduces them to a flat `results.json`: validity, PB, uniqueness,
  novelty, Tanimoto, JS of elongation/gap/U0, bond_length_w1 and bond_angle_w1 (plus a
  per-environment breakdown JSON), and mean_strain.
- `gpff_relax`: UFF, xTB, and batched MACE on the GPU.
- feat/typed-type-readout, untracked `analysis/composition.py`: element atom and molecule
  shares, heteroatom share, element TV and size W1. Standalone CLI; its docstring already
  plans to be folded into `gpff_calculate_metrics`.
- Uncommitted here: `TrajectoryDiagnosticsRecorder` and `gpff_diagnose` (per-step JSONL,
  CSVs, one PNG).

hydra:

- Fast: `hy eval <run> <models>` runs freeze.sbatch, then two slurm/fast_rdkit.sbatch jobs
  (dd16 sdd16 heun16 | dd256 sdd256 sheun256), 10k samples, seed 42069. They compute
  validity, uniqueness, novelty and JS_MPD. Generation plus the RDKit pass take about 6 min
  per model at 16 NFE and 17 min at 256.
- Full: `FULL=1` runs repro/submit.sh: 12 configs, a generation array, an evaluation array
  (adds PB and Tanimoto), then metrics.sbatch and summary.txt. PB is about 1 h per config.
- The evaluation code is pinned to repro/code_33d809d with the py312 SIF (repro/env.sh), so
  this branch's structural metrics are not in the hydra stack yet. 0 of the 2,236 repro
  `samples.db` have bond tables, and `data/qm9/current/assets` has no bond or angle
  reference.
- Comparisons: `hy res` prints tables. Each study has its own compare.py (milestone_eval,
  sigma_oct21, types_eval), with hardcoded runs and its own McNemar.
  smart_tomcat_eval/paired_compare.py adds per-seed deltas and validity by heavy-atom bin.
  There are five copies of fast_rdkit.sbatch.
- No plots anywhere in repro/, scratch/ or bin/.

## 2. Findings that shape the design

1. **Generation is the cost, and almost every metric is free.** The RDKit pass, including
   the bond and angle tables, costs 1.6 ms per molecule per core: 13 s for 10k samples with
   the worker pool (struct_metrics_port bench). The expensive analyzers are:
   - PB's "Energy ratio" module, which embeds a 50-conformer ensemble per molecule and takes
     hours on large CrossDocked molecules.
   - Relaxation: MACE about 21 structures/s on a GPU, UFF about 240/s.
   - xTB and the property models.

   So tiers should differ in generation budget (configs × samples × seeds) and in which
   expensive analyzers they run. Every tier should compute every free metric. The fast tier
   computes four today.
2. **The fast samplers probe different noise levels.** sigma_oct21 measured where each
   sampler calls the model (fancy-monkey epoch 5000):
   - dd16 makes 46% of its calls below σ=0.03, where training draws 1% of its noise.
   - sdd16 makes 91% below σ=1 but only 5% below 0.03.
   - sheun16 makes 59% at σ≥1.

   A training change at low σ should therefore move dd16, and one at high σ should move
   sheun16. Pick the panel for σ coverage and sensitivity, meaning validity away from the
   ceiling, not for paper rows. sdd256 sits at 98–99%, near the ceiling. It is the headline
   number but carries the least model-selection signal.
3. **Training-seed noise is about as large as sampling noise.** In split_variability at
   epoch 200, five runs differing in split and/or training seed span 1.0–2.0 pp of validity
   per config. Same-split pairs differing only in seed are up to 1.2 pp apart (heun16). The
   paired McNemar sd, 0.3–0.5 pp at 10k, covers only sampling noise for two fixed trained
   models. A gap under about 1.5 pp between separately trained runs is not established
   without replicates.
4. **JS_MPD is noise at this scale.** dd256 ranges from 28 to 49 across the same five runs.
   `calculate_js_divergence` takes its bins from the min and max of both sets, so a single
   long molecule moves every bin. W1 in Å has no bins.
5. **Validity hides the cause.** `analyze_with_rdkit` returns False alike for:
   - non-finite coordinates,
   - failed bond-order assignment,
   - radicals,
   - several fragments,
   - timeouts.

   Each of these points to a different defect.
6. **Composition can game validity.** In typed-composition, typed models put N in 44–48% of
   molecules against 61% for the untyped baseline, and dropping N makes valences easier. For
   typed runs, read validity together with composition.
7. **Untyped samples are row-matched to a test molecule.** Row i is test composition i. This
   allows per-sample comparison with that sample's own reference molecule (recovery, a
   finer novelty split, elongation residual) and stratification by reference properties.
   Current novelty also counts rediscovering the held-out test molecule as "novel".
8. **results.json mixes code versions.** It merges keys computed by code_33d809d with keys
   from any later snapshot, and nothing records which is which. fast_rdkit's skip test is
   only "results.json has rdkit_validity and is newer than samples.db".
9. **The PB boolean adds nothing to ranking, but PB still carries information.** PB tracking
   RDKit across models is already validated. Even so, the RDKit-to-PB gap depends on the
   config. On cosmic-akita's best_model, PB rejects 5.8% of dd256's valid molecules but
   2.8% of sdd256's. The reason (bond lengths, angles, clash, flatness, energy) is
   geometry information that the boolean throws away.

## 3. Metric families, grouped by the question each answers

| Family | Question | Metrics | Cost | Status |
|---|---|---|---|---|
| A validity | Is it a molecule? | rdkit_validity; failure cause (nonfinite / no bond orders / radical / fragments / timeout); n_fragments; V·U·N; EDM atom and molecule stability | free | validity exists; rest new |
| A' PB | Would a pose checker accept it? | pass rate; per-check pass rates; PB-fast = mol config without "Energy ratio" | PB-fast to be benchmarked; full ~1 h/10k | boolean exists |
| B geometry | Is the 3D structure right? | bond and angle W1 per environment (+ median shift, spread ratio); torsion distributions; elongation W1; excess over the finite-sample floor | free | bond/angle on this branch; rest new |
| B' relaxation | How far from a minimum? | RMSD raw→relaxed; strain; validity after relaxation; W1 relaxed vs a relaxed reference | UFF ~1 min/10k, MACE ~8 min/10k GPU | gpff_relax exists, not in the pipeline |
| C composition | Right chemistry mix? | element TV, element molecule share, size W1 (typed only); composition-standardized validity; rings per molecule by size vs reference | free | composition.py on branch; rest new |
| D novelty | New and diverse? | uniqueness; novelty split into {in train+val, own test molecule, other test molecule, unseen}; Tanimoto; SNN | free (Tanimoto O(n²), seconds) | uniqueness, novelty, Tanimoto exist |
| E cost | At what price? | NFE distribution; wall time per sample; valid-unique per GPU-second | free | nfe array exists |
| F dynamics and field | Why? | trajectory diagnostics joined to final outcome; σ visits per sampler; force MAE per σ band and per element; equivariance error | one extra generation or eval pass | diagnostics uncommitted, store_sigmas on feat/sampler-sigma-log, eval_losses.py and sigma_bands in scratch |
| G conditional | Did it follow the condition? | shape target error; scaffold RMSD; later pocket metrics | free / later | shape error in diagnostics |
| P properties | Paper parity | JS of gap and U0 (needs property models: none on the cluster); FCD/Frag/Scaf (package not in the SIFs) | model or package needed | JS code exists |

Composition-standardized validity: stratify by heteroatom count (0, 1, 2, 3, 4+), take the
validity in each stratum, and weight the strata by the reference's shares. It removes the
advantage of producing easier compositions. For untyped models it equals plain validity, so
it is the headline only for typed runs.

## 4. Tiers

| Tier | When | Generation | Analyzers | Cost | Output |
|---|---|---|---|---|---|
| T0 pulse | inside training, every 50 epochs | 512 val compositions, sdd16 | RDKit pass: validity, failure mix, bond W1 against the val molecules | seconds per check | wandb curves |
| T1 fast | every milestone, automatic | 6 probe configs × 2.5k ("milestone") or 10k ("decision"), 1 seed | families A, B, C, D, E (everything free) | ~¼ (2.5k) or 1× (10k) of today's fast eval | results.json, report card, paired deltas vs control |
| T2 probe | when T1 shows something to explain | 1–3 configs at 2–10k; NFE sweep 8–256 at 2k; diagnostics run at 1k | PB-fast per check (full PB on 1k); MACE relax; σ visits; force MAE by σ; trajectory join; gallery | 1–3 h | deep-dive report |
| T3 paper | final models | 12 configs × 3 seeds × 10k | everything, including full PB, a relaxed reference, EDM stability, torsions, (FCD) | ~1 day, mostly CPU | paper tables (LaTeX) + figures + CIs |

T0 already exists as `gpff.callbacks.SamplerCallback` (`+callbacks=sampler`). It logs
`val/validity` on validation batches. Extend it with the failure mix and bond W1, keep its
compositions fixed across epochs so the curve is paired, and check that it samples with the
same weights the milestones save. It runs in the training snapshot, so it only applies to new
snapshots. Use it for trends, never for decisions.

The T1 probe panel is today's six configs, read as σ probes. dd16 covers low σ, sdd16 the
middle, sheun16 high σ with no ceiling, sdd256 is the headline, and dd256 and sheun256 are
the long-run versions. Milestone mode at 2.5k samples has about 1 pp binomial SE and about
0.8 pp paired sd. That is enough for curves at a quarter of the cost. Use decision mode at
10k for A/B calls.

T1 decision rule, printed on every comparison:
- Paired Δ outside its 95% CI on the same split and seed: a real difference for these two
  trained models.
- |Δ| < 1.5 pp between separately trained runs: not established without a second training
  seed.

## 5. Statistics

- Report every proportion with n and a Wilson 95% CI.
- A vs B: paired McNemar when split md5, seed and the element multiset of every row match
  (paired_compare.py already checks this). Otherwise unpaired, marked `u`.
- Distribution metrics (W1, TV): paired bootstrap over sample indices, 200 resamples.
- Finite-sample floor: compute the same metric between two disjoint real sets of matched n
  (bench: bond W1 between two 4,000-molecule halves of QM9 is 2.1 mÅ, against 7.2 for
  sdd256). Show it as a band and report the excess over it.
- Training-seed band: take it from split_variability; finish that study with paired McNemar
  within the same-split pairs. Draw it on every comparison plot.

## 6. Reports and figures

Use matplotlib only, with one style file, writing PNG for the HTML and PDF for the thesis.
pandas 2.3.2, matplotlib 3.10.8 and jinja2 3.1.6 are in both SIFs; pyarrow, seaborn, plotly
and py3Dmol are not. The HTML is self-contained, with base64 PNGs and CSV tables. The 3D
gallery loads 3Dmol.js from a CDN and needs no Python package.

### Report card (one model, all T1 configs)

```
┌ fancy-monkey:5000  vs control  ·  eval v2 · split 3a8bc7b4 · seed 42069 · n=10000 ─┐
│ headline: config × {Valid±CI, V·U·N, bond W1 − floor, NFE, Δ vs control}          │
├ F1 funnel ─────────────────────┬ F2 failure mix ───────────────────────────────┤
│ generated→finite→bond orders→  │ stacked by cause per config; n_fragments hist │
│ one fragment→no radical→valid→ │                                               │
│ unique→novel   (bars per cfg)  │                                               │
├ F3 validity by size ───────────┼ F4 bond environments ─────────────────────────┤
│ valid vs heavy atoms, lines per│ x median shift (mÅ), y spread ratio,          │
│ config, Wilson bands; ref size │ area = ref weight; ideal at (0, 1);           │
│ histogram below                │ same for angles                               │
├ F5 shape and rings ────────────┼ F6 composition (typed only) ──────────────────┤
│ elongation vs ref; residual to │ atom and molecule share vs ref; validity by   │
│ own test molecule; rings/size  │ N count; standardized validity                │
├ F7 novelty split ──────────────┴ F8 gallery ───────────────────────────────────┤
│ train+val / own test / other test / unseen  │ 12 valid + 12 per failure cause, │
│                                             │ 2D + 3D, next to its reference   │
└────────────────────────────────────────────────────────────────────────────────┘
```

What each panel teaches:
- **F1** shows where molecules are lost.
- **F2** shows why. Fragments point to long-range or size failure. No bond orders points to
  local geometry or H placement. Radicals point to valence. Non-finite points to the
  explosion bug class.
- **F3** shows whether a change helps small molecules only, which is the usual story on
  GEOM and CrossDocked.
- **F4** shows which chemistry the field gets wrong and in which direction, using numbers
  `bond_length_w1.json` already holds.
- **F8** is where intuition comes from: look at the failures.

### Comparison report (study.yaml: control + arms)

- **C1 delta forest**: config × metric rows, Δ ± 95% CI, the training-seed band shaded.
  This one figure replaces the per-study compare.py tables.
- **C2 discordance by size**: only-A-valid and only-B-valid counts per heavy-atom bin.
- **C3 environment arrows**: the F4 map with an arrow from control to arm for each
  environment.
- **C4 failure-mix delta.**

### Training curves (milestones)

- **L1**: each metric against epoch, one panel per config, runs overlaid with CI bands; val
  loss and forces MAE below on the same epoch axis.
- **L2**: heatmap of Δ validity vs control, configs × epochs.

### Sampler and field (T2)

- **S1 quality vs NFE** on a log axis, one line per sampler family, from an NFE sweep config
  file (8, 16, 32, 64, 128, 256 at 2k samples). Optionally plot against GPU-seconds.
- **S2 σ coverage vs model error** on a shared log-σ axis: a histogram of where each sampler
  calls the model (store_sigmas) over force MAE per σ band (sigma_bands). This explains
  dd16 against sdd16 in one picture.
- **S3 trajectories by outcome**: radius of gyration and per-step RMS displacement against
  step, valid vs each failure cause. Join the diagnostics JSONL to the evaluation on
  sample_idx.
- **M1** force MAE per σ per element; **M2** equivariance error.

## 7. Implementation

### Code (in the repo, pinned through an eval snapshot)

```
src/gpff/analysis/
  structure.py     + failure cause, n_fragments, torsions, EDM stability
  composition.py   from feat/typed-type-readout, + per-sample element counts
  collection.py    + elongation W1, novelty split, standardized validity, floors
  report/
    load.py        tidy DataFrames: samples, results, measurements; pairing check
    stats.py       Wilson, McNemar, paired bootstrap
    plots.py       one function per figure: DataFrame -> Figure
    html.py        jinja2 template -> index.html
src/gpff/cli/
  prepare_metadata.py, calculate_metrics.py   + --tier
  report.py        gpff_report --study study.yaml | --runs A:5000 B:5000 --control A:5000
```

`calculate_metrics` keeps its explicit style, but it dispatches to one function per
family, each returning a flat dict. composition.py already has that shape. A tier is a tuple
of families. That is a table of seven families, not of 40 metrics:

```python
FAMILIES = {"validity": ..., "geometry": ..., "composition": ..., "novelty": ...,
            "posebusters": ..., "relaxation": ..., "properties": ...}
TIERS = {"fast": ("validity", "geometry", "composition", "novelty"),
         "probe": (..., "posebusters", "relaxation"), "paper": tuple(FAMILIES)}
```

### Data

```
<run>/frozen/repro/<model>/<config>/seed<N>/evaluation/
  metadata/        existing + failure.npy (int8), n_fragments.npy, element_counts.npy,
                   ref_index.npy, pb_checks.npz, torsions.npz
  results.json     flat, unchanged shape
  provenance.json  eval_version, eval code, SIF, asset md5s, split md5, n, seed, gen code, time
<run>/frozen/report/<model>/index.html, figures/
scratch/<study>/study.yaml, report/{index.html, figures/, samples.csv.gz, results.csv.gz}
```

The loaders build three tidy tables:
- **samples**: one row per run, model, config, seed and sample_idx, with every per-sample
  array and the reference molecule's properties.
- **results**: one row per metric, with value, CI, n and eval_version.
- **measurements**: bonds, angles and torsions in long form.

The report folder keeps them as CSV so a notebook, on hydra's JupyterLab
(slurm/run_jupyterlab.sbatch) or on the laptop, can load them without a container.

### hydra

- `repro/eval_v2.env`: the gcc SIF, `frozen_code/<sha>-eval-v2`, and
  `ASSETS=repro/assets_qm9_b36d9cd218f5998d`. The assets are the bond and angle references
  from `gpff_reference_assets` plus the floors. They go outside `data/`, which is read-only,
  the way assets_crossdocked_6f6e27b88d7b55d3 does.
- `slurm/eval.sbatch` with `TIER=fast|probe|paper` and `N_SAMPLES`, replacing the five
  fast_rdkit copies. `submit_eval.sh` passes TIER through.
- `scratch/<study>/study.yaml` replaces each study's compare.py:
  ```yaml
  control: fancy-monkey
  arms: {key-moray: soft-sigma + rot. align, valid-crab: k-m + knee 2 + 5% anchors}
  models: [1000, 2000, 5000]
  configs: [dd16, sdd16, sheun16, dd256, sdd256, sheun256]
  ```
- `hy report <study>` runs gpff_report and prints the path. A laptop-side `hy pull <study>`
  can rsync it and open it with `explorer.exe`.
- `hy backfill`: one CPU array job runs the eval-v2 prepare and calculate steps over existing
  samples.db, with no generation. It writes only new arrays and checks that the recomputed
  rdkit_valid equals the stored one, which doubles as a regression test of the new code
  across 2,236 databases.

## 8. Roadmap

1. **Fast tier gets everything free.** Failure cause and n_fragments; composition folded in
   with element_counts; elongation W1; novelty split; V·U·N; Wilson CIs; provenance.json;
   reference assets and floors; eval-v2 snapshot; backfill (about 10 CPU-hours, so every
   past run gains the new metrics).
2. **Report v1.** Loaders, F1–F4, C1 and L1, the HTML, and `hy report`.
3. **One eval.sbatch with TIER**, plus study.yaml and gpff_report comparisons, retiring the
   per-study compare.py.
4. **Probe tier.** Benchmark PB-fast first, then: MACE relaxation on a subsample, NFE sweep
   configs, S2 and S3, and the 3D gallery.
5. **Paper tier.** EDM stability, torsions, multi-seed LaTeX tables, relaxed-reference
   geometry, and FCD/MOSES (needs a container rebuild).

## 9. Open decisions

1. Reference for distribution metrics: the full dataset, which is one asset per dataset and
   independent of the split (recommended, with the floor at matched n), or each run's test
   split.
2. Whether to run the backfill job.
3. T0 changes training code, so it applies to new snapshots only.
4. Milestone mode at 2.5k samples against keeping 10k everywhere.
