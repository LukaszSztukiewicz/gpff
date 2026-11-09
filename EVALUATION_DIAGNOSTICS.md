# GPFF Diagnostics Suite

## Purpose

Evaluation metrics say whether a run improved. Diagnostics should show what changed, when it changed, which samples drove the result, and where a proposed mechanism failed. They should support exploratory questions without requiring a new evaluator for every experiment.

The first implemented layer records compact sampling trajectories and produces comparable per-step and per-sample reports. The existing full-coordinate trajectory option remains available for detailed structure inspection.

## Capture and report

Enable diagnostics while generating:

```bash
gpff_generate \
  sampler=heun \
  model_path=/path/to/model \
  split_path=/path/to/split.npz \
  datapath=/path/to/qm9.db \
  n_samples=1000 \
  store_diagnostics=true
```

This writes `<output_db_stem>_diagnostics.jsonl` next to the generated samples database. Each row describes one molecule at the prior or after a denoising step. The records contain no coordinates and can be substantially smaller than full trajectory storage.

Summarize one or compare several runs with:

```bash
gpff_diagnose \
  --input /path/to/run_a_diagnostics.jsonl /path/to/run_b_diagnostics.jsonl \
  --output_dir /path/to/diagnostic_report
```

The report contains:

- `trajectory_by_step.csv`: ensemble means, medians, and 10th/90th percentiles for each trajectory step and run.
- `sample_diagnostics.csv`: one row per sample with start/end shape, total and maximum movement, coordinate finiteness, final sampler state, and available condition residuals.
- `sample_triage.csv`: samples with non-finite coordinates first, then samples with the largest per-step movement. Large movement is a review cue, not an automatic failure.
- `trajectory_diagnostics.png`: run overlays for size, step movement, finite-coordinate fraction, and shape-condition error.
- `summary.json`: run counts, non-finite sample rates, NFE summaries, input paths, and interpretation notes.

Sample indices refer to output order. Paired comparisons require the same split and ordering; use the same seed and generation setup when the intent is to isolate a model or sampler change.

## Captured signals

For each molecule at each state, the recorder stores:

| Signal | What it helps reveal |
|---|---|
| Radius of gyration and three descending principal variances | Collapse, expansion, anisotropy, or abrupt shape changes |
| Principal variance ratios | Shape evolution independent of total molecular size |
| Centroid drift from the prior | Unintended translation during denoising |
| Per-step RMS and maximum atom displacement | Large jumps, oscillation, or sampler-specific movement patterns |
| Finite-coordinate fraction | When numerical instability first appears |
| Element counts | Composition changes for samplers that update atom types as well as coordinates |
| Sigma, NFE, frozen state, and sampler-provided log values | Relate trajectory behavior to noise scale, compute, and stopping |
| Shape target ratios and their error | Whether shape-conditioned sampling approaches its requested anisotropy |
| Constrained-atom RMS displacement from the prior | Whether scaffold-conditioned sampling preserves fixed atoms |

`frozen` is the sampler's internal mask. The terminal step can mark every molecule frozen even when it did not satisfy an earlier stopping criterion, so it is not a success or validity label. Chemical validity and physical plausibility remain the job of the evaluation metrics.

## Measure research ideas

| Research question | Diagnostics to inspect | Additional evidence to pair with it |
|---|---|---|
| Does a new sampler make progress more smoothly or efficiently? | Step displacement, radius/shape trajectories, sigma, NFE, frozen state, non-finite onset | Final validity, geometry quality, valid molecules per second, same-seed runs |
| Does joint atom-type and position diffusion improve coupling? | Element-count evolution and geometry change by step | Add atom-type entropy/calibration, inferred bond-order and valence evolution, and per-step chemical validity |
| Does a shape prior control the intended geometry? | Generated and target variance ratios, condition error, centroid drift | Final shape distribution and a matched unconditional baseline |
| Does scaffold conditioning preserve a motif while improving free atoms? | Constrained-atom displacement, free-atom movement, per-sample shape | Scaffold RMSD, chemistry/geometry validity, novelty outside the fixed scaffold |
| Does pocket conditioning create plausible binding interactions? | The current sampler trace is only a trajectory view | Add receptor-aware clashes, residue contacts, interaction fingerprints, strain, docking variants, and per-pocket success |
| Did an architecture or training change improve the learned field? | Sampling traces can expose downstream behavior, not isolate field accuracy | Add held-out force-vector residuals by noise scale, prediction calibration, E(3)/permutation equivariance checks, and local smoothness tests |

## Modular growth plan

Keep diagnostics as independent calculators that write per-sample or per-step records. Each diagnostic should declare its required inputs, task profile, units, and whether it adds model evaluations or external dependencies. A report should retain distributions and sample-level rows so new questions can be explored without rerunning generation.

1. **Sampling dynamics:** refine the current observer with per-sampler step-size and force summaries, adaptive stopping traces, and quality-versus-NFE curves.
2. **Chemistry evolution:** add optional per-step graph inference, valence/bond-order failures, chirality, and atom-type transition statistics. These calculations are more expensive, so keep them opt-in.
3. **Model-field diagnostics:** measure force/noise prediction error by sigma and molecular environment; check rotation, translation, reflection, and atom-permutation behavior; compare predicted and observed denoising updates.
4. **Condition diagnostics:** generalize target residuals to property constraints, scaffold preservation, and pocket-conditioned objectives.
5. **Experiment comparisons:** add paired per-sample deltas, seed-level confidence intervals, stratification by atom count/composition/condition, and quality-versus-compute frontiers.

The current recorder implements trajectory geometry, sampler state, composition, shape-target error, and scaffold displacement. The later modules are a roadmap; they should be added when the associated research task and reference signals are defined.
