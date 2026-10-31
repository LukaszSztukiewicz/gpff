# Absorbing coupling study: implementation and preliminary evidence

Updated 2026-10-29. **The research study is not complete.** Tables below are
validation pilots, not the planned 10,000-attempt, three-seed confirmations.
Implementation changes are uncommitted on `feat/absorbing-coupling-study`, based
on the explicitly committed baseline `4406a0d`. All execution used Hydra Slurm
and the original GCC container; the source worktree and running codebase were
left untouched. All 13 baseline files match frozen `absorbing-v2` by SHA-256.

## What is implemented and checked

- D1 captures actual inputs/probabilities after stochastic preprocessing, before
  decisions, and labels subsequent probes separately. Probes preserve RNG,
  tensor state, model buffers and training flags. Context comparisons use the
  same masked slots and finite molecules. Per-element count proxies, selected
  slots, committed counts and selected full states/predictions are saved.
- D2 covers the specified sigma/mask grid, exact training alignment and separately
  labelled raw Gaussian corruption, element reliability/recall, NLL, Brier,
  entropy and actual sample/argmax disagreement. Fixed-geometry clean/generated
  sequential context and parallel decoding have repeated-draw formula records.
- D3 includes the untouched legacy control, bounded simultaneous random ReMDM
  transitions and a separate stored-decoding-confidence heuristic. The latter
  has no exact marginal claim. Random mask marginals and endpoints are tested.
- T1 recomputes forces at unchanged coordinates after the type decision, without
  another noise injection. Both coordinate-step and NFE controls are present;
  early sequential blocks are a separate switch. Extra calls are counted.
- T2 provides one frozen-teacher, data-anchored update and a matched-norm isotropic
  control, with an independent augmentation RNG. Atom identities and partial
  clean context are preserved. Targets are `2*(X0-Y)` and weights use the INITIAL
  sigma; this is augmentation, not an exact new diffusion bound.
- T3 provides optional shared log-sigma conditioning, floor `1e-4`, zero initial
  output, exact new-parameter allowlisting and a declared-sigma VE ancestral
  sampler. Disabled conditioning retains the old parameter/state-key behavior.
- E1 fits whole formula frequencies from training indices only, supports both
  composition conditional on size and jointly generated size/composition, builds
  fresh exchangeable batches with synthetic IDs, and verifies requested counts.

Latest completed test job `5151522`: **86 passed**, including real-network parity
with cached full neighbors and a regression ensuring final test confirmations
appear in reports with their correct partition. Integration jobs `5147655` and `5147761` passed all
ten inference/training/reload checks. The latter also tested exact-update model
and checkpoint saving. Failures in earlier snapshots remain in their original
logs and directories; corrected runs have new names.

The final failure regression verifies that nonfinite geometry and unresolved
MASK/ABSENT tokens are counted independently when they overlap.

## Validation generation pilots

Epoch-2000 `absorbing-lw0.25`, 128 attempts, seed 42069, type seed 0. Element TV
and atom shares use **all outputs**, excluding nonphysical tokens from chemical
histograms while keeping failed attempts in molecule denominators. VUN also uses
all attempts. Uniqueness uses valid attempts and is near 100% at this small sample
size; it must not be compared directly with uniqueness on 10,000 samples.
The composition reference is the full validation partition. The copied-types
control's nonzero TV reflects comparing its 128 molecules with that partition.

The following rows all come from `pilots5/` with the same generation code.

| Setting | Valid % | VUN % | Element TV | N atom % | O atom % | NFE | Bond W1 | Angle W1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosine control | 80.47 | 46.88 | 0.0230 | 4.20 | 7.00 | 256 | 0.0068 | 1.3347 |
| Linear control | 77.34 | 46.09 | 0.0125 | 4.63 | 7.65 | 256 | 0.0064 | 1.1087 |
| Legacy confidence | 98.44 | 44.53 | 0.0335 | 3.41 | 6.73 | 256 | 0.0067 | 1.0895 |
| Legacy random | 93.75 | 47.66 | 0.0401 | 3.10 | 6.38 | 256 | 0.0061 | 1.1746 |
| Corrected random | 96.09 | 49.22 | 0.0247 | 4.02 | 7.00 | 256 | 0.0061 | 1.0275 |
| Stored-confidence heuristic | 81.25 | 39.06 | 0.0484 | 2.80 | 5.86 | 256 | 0.0069 | 1.6843 |
| T1, matched NFE | 83.59 | 50.00 | 0.0221 | 3.67 | 7.61 | 256 | 0.0066 | 1.3299 |
| T1, matched steps | 82.03 | 42.97 | 0.0195 | 4.15 | 7.39 | 512 | 0.0061 | 0.9982 |
| T1, early sequential | 64.84 | 38.28 | 0.0118 | 4.55 | 7.78 | 295.25 | 0.0064 | 1.1443 |
| Unconditioned scheduled sampler | 53.91 | 30.47 | 0.0089 | 6.60 | 7.78 | 256 | 0.0065 | 1.0356 |
| Same absorbing model, types given | 98.44 | 59.38 | 0.0089 | 5.86 | 8.53 | 256 | 0.0068 | 0.9549 |

ReMDM strength .2 rescales its admissible remasking bound; legacy strength .2
multiplies the next mask rate. These are different remasking schedules. Neither
comparison isolates the transition formula while holding per-token b identical.

Paired with the cosine control on size and initial position seed, corrected random
remasking improves validity by **15.63 pp**, molecule-bootstrap 95% interval
**[7.81, 23.44] pp**, exact McNemar **p=0.000325**. Its element TV remains similar
to the cosine control and nitrogen remains deficient. T1 at matched NFE changes
validity by **3.13 pp**, interval **[-5.47, 10.94] pp**, **p=0.585**. It does not
establish an improvement. Pairing does not assert identical stochastic paths.

E1 uses `pilots7/`, which adds fresh-batch neighbor integration and caching.
Its control rerun differed numerically (78.13% valid versus 80.47% in `pilots5/`);
CPU real-network tests verify the caching transformation's mathematical parity.
Do not treat cross-snapshot rows as bitwise paired trajectories.

| E1 setting | Attempts | Valid % | VUN % | Element TV | N atom % | O atom % | NFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training formula conditional on size | 128 | 95.31 | 61.72 | 0.0097 | 5.90 | 7.48 | 256 |
| Joint training formula, including size | 128 | 99.22 | 55.47 | 0.0060 | 5.44 | 8.28 | 256 |

Every requested formula was preserved. Counts alone do not demonstrate correct
topology or functional groups; an empirical prior also cannot generate unseen
formulas. All-output and valid-only distributions, N/O co-occurrence, formula and
size histograms, functional-group frequencies and failure causes are in each
run's machine-readable report. Bond/angle comparisons use the fixed existing
evaluator `cf5d247-eval-tiers` and its fixed QM9 assets.

Historical context for the forthcoming 10,000-attempt test evaluations, seed
42069, 256 NFE, same QM9 database and pinned evaluator/assets:

| Historical model | Epoch | Types | Valid % | VUN % | Unique / valid % | Element TV | N atom % | O atom % |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Untyped `fixed-snail` | 2000 | Given | 98.24 | 47.95 | 92.41 | — | — | — |
| Continuous `type_scale=0.15` | 5000 | Generated | 93.60 | 42.26 | 84.13 | 0.03346 | 3.974 | 6.337 |

Each historical model uses its own correct split. These are contextual results,
not paired validation controls or matched training budgets. Original results and
provenance are copied to Hydra's study `context/` folder.

## Diagnostics and training qualification

For D2's 32 validation molecules with every type masked, clean geometry has
NLL about 0.00033. At sigma .5, NLL is 0.585 and sample/argmax disagreement is
20.6%; expected N share is 4.52%, versus 5.24% in these molecules. At sigma 30,
disagreement is 45.4%. Aggregate readout agreement therefore cannot establish
deterministic per-atom marginals. The noisy conditional result is provisional:
one corruption per molecule and 32 molecules are insufficient to claim exact
calibration or establish the sole source of generation bias.

All five 100-update training arms completed from the same epoch-2000 weights,
with matched audited input hashes at updates 0, 1, 2 and 99. The two ordinary
controls differed by at most 0.0007055 in a model parameter after training;
matching inputs does not imply bitwise identical GPU optimization. All arms
used the same fresh AdamW/scheduler policy. T3's four newly initialized parameters
are listed explicitly in `initialization.json` and `reports/training100.json`.
Pilot rollout/isotropic error distributions are recorded there too. Qualification
checks and short validation loss are not generation-quality evidence.
The teacher and isotropic controls both have mean per-molecule displacement RMS
0.7621 angstrom. The corresponding clean-position error RMS averages are
0.4116 and 1.1863 angstrom, respectively; all recorded values are finite. A
matched displacement norm does not imply matching residual corruption.

## Hypothesis assessment

| Hypothesis | Assessment | Current evidence |
|---|---|---|
| Aggregate sampling/argmax agreement proves sharp marginals | **weakened** | Direct disagreement is substantial on noisy real inputs. |
| The head is calibrated on all relevant conditionals | **unresolved** | Clean recognition is excellent; noisy and generated-context conditionals require the subgroup/repeated-draw analysis and larger samples. |
| Geometry rollout is the sole cause of the nitrogen deficit | **unresolved** | Fixed-geometry conditional discrepancies exist, but small samples and approximate posteriors prevent causal attribution. |
| Corrected random remasking can improve validity | **supported**, provisionally | Large paired validity gain in the validation pilot; N deficit persists. |
| Stored-confidence remasking fixes joint generation | **weakened** for this setting | Worse composition and VUN in this pilot. No exact marginal claim applies. |
| Immediate type-conditioned forces improve composition and validity together | **unresolved** | Matched-NFE interval includes zero and nitrogen does not improve; extra-NFE geometry gains are a separate result. |
| Earlier sequential choices improve joint generation | **weakened** for this setting | Better element TV accompanies sharply worse validity. |
| One-step recovery training improves generation | **unresolved** | Implementation and qualification passed; main comparison is unfinished. |
| Declared sigma conditioning improves generation | **unresolved** | New modules are qualified; the unconditioned schedule alone loses validity. |
| A training-only joint formula prior is a viable practical baseline | **supported**, provisionally | Exact counts and strong pilot validity; topology, generalization and multi-seed distributional accuracy remain open. |

D1 artifacts permit operation-wise proxy comparisons on identical finite
molecules. A proxy change is neither a per-trajectory conservation requirement
nor proof of which model component is wrong. Do not substitute force-derived
sigma or the most recently injected noise for true residual geometry error.

## Remaining work and reproducibility

Submitted main comparison: all five arms, **2,000 fresh additional updates** from
the same epoch-2000 source, identical budgets and restart policies. The preceding
100-update pilot is a separate qualification, avoiding mid-epoch resume/data-order
ambiguities. Milestones are 100, 500, 1000 and 2000; never extend only one arm.

Recorded validation selection for epoch-5000 confirmation: cosine, linear,
corrected random, T1 matched NFE, E1 conditional size, E1 joint size. The prepared
matrix has **18 jobs**, 10,000 attempts at seeds 42069/42070/42071, type seeds
0/1/2. No confirmation result is included above. Multiple generation seeds do
not measure training-seed variability. Six generation controls and five D2
evaluations are queued automatically after all main training arms finish.
Legacy/confidence and T1
step-matched epoch-5000 validation checks also remain separate from this selection.

The first completed full-size point is the epoch-5000 cosine control, seed
42069: **8,997/10,000 valid (89.97%)**, VUN **46.74%**, uniqueness among valid
outputs **90.60%**. Five attempts have nonfinite geometry and 98 nonphysical
tokens are excluded from chemical histograms. This is one seed of the control,
not a completed comparison of the selected settings. Actual mean NFE is
255.8748, reflecting early freezing of failed attempts; generation took
433.1 seconds on its recorded GPU/node.

The cosine control has now completed all three generation seeds: mean validity
**89.85%** (sample SD across generation seeds **0.21 pp**), mean VUN **46.18%**,
mean element TV **0.02291**, N **3.979 atom%**, O **7.159 atom%**. These are
means of per-seed metrics, not pooled uniqueness/VUN or training-seed uncertainty.
The other selected settings are still pending or running.

Submission ledger (not completion claims):

| Job | Purpose |
|---|---|
| 5151348 | Five 2,000-update training arms, common 40gb GPU constraint |
| 5151349 | Eighteen epoch-5000 confirmations, two concurrent tasks |
| 5151350 | Confirmation evaluations; explicit test-split reference |
| 5151384 | Six generation and five D2 comparisons after main training |
| 5151385 | Their validation-reference evaluations |

Two obsolete jobs (`5147631`, `5147632`) are waiting on the failed snapshot-4
test dependency. Cancelling them requires the user's approval under AGENTS.md;
they have not been cancelled. They are unrelated to the corrected job chain.

Hydra root: `~/gpff/scratch/absorbing-coupling/`.

- `jobids.txt`: all submissions and failure history.
- `provenance/`: `hy cfg`, source settings, baseline/v2 hashes, checkpoint,
  dataset/split hashes and archive hashes.
- `pilots5/`, `pilots7/`: generation manifests, databases, exact NFE/time,
  selected states and full per-molecule evaluation records.
- `reports/pilot5-20261029/`: comparison table, plots, row manifest, D1 proxy
  changes and D2 repeated-completion summaries.
- `reports/pairs/`: paired statistics above.
- `reports/training100.json`: checkpoint hashes, exact new parameters,
  teacher calls, input audits and augmentation error distributions.
- `training/`: five qualification runs and subsequent main runs when submitted.
- `training-main2000.jsonl`, `confirmation5000.jsonl`, `selection-20261029.json`:
  explicit next-stage commands and selection, prepared before test generation.

`run.sbatch` executes explicit JSON argument lists without shell evaluation.
`submit.sh` defaults to dry run. `evaluate.sbatch` uses `REFERENCE_PARTITION=val`
for pilots and must use `REFERENCE_PARTITION=test` for confirmations. Existing
completed reports are retained. `prepare_comparisons.py`, `compare.py`,
`training_evidence.py` and `summarize.py` reproduce the comparisons inside Slurm.

Local review copies require no rerun:
[comparison table](artifacts/20261029/reports/pilot5-20261029/comparison.md),
[D2 plot](artifacts/20261029/reports/review-20261029/conditionals.png),
[training audit](artifacts/20261029/reports/training100.json),
[paired remasking statistics](artifacts/20261029/reports/pairs/remdm-random.json).
[Latest seed-level table](artifacts/20261029/reports/review-20261029-v2/comparison.md)
and [overlapping failures](artifacts/20261029/reports/review-20261029-v2/failure_counts.md)
include the completed control confirmations.

Supplementary summaries `reports/confirmation5000-v2/` and
`reports/main2000-v2/` retain independent, overlapping nonfinite/MASK/ABSENT
counts in `failure_counts.md` and per-row JSON. A nonfinite first-cause label
therefore cannot hide unresolved tokens. Original evaluations are retained.

## executive research summary

GPFF can make valid molecules, but it still generates too little nitrogen.
The larger baseline tests produced about 90% valid molecules. Getting the
element mix right remains a separate problem.

Two approaches look promising. In small tests of an earlier model, letting the
sampler reconsider atom-type decisions raised validity from about 80% to 96%,
although the nitrogen shortage remained. Sampling a formula from the training
data first, then generating its geometry, also worked well in pilots. This
guarantees the chosen atom counts; structure quality and diversity still need
larger tests, and this baseline cannot produce unseen formulas.

Making geometry respond immediately to newly chosen atom types has not shown a
clear benefit for the same amount of computation. Choosing types earlier made
the element mix closer to the data but produced fewer valid molecules.

The cause of the nitrogen shortage remains unresolved. Training on the model's
own geometry errors and telling it how noisy the coordinates are are still being
tested. The larger comparisons are unfinished, so the promising pilot results
remain provisional.
