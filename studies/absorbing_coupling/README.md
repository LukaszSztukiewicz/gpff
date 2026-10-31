# Absorbing coupling study

Baseline: commit `4406a0d`, parent `38ae1c1`. The user explicitly requested
committing the existing implementation before starting; source worktree is clean.
New study changes remain uncommitted. Original plan and three references are in
`docs/`. Baseline file hashes and commit details are in `provenance/`.

## Predeclared stages (before submission)

Use QM9 `b36d9cd218f5998d`, the source run's split and
`gpff-py313-torch214-gcc.sif`. Main source:
`runs/default/26-10-27/absorbing-lw0.25`, epoch 2000; confirm at 5000.
Use seed 42069 for pilots, then 42070 and 42071 for confirmation. Final
generation count is 10,000 attempts per seed. Select settings using validation
pilots only. Do not call a pilot a final comparison.

1. Container regression and failure-mode tests, including disabled-control parity.
2. D1 and D2 validation diagnostics. Pilot grid: sigma
   `[0,.1,.25,.5,1,2,30]`, rate `[.25,.75,1]`, 32 validation molecules,
   exact training transform and separately labelled raw Gaussian corruption.
   Fixed-geometry decoding: two draws at sigma `[0,.5,2,30]`, all three rates,
   clean/generated sequential and parallel generated context; initially 16
   validation molecules. Save molecule-level records and probabilities.
3. Validation generation pilots, 128 molecules, same size order. Controls:
   original cosine/linear 256 steps; legacy confidence and random remasking
   strength .2; corrected random and saved-confidence .2 at 256 steps;
   T1 original reveal timing at 256 coordinate steps and 128 steps (512 versus
   256 production NFE); earlier blocks are separate, power 2, block size 1.
   Include unconditioned scheduled sampler at 256 steps for T3.
4. E1 training-only formula table; conditional-size and joint-size pilots with
   composition-given absorbing model, and untyped model when located. Save
   requested formulas; assert exact preservation. Formula and position RNGs
   are separate and paired between alternatives.
5. T2 and T3 **separate** matched weights-only restarts from epoch 2000.
   Five arms: T2 ordinary/rollout/isotropic; T3 unconditioned/conditioned.
   T2 has one frozen-teacher full denoising step, mixture .5, no extra noise.
   Isotropic displacement matches each molecule's centered teacher displacement
   norm. Clean partial context and original alignment/atom correspondence stay
   fixed. Recompute force/energy targets and ordinary weights at INITIAL sigma.
   T3 adds per-molecule log-sigma at floor 1e-4 to the shared representation;
   new output layer is zero initialized. List all four new state keys explicitly.
   All arms restart AdamW and ReduceLROnPlateau, same learning rate 1e-4,
   same seed 999, batch 512, pinned data noise, source split. Disable early
   stopping for fixed update budgets. Pilot 100 optimizer updates, then **all
   five arms** 2,000 additional updates after pilot inspection. Milestones
   additional updates 100, 500, 1000, 2000; do not extend only a favored arm.
   Distinct T2/T3 controls can share identical weights but remain separate runs.
   Eager execution is used for every short pilot to avoid compile warm-up.
   Input hashes before augmentation verify matched corruption/data draws.
   The 100-update pilot qualifies every arm before the main comparison. The
   main comparison starts each arm afresh from the same epoch-2000 source for
   exactly 2,000 updates, avoiding mid-epoch dataloader-resume ambiguities.
   This clarification precedes submission of any main run. Every arm receives
   the same pilot and main budgets. Report raw-student update models separately
   from EMA. ReMDM strength .2 rescales the admissible bound for b; legacy
   strength .2 multiplies the next mask rate. These parameters are different
   remask schedules, not identical per-token remasking probabilities.
6. Select settings, repeat D2, epoch-5000 inference confirmation, then final
   10,000-attempt, three-seed evaluation. No claim about training-seed variance.

Production NFE counts only sampler calls; probe calls and wall time are separate.
Changing coordinate steps, active shapes, or random draws changes the noise path.
Pair comparisons on size/formula/position streams where valid, never on an
invented shared stochastic path. Confidence remasking is state dependent and
does not inherit exact random-kernel marginal claims.

## Entry points

`python -m gpff.cli.absorbing_study --help` supports `conditionals`, `pilot`,
`generate`, and `formula`. Execution refuses to run outside Slurm. All outputs
are new paths; existing reports are never overwritten. Frozen-code snapshots
include these scripts and source. `check.sbatch` runs tests from an empty dir.
The unchanged sampler is `sampler=absorbing`; study switches live under
`sampler=absorbing_study`; scheduled VE uses `sampler=absorbing_scheduled`.
Training configs: `gpff_absorbing_recovery` and `gpff_absorbing_sigma`.
`+init_weights=<epoch_model> run.ckpt_path=null` explicitly restarts optimizer
and scheduler; `+init_allow_sigma=true` allows exactly four sigma parameters.

## Reporting requirements

Every row needs checkpoint/config/snapshot/dataset/split/novelty hashes, job,
seed, attempts, actual NFE and wall time. Report failures (nonfinite geometry,
MASK, ABSENT) over all attempts. Chemical shares exclude nonphysical tokens but
keep failed attempts in molecule denominators. Keep all-output and valid-only
composition, whole formula/count and N/O joint distributions, size, VUN,
uniqueness denominator, geometry/failure metrics and functional groups. Bootstrap
whole molecules, and use paired tests only for a documented valid pairing.
No hypothesis is supported by validity or element TV alone.

Status and results are maintained in `RESULTS.md`; queued jobs are unfinished.
