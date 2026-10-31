# Absorbing atom types: experiments and implementation handoff

Written 2026-10-28. This plan follows the [review](absorbing_review.md), the
[original design](absorbing_atom_types.md), and the [recorded results](absorbing_results.md).
All experiments below are proposed, not completed. The aim is to locate the
composition error and compare remedies without mistaking validity gains for
distributional accuracy. Implement separate switches and controls; do not bundle
all changes into one model.

**Start from the actual latest implementation.** At this review, the latest
local atom-type branch is `feat/typed-type-readout`, HEAD `38ae1c1`, in
`~/STUDIES/gpff-typed-readout`. Its absorbing implementation is uncommitted:
tracked changes in `keys.py` and `sample/base.py`, plus new vocabulary, type
head, transforms, samplers, configs and tests. Hydra has reproducible versions
in `frozen_code/absorbing-v1` and `absorbing-v2`.

The local and remote branch `feat/absorbing-state-diffusion-atom-types` instead
points to `759d3d6`, without that implementation. Remote `feat/atom-types-port`
points to `59a4012`. The documentation worktree is based on `38ae1c1` but does
not contain the uncommitted implementation. Recheck all of this before starting:
branch names and commit dates alone are insufficient.

Create an implementation branch such as `feat/absorbing-coupling-study` in
`~/STUDIES/gpff-absorbing-coupling-study` from the newest verified atom-type
ancestor. If the implementation remains uncommitted, copy its tracked diff and
only its relevant untracked source/config/test files into the new worktree.
Record the source HEAD, file list and hashes; compare with frozen v2. Do not
stash, reset, commit, or edit the source worktree. Do not silently start from a
branch that lacks absorbing types. Carry these documents and their references
into the implementation worktree as well.

**Use stable controls and explicit provenance.** Reconstruct run settings using
`hy cfg`, not defaults. The main checkpoint is `absorbing-lw0.25:2000` for
diagnosis and its epoch-5000 checkpoint for confirmation. Preserve v1/v2 as
controls. Use `abs256` and `abs256lin` for generated types, and explicitly mark
the absorbing model's `sdd256` configuration as types given. Include the untyped
baseline and tuned continuous `type_scale=0.15` results as context where their
evaluation protocol matches.

Keep QM9 `b36d9cd218f5998d`, each model's correct split, its container and the
matching novelty assets. Record checkpoint hash, code snapshot, resolved config,
dataset/split hashes, seed, sample count and actual network evaluations (NFE).
New work goes in a separate Hydra study directory; do not overwrite old reports.

**D1 — Measure the actual type decisions.** Instrument the model call inside
the sampler, after stochastic preprocessing and before coordinate/type updates.
Store the exact input state and probabilities used for unmasking. Separately
label post-update probes. Record the injected noise scale, force-derived sigma,
mask fraction, entropy, selected slots, and committed counts. Never label a
force estimate or the latest injected noise as the true residual geometry error.

On selected steps, compare actual partial context with all types remasked, using
exactly the same previously masked slots and finite-molecule subset. Diagnostic
calls must not consume the sampler's RNG or mutate its state. Save a manageable
subset of states and full predictions for later independent inspection.

For each element k, log `M_k = committed_count_k + sum_masked p(type=k)` around
the geometry and type operations separately. Use it to locate systematic loss
of expected nitrogen count. Posterior-count conservation is an expectation
property of exact reverse processes with exact posteriors, not a per-trajectory
requirement of this approximate sampler. A change in this proxy alone is not a
proof of which component is wrong.

**D2 — Separate conditional prediction from generated geometry.** Use the same
validation molecules across a small grid: sigma `0, 0.1, 0.25, 0.5, 1, 2, 30`
and mask rates `0.25, 0.75, 1`. Include the exact training transform; label raw
Gaussian corruption separately because training uses alignment. On masked atoms,
measure negative log likelihood, Brier score, reliability by element, entropy,
recall, and aggregate frequencies, stratified by molecule size and mask fraction.
Report subgroup sizes. Directly measure per-atom sample/argmax disagreement.

Then hold each corrupted real geometry fixed and reveal types sequentially in
random order, recomputing predictions after each reveal. Compare clean visible
context with self-generated visible context, and compare sequential versus
parallel decoding. Use repeated draws to examine formula/count distributions
and N/O co-occurrence. Generated context can legitimately choose an alternative
completion; distinguish its distributional quality from reconstruction accuracy.
If deficits already appear with fixed real geometry, the geometry rollout is
not the sole cause.

**D3 — Test corrected remasking separately.** Keep the old heuristic as a named
control. Implement the random transition from
[ReMDM equation 6](https://arxiv.org/html/2503.00307v4), with bounded remask
probability. If r is the current mask rate, r_next the desired rate, and b the
visible-to-mask probability, the simultaneous masked-to-visible probability is
`a = (r - r_next + (1-r)*b)/r`. Require valid probabilities and handle endpoints
explicitly. Determine transitions from the pre-step state; do not immediately
remask freshly decoded atoms in that same transition.

Verify the intended mask fractions statistically. Run correction even when no
token happens to be newly decoded. For a separate confidence ablation, use saved
probabilities from decoding or an explicitly masked probe, never untrained
visible-position logits. Do not automatically extend exact marginal claims to
a state-dependent confidence heuristic. Compare old, corrected-random and
confidence variants at epochs 2000 and 5000, at matched NFE.

**T1 — Condition geometry on newly sampled types.** This is an inference-only
change. After the usual noisy-input pass, sample a type or small block, update
the type state, and recompute forces at the same coordinates before moving
positions. Do not inject noise a second time. Extra model calls must be counted.

First retain the original reveal schedule to isolate the effect of recomputing
forces. Then test earlier, sequential decisions as a separate setting. Compare
both matched coordinate steps and matched total NFE, for example the same model
call budget as `abs256`. Keep type RNG separate and log actual call counts;
changing the number of stochastic steps changes the path and must be reported.
The test asks whether letting geometry respond immediately to a chemical choice
improves composition and validity together. It does not assume that another
forward pass must help.

**T2 — Train recovery from short, data-anchored rollouts.** Start from a noised
training molecule with known clean positions and types. Use a frozen teacher or
detached EMA prediction to make one sampler-like geometry update. Feed that
geometry, optionally with the next documented noise injection, to the student.
Keep clean partial type context initially, so this tests coordinate errors
without also changing the categorical corruption. Mix ordinary and rollout
examples; a 50/50 mixture and one teacher step are initial ablation settings,
not claimed optima. Only add a second rollout step after inspecting displacement
and correspondence. Log the rollout error distribution.

Preserve atom identities and the existing alignment convention. Do not assign
training labels to unrelated generated molecules, or introduce new matching
that uses masked ground-truth types without documenting it. For student input
Y, recompute pseudo-forces as `2*(X0-Y)`. Apply the ordinary loss-weight policy
using the recorded initial corruption sigma, and state that this is an
augmentation objective, not an exact new diffusion bound. Do not invent a true
sigma for the rollout state from its force norm.

Compare with ordinary continued training from the same checkpoint, using the
same initialization, learning-rate policy, data draws and additional updates.
Also include an isotropic-noise augmentation control matched to the rollout's
displacement scale. Keep its force targets and weighting treatment identical.
This distinguishes exposure to structured model errors from merely more noise
or more training. Self-conditioning is a separate method, outside T2's scope.

**T3 — Supply known geometry-noise information.** Add an optional per-molecule
log-sigma embedding to the shared representation so both heads can use it.
Use the recorded corruption sigma during training, with a documented floor for
zero noise. Keep mask rate independent; no masking-clock input is required for
this experiment. Disabled conditioning must preserve old checkpoint behavior.

At sampling, use an explicit coordinate-noise schedule and an adapted existing
schedule-based sampler with absorbing updates. Supply the scheduled sigma,
not the previous force norm or just the latest extra-noise amplitude. Compare
the conditioned and unconditioned models using that same sampler. Also evaluate
the unconditioned model with original direct denoising, separating a sampler
effect from a conditioning effect.

Use matched training initialization and budgets. If initializing new modules
from an old checkpoint, list every new parameter and use a matched control with
the same optimizer/restart policy; do not hide arbitrary missing state keys.
Keep alignment fixed initially. Passing nominal sigma does not undo the
type-constrained alignment or prove that the joint reverse process is exact.
Evaluate D2 again to see whether sigma improves uncertain conditionals before
attributing any generation gain to it.

**E1 — Generate joint composition before geometry.** Build a formula prior only
from the training split, preserving whole count vectors and their frequencies.
First sample `C | N` at the existing evaluation sizes to isolate composition;
then sample full C so size is generated too. Construct fresh batches with
exchangeable element assignments and fresh positions, and use composition-given
GPFF. Save each requested formula and confirm the output count vector matches.
No test molecule supplies its formula to this generator.

Compare generated-composition validity, functional groups, diversity and geometry
with the absorbing controls. Exact sampled counts do not guarantee the correct
structures, and filtering invalid molecules can bias the retained distribution.
Report both. Learning a pocket-conditioned count model and conditioning an
absorbing model on a remaining count budget are later extensions; the empirical
joint-formula baseline is required in this study. ABSENT and pocket training are
outside this study so they do not obscure the diagnosis.

**Run in stages and leave interpretable results.** Implement D1–D3, T1–T3 and E1
as independent options. Start with container tests and small Hydra pilots.
Use validation data for choosing settings. Diagnose at epoch 2000, confirm
inference findings at epoch 5000, and use matched controls for every training
change. Predeclare training update budgets and comparison milestones in the
study README before submitting; do not extend only a favoured variant. Run
T2 and T3 separately before considering a combination.

After pilots, evaluate selected settings with 10,000 molecules per generation
seed. Include seed 42069 for continuity and two additional recorded seeds for
confirmation. A single-seed pilot is provisional. For formula-generating runs,
pair alternatives on the sampled formula and position seed. Elsewhere pair on
size and independent RNG streams where possible; do not claim identical noise
paths when an intervention changes random draws or active batch shapes.

Every final comparison must include:

- Validity and VUN over all generation attempts, uniqueness with its denominator
  stated, and failure counts including non-finite geometry and unresolved
  MASK/ABSENT tokens.
- Element shares/TV, N/O molecule presence, joint formula/count distributions,
  N/O co-occurrence and size distribution, with reference split and denominator.
- Separate all-output and valid-only composition. Exclude nonphysical tokens
  from chemical histograms while retaining their failed samples in denominators
  and reporting excluded counts. Do not silently discard failures.
- Existing bond/angle and failure metrics, plus functional-group frequencies
  among valid molecules. These distinguish restored chemistry from restored
  element totals. Keep novelty assets and evaluator versions fixed.
- Actual NFE and wall time. Use molecule-level bootstrap intervals and paired
  tests when pairing is valid; atoms within a molecule are not independent.
  Multiple generation seeds do not measure training-seed variability.

Add tests for the actual failure modes: probe hooks preserve state/RNG and capture
pre-decision inputs; context comparisons use identical slots; ReMDM transitions
follow the prescribed mask marginals and endpoints; T1 uses updated types at
unchanged coordinates and counts extra calls; T2 targets reconstruct the clean
position from the modified input; T3 receives the declared sigma; formula batches
use training-only counts and no test composition. Run these only on Hydra through
Slurm, inside the run's container from an empty directory.

Save machine-readable metrics and per-molecule records, compact diagnostic
plots/tables, the selected trajectory states, and scripts to reproduce each
comparison. Include a manifest connecting every row to its snapshot, checkpoint,
config and Slurm job. Write an analysis with one of `supported`, `weakened`, or
`unresolved` for each hypothesis, and explain conflicting metrics. No change is
a success solely because validity or element TV improves. Preserve unsuccessful
results as evidence rather than replacing their files.

**Handoff for the implementing agent.** Read `~/STUDIES/gpff/AGENTS.md`, this
plan and its three companion documents first. Implement and analyse D1–D3,
E1 and all three
targeted changes T1–T3, in a new worktree based on the newest verified atom-type
implementation. Follow the branch-copy instructions above: the branch named
“absorbing” is currently stale and the required implementation is uncommitted.
Keep the source worktree untouched and leave your changes uncommitted unless
the user requests a commit.

Work in this order: provenance and controls; corrected diagnostics; T1 and
remasking; E1; isolated T2 and T3 training comparisons; confirmation evaluations
and written analysis. Use separate switches and reproducible entry points. Run
all Python, tests, training and evaluation on Hydra with Slurm and frozen code;
never switch `~/gpff/codebase`. Use `hy cfg`, dry-run submissions, and run-specific
resume settings. Do not cancel jobs or delete/move/rebuild data or runs without
permission. Finish with a concise results document, comparison table, artifact
paths and any unresolved questions so the reviewing agent can inspect the
evidence without rerunning the study. Mark unfinished work explicitly; do not
present queued jobs as completed experiments.
