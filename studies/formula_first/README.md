# Formula-first GPFF: minimal implementation and training study

2026-10-30. User-authorized implementation, Hydra tests, and three full trainings.
No architecture, force loss, training loop or sampler kernel changes.
Worktree `feat/formula-first-proven` starts at `759d3d6`, the proven-ox source
revision. This also preserves its hashed epoch-shuffle seeding; the earlier
absorbing ancestor used a different shuffle policy. `sample/formula.py` is reused
from the uncommitted coupling study (source SHA256
`2641b02b6c66cbd7b14842388316f993d18abac42a6a4d3d03bef9f91c6111da`).
The source coupling worktree is untouched. Only provenance validation is added
to that prior. The standard generation CLI gains `prior=formula`, source checks,
formula verification and a joint-mode batch iterator independent of test data.

## Model and objective

Let C=(n_H,n_C,n_N,n_O,n_F), n=sum(C), and M=(X,A) be a molecule with explicit
hydrogens. Factor the distribution as p(C) p(M|C). The empirical maximum-
likelihood formula prior is q(C)=training_frequency(C)/training_sample_count.
Count whole molecules, including repeated formulas, rather than sampling unique
formulas uniformly or drawing elements independently. No held-out composition
enters q. Both the database and the model's exact split are hashed and checked.

Given C, construct its multiset of elements and randomly permute the slots.
Each distinct labeled arrangement A has probability prod_k(C_k!)/n!. Draw fresh
Gaussian coordinates and run the existing equivariant GPFF with A fixed. The
uniform permutation and isotropic positions provide exchangeable initialization;
an equivariant denoiser then gives an exchangeable output law. Formula C is
already determined by A, so a separate formula embedding or atom-type head is
unnecessary. This does not supply a molecular graph: topology and isomer choice
remain the job of the conditional geometry generator.

Train standard `experiment=gpff` on real training molecules with every type
visible. For the existing aligned corruption Y and clean target X0 the objective
is E[w(sigma) ||F_theta(Y,A)-2(X0-Y)||^2], with w=min(1,sigma^-2) and the
repository's atom reduction. Its pointwise optimum is

    F*(Y,A)=2 E[w(sigma)(X0-Y)|Y,A] / E[w(sigma)|Y,A].

Thus a time-agnostic field depends on the training distribution of sigma even
when all types are known. Changing that distribution tests the balance between
local chemical refinement and recovery from broadly corrupted coordinates.
The existing type-constrained alignment and direct-denoising sampler remain
fixed; this objective is not claimed to define an exact Gaussian reverse SDE.
Known types remove the averaging over alternative elements, but not the
multimodality of conformations and molecular topologies within a formula.

For a target distribution p and conditional geometry law q_theta,

    KL(p(M)||q(C)q_theta(M|C))
      = KL(p(C)||q(C)) + E_p(C) KL(p(M|C)||q_theta(M|C)).

This is an identity where the distributions admit the required densities and
support; it is not a computable GPFF likelihood. An empirical prior assigns zero
mass to unseen formulas, so the population KL can be infinite. The identity
separates composition coverage from conditional structure quality. E1 cannot
generate unseen formulas, although new isomers of known formulas are possible.
We deliberately defer a learned count prior until this baseline is confirmed.

Validity filtering changes the output formula distribution to
q(C|valid) proportional to q(C) a_theta(C), where a_theta(C) is acceptance.
Always report all attempts and valid-only composition, validity by nitrogen
count and training-formula frequency, N/O co-occurrence, functional groups,
novelty, uniqueness and isomer diversity within repeated formulas. Restoring
counts alone is not success. Retrying geometry for a fixed sampled C could reduce
selection bias, but a finite retry cap still weights formulas by
1-(1-a_theta(C))^K; retries are therefore excluded from this initial comparison.

The separation of noise weighting and sampling choices is consistent with the
design perspective of Karras et al., https://arxiv.org/abs/2206.00364. The above
factorization and its application to GPFF are our study design, not a claim that
that paper validates formula-first molecular generation.

## Three predeclared full trainings

All start from scratch with seed 999, the SAME explicit `proven-ox-999` split,
QM9 `b36d9cd218f5998d`, GCC container, PaiNN 256 channels / 4 interactions,
batch 512, AdamW lr 1e-4, original scheduler and EMA, deterministic per-sample
noise, compilation, and the reference's **100,000 maximum epochs and early
stopping with patience 500**. This replaces the initial 5,000-epoch proposal
following the user's correction of the reference baseline. Initial weights are
matched because only the corruption config
differs. No checkpoint initialization, masking or categorical loss is used.
Use the explicitly pinned reference split rather than recreating it. The user
designated proven-ox-999 as the no-atom-type-prediction baseline, not fixed-snail.

| Run | log-sigma mean | log-sigma std | sigma cap | Rationale |
|---|---:|---:|---:|---|
| formula-base | -0.7 | 1.2 | 30 | Existing, successful types-given GPFF control |
| formula-local | -1.2 | 1.2 | 30 | More local geometry/chemical refinement |
| formula-broad | -0.2 | 1.2 | 30 | More exposure to large geometry errors |

The medians before truncation are approximately .50, .30 and .82 angstrom.
These are motivated hypotheses, not established improvements. Training/validation
losses use different sigma distributions and must not be ranked across runs as
if they were a common metric. Compare generation at fixed epochs and, if needed,
evaluate all models on one common held-out corruption grid. Equal epochs and
data batches at fixed milestones imply equal intended optimizer budgets; record
any skipped batches. Early stopping can produce unequal final training budgets;
report final results separately from fixed-milestone comparisons.

Milestones match proven-ox: 100, 200, 500, 1000, 2000, 5000, 10000, 12000, 12560,
15000, 20000, 30000, 40000, 50000. Slurm jobs request
48 hours; should a run time out, use its own latest completed milestone and
`hy resume`, never the historically best `last.ckpt`. Full training completion
is distinct from submission. The maximum budget can require multiple allocations.

## Qualification and evaluation

Run all Python/tests exclusively in Hydra Slurm/container jobs from empty dirs.
CPU tests cover train-only support/frequencies, exact counts, independent RNG
streams, source mismatch, and joint generation beyond the test-set size without
accessing its molecules. GPU qualification trains all three actual architectures
for short runs, saves/reloads models and generates formulas through the ordinary
CLI. A pretrained proven-ox-999:5000 joint-formula pilot checks practical generation;
it is a smoke/pilot result, not evidence of the new trainings' final quality.

Fit the formula table once from proven-ox-999's training indices; it has no gradient
training. Three requested full trainings are the conditional geometry models.
The main generation protocol is joint formula+size, 256 NFE, 10,000 attempts at
seeds 42069/42070/42071. Pair models on formula and starting-coordinate streams.
The smaller pilot uses 128 attempts. Compare with empirical-prior resampling
uncertainty and held-out formula support before attributing composition TV to
the geometry model. Final evaluation uses the pinned evaluator and each model's
own split. Fixed checkpoints avoid selecting on test metrics. Multiple sampling
seeds do not establish training-seed robustness.

At the same formula seed and number of attempts, all three geometry models must
have IDENTICAL all-output formula counts, even if coordinates fail. Their
all-output composition TV is consequently not a model-selection metric here.
Differences identify broken pairing or count preservation. Select using validity,
VUN, valid-only composition, functional groups and geometry, including results
within each formula. Atom fractions converge to E[C_k]/E[sum(C)], whereas
molecule-level averages of C_k/sum(C) are a different statistic.

Example after the table and a checkpoint exist:

```sh
gpff_generate prior=formula prior.formula_path=/path/formula.json \
  model_path=/path/epoch_5000_model split_path=/path/split.npz \
  datapath=/path/qm9_filtered.db sampler.n_steps=256 n_samples=10000 \
  output_db=/new/path/samples.db
```

Job IDs, snapshot hashes, checks and observed results are recorded in RESULTS.md.
