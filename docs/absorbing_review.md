# Absorbing atom types: review of the diagnosis

Written 2026-10-28. Read alongside the [original design](absorbing_atom_types.md)
and [experimental results](absorbing_results.md). This review corrects the
interpretation of some results; it does not replace the measurements. The
[experiment plan](absorbing_experiments.md) turns the open questions into tests
and ends with a handoff for implementation.

**The composition bias is established; its root cause is not.** Absorbing
diffusion is a reasonable extension of GPFF. The current evidence points toward
a problem in how geometry and atom types develop together, but does not prove
that the type head is correct during generation or that extra training noise is
the only possible remedy.

The review inspected the local implementation and Hydra's frozen
`absorbing-v1/v2`, training configuration, and scripts and reports under
`scratch/absorbing/` and `scratch/typed-composition/`. No new experiments were run.

**What the results establish.** Absorbing improves oxygen frequencies,
uniqueness and valid/unique/novel generation compared with the reported
continuous-type models, at a cost in validity. Nitrogen remains deficient.
For `absorbing-lw0.25`, cosine sampling gives 3.89 atom% N at epoch 2000 and
3.88 at epoch 5000, against about 5.68 in the reference. Improving total element
TV therefore does not show that the nitrogen problem is merely slow convergence.
The strong types-given results show that GPFF can generate good geometry when
composition is supplied. They do not establish the quality of its force field
when most types are unknown.

**Several conclusions need narrowing.**

| Earlier conclusion | What the evidence supports |
|---|---|
| The type head is essentially perfect. | It recognizes elements very accurately on clean real geometry and matches average frequencies on the tested noisy inputs. Its conditional probabilities during generation remain unverified. |
| Sampling and argmax agree, so the marginals are deterministic. | The tested readout change barely affects aggregate composition. Per-atom differences can cancel; entropy and paired disagreement are needed to establish sharpness. |
| More type training, another loss, and inference fixes are ruled out. | Particular settings failed to fix the deficit. They do not exclude whole classes of solutions. |
| Absorbing composition error is only slow convergence. | Oxygen improves, while nitrogen is nearly flat in the cited cosine runs. |
| Loss weight affects convergence speed, not final quality. | Weights 0.25 and 1 reach similar measured quality at epoch 5000. This does not establish their eventual limits. |

Matching the average predicted nitrogen share is necessary but insufficient for
calibration. A classifier that always returns the dataset prior passes that test.
We also need probabilities and joint composition conditioned on size, visible
types and geometry. Generated molecules have no known per-atom ground truth;
classification accuracy belongs to real or data-anchored diagnostic inputs.

**The trajectory probe has two important confounds.** In
`hydra:~/gpff/scratch/absorbing/trajectory_probe.py`, the extra model call happens
after `denoise_step`. The real sampler adds noise, predicts types and forces,
updates coordinates, and then commits types using the probabilities from before
the coordinate update. The probe consequently reads different coordinates from
those used for the decision. Its reported sigma is a force-derived estimate
from the preceding input, not a measured error of the probed structure.

At step 128 of the 256-step sampler, fresh positional noise has standard
deviation about 0.496 angstrom, while the table reports a force-derived sigma of
0.255. Comparing that post-update structure with a real molecule corrupted by
0.25 angstrom is not a comparison at matched corruption levels. Even the first
reported trajectory row is measured after an update, rather than on the prior.

The context comparison also changes the population: the all-remasked column
averages over all atoms, while the partial-context column averages only over
remaining masked atoms. Evaluate both predictions on the same slots before
attributing their difference to the visible context. These corrections may
preserve the drift diagnosis, but the existing comparison does not prove it.

**The remasking experiment tested a heuristic, not the published ReMDM kernel.**
The v2 implementation applies ordinary unmasking and then adds remasking without
adjusting the unmasking transition. Its actual mask fraction therefore need not
follow the assumed schedule. It also runs the remasking operation only when
some atom was unmasked in that step.

Confidence is read from current logits at already-visible atoms, although the
type loss trains only masked positions. Those logits are not trained uncertainty
estimates. Published ReMDM changes both transitions and uses the probability
stored when a token was decoded for its confidence scheme. The random-remasking
control avoids the confidence issue but retains the transition issue.
The experiments show a validity/composition trade for this implementation, not
that principled remasking cannot help. See [ReMDM, sections 3–4](https://arxiv.org/html/2503.00307v4).

**Time-free masking does not imply time-free geometry conditioning.** With
independent geometry corruption and masking, the masking rate can cancel from
the type posterior when the visible pattern is given. Geometry noise generally
cannot: the required conditional still depends on the reliability of the
coordinates. Thus RADD supports omitting the masking clock, but does not prove
that a type head reading noisy coordinates can omit sigma. This is our extension
of the argument in [RADD](https://arxiv.org/html/2406.03736v2).

Without sigma, the head learns a conditional averaged over the training noise
distribution. Whether that conditional serves the sampler is an empirical
question. Also, unchanged types do not permit caching logits while geometry
changes. The existing type-constrained Hungarian alignment modifies the
coordinate corruption/coupling and must be accounted for before claiming an
exact Gaussian reverse process.

**The leading mechanism is a feedback loop between uncertain geometry and
irreversible type choices.** A squared-error denoiser predicts a conditional
mean, weighted according to its training loss. With types unknown, that estimate
can average over different chemical possibilities. GPFF takes a full step toward
it, then commits types without recomputing the coordinate update using those new
types. Later predictions treat the resulting geometry and labels as evidence.

For example, an uncertain carbon-like and nitrogen-like environment can produce
an averaged coordinate estimate. Sampling nitrogen does not turn that estimate
into nitrogen-conditioned geometry. Repeated updates may favour common chemical
environments and lose rare alternatives. This is a hypothesis to test, not a
demonstrated explanation of the nitrogen deficit.

Exposure bias describes a training/generation mismatch; it does not identify
which operation creates the mismatch. Nor must a generator identify a hidden
true nitrogen atom before choosing nitrogen. It can sample a chemical choice
under uncertainty and then generate compatible geometry.

**There are two complementary routes forward.** For a practical generator,
sample a joint composition vector first, then generate geometry conditioned on
it. For understanding and improving absorbing generation, test immediate
type-conditioned coordinate updates, training on short data-anchored rollouts,
and explicit geometry-noise conditioning. These are T1–T3 in the experiment plan.

The composition route factors generation as
`p(C | pocket) p(X, A | C, pocket)`, where C contains the count of each element.
An initial baseline can sample whole formulas from the training split and use
the existing conditional GPFF. A later learned count model can condition on a
pocket. Alternatively, C can condition an absorbing model whose remaining count
budget constrains local assignments. Counts must be sampled jointly, preserving
their correlations. Total size is the sum of the counts, so ABSENT is optional
for generating size; it is useful for changing size during refinement.

This prevents local decisions from eliminating the chosen nitrogen budget. It
does not guarantee the right topology or functional groups. An empirical formula
prior also cannot produce unseen formulas. These limitations must be reported.

**Generic extra noise is an ablation, not the uniquely justified fix.** Before
alignment, adding independent Gaussian noise to a VE-corrupted coordinate simply
increases its variance. With a clean type as the unchanged target, this largely
changes the corruption distribution. It does not automatically reproduce the
structured errors made by the sampler. The target treatment that distinguishes
[Ning et al.'s input perturbation](https://arxiv.org/html/2301.11706v3) from an
ordinary noise change needs separate justification in GPFF.

The claimed literature gap also needs narrowing.
[FlowMol3, section 2.6](https://arxiv.org/html/2508.12629) directly studies drift
in joint continuous/discrete molecular generation, using self-conditioning,
fake atoms and geometry distortion. Self-conditioning supplies a previous
prediction as an additional input; it is distinct from replacing a training
input with a short model rollout. This is relevant prior art, not evidence that
any one of those remedies will solve GPFF's nitrogen deficit.
