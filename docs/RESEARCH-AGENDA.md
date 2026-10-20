# GPFF — Research Agenda

Working notes on where GPFF goes next. The notes cover:

* which design choices define the method;
* which settings were inherited from image diffusion and never retuned for
  chemistry;
* which new capabilities follow from time-agnosticism.

Written 2026-10-20 against `feat/soft-sigma-weights` (`d39a9fe`). File links
point at this branch's tree (`04252cb`), the same tree [`SPEC.md`](SPEC.md)
describes.

Sections 1–4 state the agenda. Section 5 checks it against the code and a
small simulation. Several claims in §§1–4 change as a result, and each check
says how. Section 6 maps the label-noise literature onto GPFF: which tools
transfer, and which are traps. Section 7 orders the work.

---

## Table of contents

1. [The organizing idea](#1-the-organizing-idea)
2. [What makes GPFF GPFF](#2-what-makes-gpff-gpff)
3. [Where the current design falls short](#3-where-the-current-design-falls-short)
4. [New directions](#4-new-directions)
5. [Checks against the code and a simulation](#5-checks-against-the-code-and-a-simulation)
6. [What the label-noise literature offers](#6-what-the-label-noise-literature-offers)
7. [Suggested order](#7-suggested-order)
8. [Candidate paper contributions](#8-candidate-paper-contributions)
9. [Appendix A — shrinkage simulation](#appendix-a--shrinkage-simulation)
10. [Appendix B — Stein estimator toy](#appendix-b--stein-estimator-toy)

---

## 1. The organizing idea

GPFF's network reads the noise level off the geometry instead of being told
it: `σ̂ = std(F_θ(x)/2)` per molecule (SPEC §1). Every item below is one of:

* a condition that makes that read-back possible (§2);
* a place where the current implementation corrupts the read-back, or a
  setting copied from image diffusion without being retuned (§3);
* a capability the read-back enables (§4).

## 2. What makes GPFF GPFF

### 2.1 Defining properties

Change any of these and the method is a different one.

* **Time-agnostic.** The network gets no σ or t input.
* **Force output.** The label is `F = 2(x₀ − x_t)`, the negative gradient of
  the pseudo-PES `E = ‖x_t − x₀‖²`.
* **Variance-exploding corruption.**

### 2.2 Commitments

These could change, but should not.

* **Cartesian space.** It is non-compact, it is in force units, and it is
  additively compatible with physical force fields.
* **SE(3) equivariance.**
* **Direct denoising.** The predicted forces go to an optimizer.
* **σ⁻² weighting and permutation handling, in spirit.** Retune the first
  (the soft-sigma-weights branch) and generalize the second (§3.1, §5.2).

### 2.3 Admissibility

A triple (corruption, pseudo-PES, space) supports a time-agnostic generative
force field if and only if:

1. **The space is non-compact.**
2. **The network output lives in the data space.**
3. **The readout is monotone.** With `E = φ(r)` and `r = ‖x_t − x₀‖`, the
   label magnitude is `‖F‖ = φ′(r)`. The network recovers σ by inverting `φ′`,
   so `φ′` must be strictly monotone.
4. **The scale statistic concentrates.** `‖x_t − x₀‖ ≈ σ√(3N)`, with
   relative fluctuation `O(1/√(3N))`. Noise shells at different σ are
   therefore nearly disjoint. This is also why blind models fail at low
   ambient dimension: there, the shells overlap.

Gaussian corruption plus a quadratic PES buys exactly one more thing. Through
Tweedie's formula, the force label becomes proportional to the score, which
is what ancestral and Heun samplers need. Direct denoising needs only
conditions 1–4. Going non-Gaussian keeps the MLFF half of GPFF and forfeits
the drop-in-diffusion half.

Remarks:

* **The conditions are not independent.** Compactness breaks condition 3 at
  the top end. On a torus (torsion space), a wrapped Gaussian is uniform past
  σ ≈ π and stays uniform. Every larger σ then produces the same label
  statistics, and the readout saturates. VE needs a non-compact space because
  "exploding" has to actually explode.
* **The force parameterization carries the scale.** An ε target has
  `‖ε‖ ≈ √(3N)` at every σ, so it contains no scale information. The code
  already reflects this: the noise-model path refuses to run without `σ_t`
  ([`src/gpff/sample/base.py:158`](../src/gpff/sample/base.py)).
* **Cartesian is stronger than condition 1 requires.** It was chosen for
  compatibility with physics, not for admissibility.

### 2.4 Prior: stay VE, cap σ_T, shape the covariance

Accept that the prior is wrong. The alternative, switching to VP for a
bounded prior, fixes prior mismatch. Prior mismatch is the one problem GPFF
already absorbs:

* A t-conditioned model started off-schedule is handed a false `t` and never
  recovers.
* GPFF re-reads σ̂ off the geometry at every step and corrects itself. Guidance
  and shape projection work through the same mechanism.

A pocket-shaped cap on σ_T also eases the cutoff problem (§3.5).

## 3. Where the current design falls short

### 3.1 Hungarian matching

`align_permutation`
([`src/gpff/utils/alignment.py:390`](../src/gpff/utils/alignment.py)) runs a
per-element Hungarian assignment over all atoms for every training sample
(SPEC §2.3). The step has a good part and a broken part, and it does not
survive conditioning.

**The good part: a paper framing.** Matching same-element atoms to minimize
`‖x_t − P x₀‖²` defines a multi-reference pseudo-PES:

* the reference set is the permutation orbit of `x₀`;
* the combination rule is hard-min, the β → ∞ limit of softmin.

Generalizing to conformers, or to reactant/product pairs, changes the
reference set, not the architecture.

**The broken part.**

* **Discontinuity.** Hard-min makes the target discontinuous at assignment
  ties, and ties become dense once σ is comparable to the spacing between
  same-element atoms. SPEC §2.3 calls the target "smooth". It is only
  piecewise smooth.
* **Shrinkage.** Reassignment shrinks `‖x_t − P x₀‖` relative to identity
  matching. The label therefore understates the true displacement, which
  biases σ̂ and, through it, the adaptive sampler.

Softmin at finite β removes the discontinuity. §5.2 covers its cost and the
right β.

**Incompatible with conditioning.** Permutation invariance is correct only
because the composition is fixed and bonds are inferred post hoc. With joint
atom types, a clamped scaffold or a bonded pocket, a specific atom must stay
that atom. For conditional GPFF, restrict matching to the free atoms or drop
it.

### 3.2 σ̂ is biased, and calibration is nearly free

σ̂ is `std(F̂/2)` per molecule
([`src/gpff/sample/base.py:184`](../src/gpff/sample/base.py)). The network
regresses a posterior mean, which shrinks toward zero, and Hungarian
shrinkage has the same sign. The adaptive sampler consumes this biased σ̂.

**Fix:**

1. On held-out data, fit the curve `σ ↦ E[σ̂ | σ]`.
2. Invert it at sampling time.

This needs no retraining, takes hours of work, and acts directly on the
headline NFE number. §5.1 revises where along σ the bias sits.

### 3.3 One scalar σ per structure

Training draws one σ for the whole structure
([`src/gpff/pseudo_pes/geometries.py`](../src/gpff/pseudo_pes/geometries.py)).
Sampling-time scaffold conditioning (SPEC §7.3, §8.3) restores the scaffold
atoms to their reference positions every step, which is σ = 0, while the free
atoms sit at large σ. The model never saw that mixed-noise state in training.

**Fix:** train with hierarchical per-atom σᵢ. Draw a global scale, then
per-atom multipliers, with probability mass on both the uniform case and the
strongly mixed case. This one change gives:

* scaffold, linker and fragment-growing conditioning that is in-distribution
  instead of a robustness exploit;
* a meaningful per-atom σ̂, so the sampler can freeze converged atoms and step
  only the disordered ones;
* partial re-noising mid-trajectory, selective resampling and per-atom
  inpainting;
* pocket conditioning (ligand noised, protein clamped), which is literally a
  mixed-noise structure, trained for instead of hoped for.

§5.3 adds a constraint on how the multipliers are drawn.

### 3.4 σ distribution inherited from image diffusion

Training uses `ln σ ~ N(−0.7, 1.2²)`, truncated at σ_max = 30 Å
([`src/gpff/pseudo_pes/geometries.py:38`](../src/gpff/pseudo_pes/geometries.py)).
The standard deviation is exactly EDM's. Molecules, unlike images, have known
length scales:

| scale | length |
|---|---|
| thermal motion | ~0.1 Å |
| bond | ~1.5 Å |
| van der Waals contact | 3–4 Å |
| molecular diameter | 5–10 Å |

The informative σ range is where the distance-inferred bond graph becomes
ambiguous. That range can be measured instead of guessed. Putting training
mass where the topology dissolves is a free improvement. §5.10 shows where the
mass sits today.

### 3.5 The radial cutoff is the wall at scale

At σ = 30 Å, atoms are tens of Å apart. Any cutoff-based message passing
disconnects the graph, and the model sees isolated atoms it cannot predict
anything for. QM9 hides this completely: the neighbour list is fully
connected (SPEC §4.2, §13 item 5).

A pocket complex cannot be fully connected, and it needs long-range coupling
precisely at high noise. This, rather than composition or data volume, is the
binding constraint on scaling GPFF. Options:

* **σ-adaptive cutoff.** Bootstrap it from a cheap geometric scale estimate
  rather than from σ̂, to avoid circularity.
* **Coarse global attention channel** alongside the local messages.
* **Cap σ_T with a pocket-shaped prior** so the ligand never scatters out of
  range. This is the cheapest option and is wanted anyway (§2.4).

§5.9 adds two more options.

### 3.6 The metric can't see the claim

Validity means RDKit sanitizing a distance-inferred bond graph (SPEC §9.3). A
molecule with every bond 5% too long is still "valid". The thesis is
physical realism from force-field-style relaxation, but validity is
topological and blind to geometry quality.

Report instead:

* RMSD to the MMFF- or DFT-relaxed structure;
* strain energy;
* bond-length and angle distributions against reference.

These cost no retraining. They are the metrics under which GPFF should beat
diffusion baselines by more than it currently appears to. They also give the
loss weighting and the σ distribution something to be tuned against. The
existing PoseBusters pass (`posebusters_valid`) checks bond and angle bounds,
but only as pass/fail.

## 4. New directions

### 4.1 Samplers

* **Quasi-Newton (L-BFGS) or FIRE.** L-BFGS builds an inverse-Hessian
  estimate from the force history. On a near-quadratic pseudo-PES that model
  is close to exact. The original expectation was convergence in under 6 NFE.
  This is the smallest implementation on the page, and it needs no
  retraining. §5.6 tempers the expectation.
* **Langevin / NVT on the pseudo-PES.** Because `F` is a force, thermostatted
  dynamics at a pseudo-temperature sample an ensemble instead of descending
  to one mode. Combined with the softmin multi-basin PES, this would make
  metadynamics and umbrella sampling available on a generative model's own
  surface. Whether anyone has done this is not yet verified. §5.7 covers what
  the ensemble would actually contain.
* **Non-monotone, per-atom σ trajectories.** A t-conditioned model must
  follow one global schedule. GPFF can re-noise individual atoms
  mid-trajectory and keep going, because the model re-reads the state at
  every step. This is the capability most specific to GPFF's invariants and
  the least explored. §5.4 and §5.5 qualify the claim that it comes with
  zero retraining and gives sublinear NFE.

### 4.2 NFE versus wall-clock

A step costs about one forward pass. L-BFGS and FIRE bookkeeping is `O(m·3N)`
vector arithmetic per step, which is negligible. At a fixed batch size,
wall-clock time therefore tracks NFE, with two exceptions:

* stopping each molecule adaptively leaves batch slots idle unless the batch
  is compacted;
* per-atom freezing saves nothing unless the forward pass itself shrinks
  (§5.5).

Report NFE, wall-clock time at a fixed batch size, and the per-molecule NFE
distribution.

### 4.3 Internal-coordinate pseudo-PES

Replace the target with a pseudo-PES in internal coordinates, while the
network still outputs a Cartesian force:

    E = Σ_bonds k (r_ij − r⁰_ij)² + Σ_angles … + …

The constants are read off the reference structure. The label is still free
and the output is still a Cartesian force. The target is now shape agreement
rather than position agreement, so at large σ the network spends no capacity
recovering the global frame from noise.

This is a classical force-field functional form with reference-derived
constants, which strengthens the "GPFF as MLFF pretraining" claim.

SPEC §13 item 1 still applies: changing the pseudo-energy invalidates the
`x̂₀ = x + F/2` and `σ̂ = std(F/2)` conversions every sampler uses. This
variant needs its own σ readout and is limited to direct denoising (§5.8).

## 5. Checks against the code and a simulation

### 5.1 Shrinkage peaks mid-range, not at σ_max

Appendix A simulates per-element Hungarian matching, as `align_permutation`
does it, on a synthetic molecule-like blob with composition C₇O₂H₁₀ (19
atoms):

| σ (Å) | ‖matched‖ / ‖identity‖ | samples reassigned |
|---|---|---|
| 0.3 | 1.000 | 0% |
| 0.5 | 0.995 | 16% |
| 0.7 | 0.984 | 46% |
| 1 | 0.964 | 83% |
| 2 | 0.895 | 99% |
| 3 | **0.869** | 100% |
| 5 | 0.879 | 100% |
| 10 | 0.916 | 100% |
| 30 | 0.966 | 100% |

* **Hungarian shrinkage.** Absolute shrinkage grows with σ. Relative shrinkage
  peaks near 3 Å and then decays roughly as 1/σ, because the molecule has a
  fixed size and the noise does not.
* **Posterior-mean shrinkage behaves the same way.** For the MSE-optimal
  predictor,

      E‖F̂‖² / E‖F‖² = 1 − E tr Cov[x₀ | x_t] / (3(N−1)σ²)

  The posterior covariance is bounded by the size of the molecule, so this
  ratio tends to 1 as σ grows. Its size is modest, though: in the toys of
  §6.4 it is 2–5% at 19 atoms, below the ±9% per-sample spread. The
  Hungarian term (up to 13%) probably dominates the mid-band bias.
* **The bottom of the range has the opposite sign.** On clear-chimp
  validation rows, the soft-sigma-weights study measured a fixed-point bias of
  b = 0.026 Å per atom. That bias puts a floor of about `b/√3 ≈ 0.015 Å` on σ̂,
  which is above σ_min = 0.01 Å.

σ̂ is therefore biased:

| σ range | bias of σ̂ |
|---|---|
| ~0.5–5 Å | low |
| near σ_max | roughly unbiased |
| bottom of the range | high (floor) |

Calibration (§3.2) is still the right first move, because it absorbs all
three effects. Two adjustments:

* Calibrate as a function of N. Both shrinkages depend on molecule size and
  on how many atoms are exchangeable.
* Invert the curve only above the floor.

**The consequence for sampling is not simply "steps too small".**
`AdaptiveStochasticHeunSampler` derives `σ_next` from σ̂
([`src/gpff/sample/heun.py:265`](../src/gpff/sample/heun.py)). A low σ̂ makes it
believe it is further along the schedule than it is. Measure which way it
fails rather than predicting it.

**Ties are not an edge case.** Reassignment reaches half of the samples at
σ ≈ 0.7 Å, and the training median is σ = 0.5 Å. The discontinuity therefore
touches a large share of training draws.

The blob is synthetic. Rerun the simulation on real QM9 before quoting
numbers.

### 5.2 Softmin at β = 1/(2σ²) is the no-permutation optimum

An equivariant network cannot tell which same-element atom "was" atom `i`.
Training with identity labels (`experiment=gpff_no_permutation`) therefore
already converges to the posterior over labelings. With
`p(P | x_t, x₀) ∝ exp(−‖x_t − P x₀‖² / 2σ²)`, the per-sample expected label is
exactly the softmin target with β = 1/(2σ²). Its expectation over `x₀` given
`x_t` is the identity-label regression target.

Consequences:

* **Same optimum, less label noise.** Softmin at β = 1/(2σ²) has the same
  minimizer as `gpff_no_permutation`, with lower-variance labels (it is the
  Rao–Blackwellized version). Hungarian matching is the β → ∞ end.
* **β is a real knob.** It interpolates between a mean-seeking target
  (posterior mean) and a mode-seeking one (Hungarian). The existing `gpff`
  versus `gpff_no_permutation` comparison is already the ablation of the two
  endpoints. If Hungarian wins it, part of its "bias" is helping, through
  sharper targets. The fix is then a β schedule, not the exact posterior.
* **Cost.** The full softmin over the permutation orbit is a permanent, which
  is #P-hard to compute. The force needs only the expected permutation
  matrix, `F = 2(E_w[P] x₀ − x_t)`, and per-element Sinkhorn approximates it.
* **It does not remove the bias.** The soft target shrinks too, because it is
  a posterior mean. Calibration (§3.2) is still needed.

### 5.3 Per-atom σᵢ is not identifiable from one atom

A single atom has 3 degrees of freedom. `‖x_t,i − x₀,i‖ / σᵢ` follows a χ₃
distribution, whose relative spread is about 0.42. Condition 4 of §2.3 fails
at the scale of one atom.

* Draw the multipliers spatially coherent: per fragment, from
  scaffold/linker masks, or as smooth fields over space.
* Read σ̂ pooled over a neighbourhood.

The same applies to freezing converged atoms (§3.3).

### 5.4 "Zero retraining" versus out-of-distribution

§4.1 says per-atom inpainting, re-noising and freezing come with zero
retraining. §3.3 says mixed-noise states are out of distribution. Both are
true. The sampler can express these operations today, but the model is only
competent at them after per-atom σ training. Until then they are the
robustness exploit §3.3 describes.

### 5.5 Freezing atoms does not reduce NFE

A forward pass covers the whole structure, so freezing atoms does not reduce
the number of calls. It saves compute only if each step recomputes just the
neighbourhood of the moving atoms. That is a real saving for a clamped
protein pocket, and a negligible one on QM9. Count atom-updates or wall-clock
time rather than NFE.

### 5.6 L-BFGS faces drifting curvature, not one quadratic

The Jacobian of the learned field is `∂F̂/∂x_t = 2(Cov[x₀ | x_t]/σ² − I)`.

* **At high σ** it is about `−2I`. A step of ½ is then exactly the Newton
  step and lands on the posterior mean, so direct denoising is already optimal
  there. The bottleneck is posterior-mean blur, not ill-conditioning.
* **As σ̂ falls**, the curvature drifts, so secant pairs from earlier steps go
  stale.

Expect L-BFGS gains in the final low-σ relaxation. Implementation notes:

* use a short memory, or reset it;
* seed `H₀` from σ̂;
* use force-only variants (L-BFGS with a max step, as in ASE, or FIRE),
  because there is no energy for a Wolfe line search.

The heads output forces directly (`GatedEquivariant`), so the field is not
guaranteed to be conservative.

### 5.7 Langevin samples isomers at a self-chosen σ

* **The basins are isomers.** The composition is fixed, so the basins of the
  pseudo-PES are all isomers of the formula plus their conformers.
  Unconditional NVT hops between molecules; it does not sample a
  thermodynamic ensemble of one molecule.
* **The σ is self-chosen.** Thermal noise raises σ̂, and the model then pulls
  toward a blurrier posterior mean. The stationary state sits at a
  self-consistent σ*(T), so the dynamics sample a blurred `p_σ*` rather than
  a thermal ensemble.
* **No free energies.** A non-conservative field has no Boltzmann stationary
  distribution, so umbrella-sampling or WHAM free energies are undefined
  without a potential.

The idea becomes meaningful once sampling is conditioned on a bond graph or a
scaffold. Characterize σ*(T) first.

### 5.8 What the internal-coordinate PES gives up

* **Diffusion samplers.** The PES is quadratic in internal coordinates, not
  Cartesian ones. By §2.3 it keeps only the MLFF half: direct denoising, but
  no Heun or ancestral sampling.
* **Readout at high σ.** Angles and torsions are compact, so their terms
  saturate past σ ≈ 1 Å. This is the torus problem from §2.3, reappearing
  inside the PES. Only distance terms keep the readout monotone.
* **Degeneracies.** Bond and angle terms are reflection-invariant and blind to
  torsion, so enantiomers and rotamers are degenerate. Chirality or improper
  terms are needed.
* **Non-convexity.** The PES is non-convex at high σ; distance geometry has
  spurious minima.

A hybrid is safer: a Cartesian target at high σ, blended into the internal
target at low σ.

**Payoff.** The soft-sigma-weights study attributes about ⅓ of the small-σ
residual to rotation noise (`align_rotation` is off). A shape target removes
that at the source.

**Prior art.** Before calling the MLFF-pretraining claim tautological,
position it against denoising as force-field learning (Zaidi et al.), Frad,
and SliDe (bond/angle/torsion force-field denoising).

### 5.9 Two missing cutoff options

* **A kNN graph** never isolates an atom.
* **A single virtual node.** At high σ the optimal force is close to "pull
  toward the centroid at a scale set by σ". That depends only on global
  quantities, so one virtual node may cover the high-σ regime while local
  messages handle low σ.

The cheap geometric estimate needed by §3.5 comes from the radius of
gyration: `E R_g(x_t)² = R_g(x₀)² + 3σ²(N−1)/N`. Use the dataset average of
`R_g(x₀)` as a function of N.

### 5.10 Where the σ mass currently sits

Under `ln σ ~ N(−0.7, 1.2²)`:

| quantity | value |
|---|---|
| median σ | 0.5 Å |
| P(σ > 5 Å) | ≈ 2.7% |
| P(σ > 10 Å) | ≈ 0.6% |

The sampler starts at 30 Å. The top 1.5 decades of every trajectory, which
is where the cutoff and σ̂ issues live, are therefore learned from under 3% of
draws.

When retuning, use a mixture: a lognormal centred on the measured topology
band, plus a log-uniform floor over [σ_min, σ_max] so the readout stays
covered everywhere. Two measurements decide where the band is:

* bond-graph agreement against σ on clean data (no model needed);
* per-σ reducible loss from existing checkpoints, using the eval-noise data.

### 5.11 No multi-basin code yet

`src/` has no softmin, logsumexp or multi-basin code on `feat/soft-sigma-weights`.
The "Hungarian is the β → ∞ case of existing machinery" framing (§3.1) needs
that code to exist, or needs to say it is new.

## 6. What the label-noise literature offers

### 6.1 Why most of it does not transfer

The classification label-noise literature assumes that labels are wrong and
that the corruption process is unknown. Neither holds here:

* `F = 2(x₀ − x_t)` is a one-sample, unbiased estimate of the target the
  network should learn, `E[F | x_t]`.
* The training code generates the noise, so its process is known exactly.

Mean-squared error is already the consistent loss for this setting. Three
standard tools are traps:

* **Robust losses** (L1, Huber, GCE, truncation) move the minimizer from the
  conditional mean to a median or a mode. The output stops being a score,
  which breaks Tweedie, the σ̂ readout, and Heun or ancestral sampling (§2.3).
* **Small-loss selection and co-teaching** drop high-loss samples. In GPFF
  those are the mid-σ draws, where labels are noisiest and most of the
  learning happens, so dropping them biases the field.
* **Transition-matrix estimation** is unnecessary, because the noise process
  is known.

### 6.2 Two kinds of label noise

1. **Built-in.** The label scatters around its mean with variance
   `Var[F | x_t] = 4 tr Cov[x₀ | x_t]`. That variance is small at low σ,
   where x₀ is pinned down, small relative to the signal at high σ, and
   largest in the 0.5–5 Å band. Hungarian matching adds a biased, structured
   component. That is the one place where the classical view applies, and
   because the process is known the correction can be exact: softmin over
   atom assignments (§5.2) is loss correction (Patrini et al. 2017) with a
   known transition.
2. **In the data.** x₀ itself is imperfect. This is negligible for
   DFT-relaxed QM9 and dominant for PDB pockets and xtb-level conformers
   (§6.8).

### 6.3 Lower-variance targets

Each of these keeps the optimum and lowers the label variance
(Rao–Blackwellization).

* **Softmin over atom assignments** with β = 1/(2σ²) (§5.2).
* **Stable Target Field** (Xu, Tong & Jaakkola, ICLR 2023). The target is a
  posterior-weighted average over a reference batch of other molecules with
  the same formula. QM9 has many: 6,095 isomers of C₇H₁₀O₂ alone. In the
  terms of §3.1, this is a multi-reference pseudo-PES whose reference set is
  the dataset neighbourhood, alongside atom assignments and conformers. Each
  reference must be aligned (permutation and rotation) to `x_t`, so limit it
  to the mid-σ band, where label variance is largest.
* **Bootstrapping with an EMA teacher** (Reed et al. 2014). Train on
  `(1 − α) F + α F̂_EMA`. If the teacher has learned the conditional mean, the
  optimum does not move and the variance drops. The risk is that the model
  reinforces its own errors, so ramp α up slowly. The EMA callback already
  exists.
* **K noise draws per loaded molecule.** Training was input-bound on database
  reads, so extra label draws per x₀ cost no extra I/O.

### 6.4 Stein and Tweedie identities

Second-order Tweedie links the posterior covariance to the Jacobian of the
learned field. With `x̂₀ = x_t + F̂/2` and `d = 3(N − 1)`:

    ∂x̂₀/∂x_t = Cov[x₀ | x_t] / σ²   ⇒   tr Cov[x₀ | x_t] = σ² (d + ½ ∇·F̂)

The divergence `∇·F̂` comes from a finite difference of `F̂` along one
random ±1 probe (Hutchinson's estimator), at the cost of one extra forward
pass. Combine it with the decomposition from §5.1,
`E‖F‖² = ‖F̂‖² + 4 tr Cov[x₀ | x_t]` with `E‖F‖² = 4dσ²`. Then d cancels, and
σ follows from the field alone. This is Hyvärinen's identity, `‖s‖² = −∇·s`,
used as an estimator:

    σ̂² = ‖F̂‖² / (−2 ∇·F̂)

As a sanity check, take a single reference point: `F̂ = 2(x₀ − x_t)` and
`∇·F̂ = −2d`, which gives `σ̂² = ‖x_t − x₀‖² / d`, as it should.

Appendix B tests this against the naive readout `std(F̂/2)`. It uses the
exact Bayes-optimal time-agnostic field, with the lognormal σ prior from
§3.4:

| toy | σ | naive σ̂/σ | Stein σ̂/σ |
|---|---|---|---|
| 400 random points, d = 57 | 3 Å | 0.954 | 1.001 |
| 400 random points, d = 57 | 5 Å | 0.968 | 1.006 |
| 5 templates × 400 rotations, d = 57 | 2–5 Å | 0.96–0.99 | 0.99–1.03 |

Findings:

* **The identity holds, and one probe is enough.** A single probe and the
  exact divergence agree within the sample spread.
* **The bias it corrects is small.** In the toys, posterior shrinkage is
  2–5%, below the ±9% sample-to-sample spread (≈ 1/√(2d) for 19 atoms).
* **It cannot correct the Hungarian bias.** The identity needs `F̂` to be a
  score, and the Hungarian field is not the score of any distribution. It
  applies exactly only to the softmin or no-permutation model.
* **It assumes `p(σ | x_t)` is peaked.** Concentration (§2.3, condition 4)
  provides that, and it held in both toys.

The more durable use is **uncertainty without training**. The per-atom
version, `σ² (3 + ½ ∂ᵢ·F̂ᵢ)` built from atom i's 3×3 diagonal block of the
Jacobian, estimates how uncertain the denoiser is about each atom. It can
drive the freeze criterion (§3.3), step-size control, and a confidence score
per generated molecule. A single probe mixes in cross-atom terms, so average
it over a few probes or steps. Analytic-DPM (Bao et al., ICLR 2022) uses the
same score-norm identity to set reverse-process variances.

### 6.5 Variance head

Add a scalar output per atom, `v̂ᵢ`, that predicts the squared residual
`‖Fᵢ − F̂ᵢ‖²`. Train it on detached residuals so the force fit is untouched
(Kendall & Gal 2017; Seitzer et al., ICLR 2022, on why joint Gaussian-NLL
training degrades the mean). It can be trained on a frozen trunk in hours.
It gives what §6.4 gives without the extra forward pass:

    σ̂² = (‖F̂‖² + Σᵢ v̂ᵢ) / (4d)

Being learned, it also works for the Hungarian model.

### 6.6 Weighting from the noise decomposition

The loss at each σ splits into an irreducible part, `4 E tr Cov[x₀ | x_t]`,
and a reducible part. Measuring the irreducible floor per σ with §6.4 or §6.5
gives two things:

* **A principled weighting to test against σ⁻².** The textbook
  weighted-least-squares choice is inverse noise variance. In time-agnostic
  training, weights also tilt `p(σ | x_t)`, but the soft-sigma-weights study
  found that effect small below 0.5 Å.
* **A measured version of EDM's rule for the σ distribution** (§3.4, §5.10):
  put mass where the reducible part is large.

### 6.7 Less noisy evaluation

For models A and B scored on the same draws, the loss difference contains a
pure label-noise term. With `m = E[F | x_t]`:

    ΔL = [‖F̂_A − m‖² − ‖F̂_B − m‖²] − 2⟨F̂_A − F̂_B, F − m⟩

The eval-noise study found that σ explains only about 11% of
paired-difference variance, and the cross term is a candidate for much of the
rest. Scoring against a lower-variance target (at minimum, softmin over
assignments, §6.3) shrinks it without changing the expected ranking. When
comparing models trained with different matching, fix one target and state
which.

### 6.8 Noise in the data

These tools matter for scaling beyond QM9.

* **Ambient diffusion with Tweedie consistency** (Daras, Dimakis &
  Daskalakis, ICML 2024). A structure with Gaussian-like coordinate error of
  known size σ_n is a sample at σ_n. Under VE noise,

      E[x₀ | x_t] = (σ_t² E[x_n | x_t] − σ_n² x_t) / (σ_t² − σ_n²)

  So train on such structures only at σ_t > σ_n, with the inflated label
  `2σ_t² / (σ_t² − σ_n²) · (x_n − x_t)`, and let clean DFT data cover low σ.
  The factor diverges as σ_t → σ_n, so keep a margin. This applies directly
  to PDB ligands.
* **Noise2Noise** (Lehtinen et al. 2018). Given two noisy observations of the
  same structure, such as MD snapshots, noise one and use the other as the
  target; the model learns their mean. GPFF training already has this shape.
  This is more speculative than the other tools.
* **Confident learning** (Northcutt, Jiang & Chuang 2021). On a clean
  structure, `‖F̂(x₀)‖/2` is the model's estimate of how far x₀ is from where
  it should be. It costs one forward pass per molecule and needs no noise.
  - Use out-of-sample predictions (a test split or k-fold), because the model
    has fit its own training molecules.
  - Validate it for free before trusting it on GEOM or PDB data: does it flag
    the 3,054 molecules the QM9 authors list as failing their geometry
    consistency check?

## 7. Suggested order

| # | step | retrain | why here |
|---|---|---|---|
| 1 | Read the mean σ̂/σ curve off the existing eval-noise validation rows (§5.1, §6.4) | no | Zero compute; shows whether the bias is a few percent (posterior) or ~13% (Hungarian) |
| 2 | Geometry metrics (§3.6), run together with σ̂ calibration per N (§3.2, §5.1) | no | "Fewer NFE at equal quality" needs a quality metric that sees geometry |
| 3 | On real QM9: shrinkage and tie rate against σ; bond-graph dissolution curve; per-σ reducible loss; existing `gpff` versus `gpff_no_permutation` results | no | Decides whether the softmin and σ-retune runs are worth it |
| 4 | Out-of-sample `‖F̂(x₀)‖` audit, including the uncharacterized QM9 molecules (§6.8); paired-difference variance against a softmin target (§6.7) | no | Validates the audit tool; sharpens every later ablation |
| 5 | Force-only L-BFGS / FIRE with σ̂-seeded `H₀` (§4.1, §5.6); Stein divergence as per-atom uncertainty (§6.4) | no | Report NFE and wall-clock time |
| 6 | One change per run, with paired σ draws: K draws per molecule; σ distribution; Sinkhorn β schedule or Stable Target Field; EMA bootstrapping; region-coherent σᵢ | yes | Single-variable ablations |
| 7 | Conditional and scale work: match free atoms only; kNN or virtual node; capped shaped prior; internal-coordinate PES; ambient training for pocket data (§6.8) | yes | Needed before pockets |
| 8 | Langevin and enhanced sampling (§5.7) | no | Only after conditioning exists; otherwise it samples isomers |

## 8. Candidate paper contributions

1. **Admissibility criterion** (§2.3): what makes a time-agnostic generative
   force field possible, and what Gaussian corruption with a quadratic PES
   adds on top.
2. **Multi-reference pseudo-PES** (§3.1, §5.2, §6.3): β runs from the
   posterior (identity labels) to hard-min (Hungarian). Conformers, reaction
   pairs and same-formula dataset neighbours enter as reference sets.
3. **Read-back sampling** (§2.4, §3.3, §4.1): per-atom σ, mid-trajectory
   re-noising, and a self-correcting response to a mismatched prior.
4. **Geometry-aware evaluation** (§3.6).

---

## Appendix A — shrinkage simulation

Produces the table in §5.1. Run it with
`uv run --with numpy --with scipy python shrink.py`.

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

rng = np.random.default_rng(0)
# QM9-like composition C7 O2 H10 (19 atoms), packed blob, min separation 1.0 A
Z = np.array([6] * 7 + [8] * 2 + [1] * 10)


def make_mol():
    pts = []
    while len(pts) < len(Z):
        p = rng.normal(size=3) * 1.8
        if all(np.linalg.norm(p - q) > 1.0 for q in pts):
            pts.append(p)
    x = np.array(pts)
    return x - x.mean(0)


def hungarian_sq(xt, x0):
    tot = 0.0
    for z in np.unique(Z):
        m = Z == z
        c = ((x0[m][:, None] - xt[m][None]) ** 2).sum(-1)
        r, k = linear_sum_assignment(c)
        tot += c[r, k].sum()
    return tot


mols = [make_mol() for _ in range(8)]
print("sigma  sqrt(E|matched|^2/E|identity|^2)  fraction_reassigned")
for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2, 3, 5, 10, 30]:
    num = den = 0.0
    reass = 0
    for x0 in mols:
        for _ in range(60):
            e = rng.normal(size=x0.shape)
            e -= e.mean(0)  # mean-free noise, as in ve_geometries.yaml
            xt = x0 + s * e
            h = hungarian_sq(xt, x0)
            i = ((xt - x0) ** 2).sum()
            num += h
            den += i
            reass += h < i - 1e-9
    print(f"{s:5.2f}  {np.sqrt(num / den):.3f}  {reass / (len(mols) * 60):.2f}")
```

## Appendix B — Stein estimator toy

Produces the rotation row of the table in §6.4: 5 random 19-atom templates,
each under 400 random rotations (`align_rotation` is off in GPFF). `F_hat` is
the exact Bayes-optimal time-agnostic field, with the posterior taken jointly
over data point and σ under the §3.4 prior. Each σ uses 30 draws, so the
standard error of each mean is about 0.02.

For the random-point rows, use seed 0 and replace `X0` with
`rng.normal(size=(400, 57)) * 1.5`. Those rows use the exact divergence
(central differences along every coordinate); a single probe agrees within
the sample spread.

Run it with `uv run --with numpy --with scipy python stein_rot.py`.

```python
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

rng = np.random.default_rng(1)


def make_mol(n=19):
    pts = []
    while len(pts) < n:
        p = rng.normal(size=3) * 1.8
        if all(np.linalg.norm(p - q) > 1.0 for q in pts):
            pts.append(p)
    x = np.array(pts)
    return x - x.mean(0)


T = [make_mol() for _ in range(5)]
X0 = np.stack([
    (Rotation.random(random_state=rng).as_matrix() @ t.T).T.ravel()
    for t in T
    for _ in range(400)
])
M, d = X0.shape
sig_grid = np.exp(np.linspace(np.log(0.01), np.log(30), 300))
lw = -0.5 * ((np.log(sig_grid) + 0.7) / 1.2) ** 2  # lognormal prior on a log grid


def F_hat(x):
    d2 = ((X0 - x) ** 2).sum(1)
    lp = lw[None, :] - d2[:, None] / (2 * sig_grid ** 2) - d * np.log(sig_grid)
    lp -= logsumexp(lp)
    return 2 * (np.exp(logsumexp(lp, axis=1)) @ X0 - x)


def div_hutch(x, h, k=4):
    tot = 0.0
    for _ in range(k):
        v = rng.choice([-1.0, 1.0], size=d)
        tot += v @ (F_hat(x + h * v) - F_hat(x - h * v)) / (2 * h)
    return tot / k


print(" sigma | naive/sigma     | stein(4 probes)/sigma")
for s in [0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 30]:
    nv, st = [], []
    for _ in range(30):
        x = X0[rng.integers(M)] + s * rng.normal(size=d)
        F = F_hat(x)
        nv.append(np.sqrt((F ** 2).mean()) / 2 / s)
        st.append(np.sqrt(max((F @ F) / (-2 * div_hutch(x, 1e-3 * s)), 0)) / s)
    f = lambda a: f"{np.mean(a):.3f}±{np.std(a):.3f}"
    print(f"{s:6.2f} | {f(nv):>15} | {f(st):>15}")
```
