# GPFF internal diagnostics

**Status: proposal.** Companion to [`geom_drugs_benchmark.md`](geom_drugs_benchmark.md). That
document measures *what comes out*; this one measures *what the model is*. The benchmark tells you
a model is worse; these diagnostics tell you which knob to turn.

Every diagnostic below states what it measures, how to compute it, what the *exact* value would
be, and which configuration option it implicates. The ones that need no labels and no sampling run
are marked **(free)** — those should go into CI and into the training logs.

---

## What GPFF actually is (the facts the diagnostics hang off)

From `configs/experiment/gpff.yaml`, `pseudo_pes/`, `data/transform/pseudo_pes.py`,
`model/heads.py` and `sample/base.py`:

* **Target.** The pseudo-PES is `E(x_t; x_0) = ‖x_t − x_0‖²` (`VEPseudoCalculator`), so the
  analytic label is a **pseudo-force**

  ```
  F*(x_t ; x_0) = −∇_{x_t} E = 2 (x_0 − x_t)                               (per atom, 3-vector)
  ```

* **Model.** PaiNN (`n_atom_basis=256`, `n_interactions=4`, fully connected neighbour list) with a
  `GatedEquivariant` head that emits **forces directly** — there is no energy output and no
  autograd differentiation. The field is therefore **non-conservative by construction**.
* **Time-agnostic.** `experiment=gpff` uses the plain `GatedEquivariant` head: the network never
  sees `σ` or `t`. It has to *infer* the noise level from the geometry. (`gpff_t_aware` is the
  conditioned variant.)
* **Noise.** `VarianceExplodingGeometries(sigma_max=30, sigma_mean=-0.7, sigma_std=1.2,
  mean_free_noise=true)`: `σ ~ LogNormal(−0.7, 1.2)` with rejection above 30, `x_t = x_0 + σ ε`,
  `ε` centred per molecule.
* **Loss.** `SigmaWeights` → `force_weight = 1/σ²`, **clipped to `[0, 1]`**; squared error,
  `mean(w · (F̂ − F*)²)`.
* **Alignment.** `align_permutation=true` — the target `x_0` is Hungarian-matched to `x_t` before
  the label is formed.
* **Sampler.** `DirectDenoisingSampler` iterates `x ← x + F̂/2` (i.e. `x ← x̂_0`) with an annealed
  stochastic perturbation, 50 steps, prior `N(0, 30²)`.
* **σ estimation.** `Calculator._compute_std_per_molecule` sets `σ̂ = std(F̂/2)` over the molecule's
  flattened `3N` values, and the adaptive samplers steer on it.

Two exact identities make most of what follows falsifiable rather than merely descriptive. Writing
`D(x) = E[x_0 | x_t = x]` for the true posterior mean at noise level σ:

```
F*(x) = 2 ( D(x) − x )                                    [the exact pseudo-force]
∇_x log p_σ(x) = ( D(x) − x ) / σ²  =  F*(x) / (2σ²)      [Tweedie]
∂D/∂x = Cov( x_0 | x_t = x ) / σ²                         [symmetric, positive semi-definite]
```

So the ideal GPFF field is (i) a gradient field, (ii) with a symmetric PSD Jacobian up to the
`−2I` shift, and (iii) it vanishes exactly on the data manifold. A direct force head is not
constrained to any of those. Each is measurable.

---

## D1. Is the learned field a valid score field? **(free)**

The head emits a force without differentiating an energy, so there is no guarantee a potential
exists. This is the most fundamental thing to check and it needs no data labels.

### D1.1 Jacobian symmetry defect

Let `J(x) = ∂F̂/∂x` (`3N × 3N`). The exact field has `J* = 2(Cov/σ² − I)`, which is **symmetric**.
Report

```
asym(x) = ‖ J − Jᵀ ‖_F  /  ‖ J ‖_F        ∈ [0, 1]
```

Cheap estimator, no `J` materialised: for random unit `v`, compute `Jv` by forward-mode `jvp` and
`Jᵀv` by `vjp`, then `‖Jv − Jᵀv‖ / ‖Jv‖` averaged over `v` is an unbiased probe of the defect
direction-by-direction. For molecules up to ~30 atoms, materialise `J` exactly with `3N` JVPs and
report the spectrum of the antisymmetric part too.

Report `asym` as a function of σ and of molecule size. **Interpretation:** `asym` is pure model
error — the exact target has none of it. A large `asym` at the small-σ end is the worst case,
because that is where the final geometry is decided.

### D1.2 Loop circulation — "does a PES exist at all"

For a closed loop `γ` in configuration space, a conservative field satisfies `∮_γ F̂ · dl = 0`.
Take small rectangular loops in random 2-planes: pick orthonormal `u, w` (3N-dimensional,
projected orthogonal to the 6 rigid-body modes), side `h`, and evaluate the circulation by
midpoint quadrature on the four edges. Report the distribution of

```
C(h) = | ∮ F̂ · dl |  /  ( h² · ‖J‖ )        [dimensionless, ~ the local curl magnitude ]
```

versus `h ∈ {0.01, 0.05, 0.2, 1.0} Å` and versus σ.

**Why this is the key diagnostic, not a curiosity.** Non-zero circulation means the "learned PES"
of the title is not single-valued: the energy you get by integrating the force depends on the path.
It also makes a *testable prediction about samplers*: the deterministic ODE samplers (`heun`,
`adaptive_heun`) integrate the field as if it were a gradient flow, while `direct_denoising`
performs a fixed-point iteration that is far less sensitive to curl. So **if circulation is large,
Heun and direct denoising must systematically disagree**, and the disagreement should scale with
`C`. Measure both and check. If they agree, curl is not your problem; if they diverge, you have
localised a real defect.

**Fixes this points at**, in increasing order of cost:
1. Predict an energy and take `F = −∇E` by autograd. The machinery is already half-present:
   `PseudoCalculator.get_forces` does exactly this for the *labels*, and `heads.Atomwise` can
   produce an energy. Cost: one backward pass per model call, and gradients through it in training.
2. Keep the direct head but add a curl penalty: sample random `v`, penalise `‖(J − Jᵀ)v‖²`
   (two JVP/VJP calls, no `J`).
3. Accept non-conservativity and drop the ODE samplers — but then stop calling the learned object
   a PES.

### D1.3 Positive-semidefiniteness defect

For `D̂(x) = x + F̂/2`, the exact `∂D/∂x = Cov(x_0|x)/σ²` is symmetric **PSD** with eigenvalues that
→ 0 as σ → ∞ (since `Cov → Σ_data`) and stay bounded as σ → 0. Report, on the symmetric part
`S = (J_D + J_Dᵀ)/2`:

* `λ_min(S)` (should be ≥ 0),
* the fraction of eigenvalue mass that is negative,
* `tr(S) · σ²` compared with `tr(Σ_data)` at large σ and with the local vibrational curvature at
  small σ.

A negative eigenvalue means the implied posterior covariance is not a covariance — i.e. `F̂` is not
the pseudo-force of *any* denoising problem at that point. Lanczos on `S` via JVP/VJP gives
`λ_min` and `λ_max` without materialising anything.

---

## D2. Is the data manifold the fixed point?

`direct_denoising` iterates `x ← x + F̂/2`. Its fixed points are exactly `{x : F̂(x) = 0}`. The
model is only usable if that set is the data manifold, and the *rate* at which the iteration
approaches it is what sets how few steps you can get away with.

### D2.1 Force residual on clean data — the single best scalar **(free, one forward pass)**

Evaluate `F̂` on held-out **test conformers** (`σ = 0`). Report `‖F̂(x_0)‖` per atom: distribution,
median, tail, stratified by element, by molecule size, and by local environment. It should be ≈0.

Also report `σ̂(x_0) = std(F̂(x_0)/2)` on clean data — the model's own noise-level estimate on
noise-free input. If it is not ≈0, the σ estimator is biased exactly at the low-σ end where
geometry is decided, and every adaptive sampler inherits that bias (D3.1).

This costs one forward pass over the test split and is the first thing to run on any new
checkpoint. Add it to `SamplerCallback` so it is logged during training.

### D2.2 Contraction spectrum at the fixed point — predicts the step count

Let `T(x) = x + F̂(x)/2` be the direct-denoising map and `J_T = ∂T/∂x` evaluated at a test
conformer. Project out the six rigid-body modes: three translations lie exactly in the kernel by
translation equivariance, and three rotations are tangent to the fixed-point *orbit* (a rotated
molecule is an equally valid fixed point), so they are not errors. On the orthogonal complement,
report the spectral radius

```
ρ = max { |λ| : λ ∈ spec(J_T) , restricted to the complement of the 6 rigid modes }
```

**Interpretation, and why this is worth the compute.** `ρ < 1` ⇒ direct denoising converges
locally, geometrically, at rate `ρ`. The number of steps to reach accuracy `tol` from distance `d`
is `log(tol/d) / log ρ`. So:

* `ρ ≈ 0.3` → ~10 steps to 0.01 Å. The few-step claim holds.
* `ρ ≈ 0.95` → ~130 steps. The default `n_steps=50` is not converged, and the Tier-B geometry
  numbers are measuring an unconverged iterate, not the model.
* `ρ > 1` → the iteration is locally *repelling*; samples will oscillate around the data manifold
  and the stochastic perturbation is the only thing keeping them in place.

This converts a model property into a sampler setting, and it explains the `n_steps` sweep in
§3.7 of the benchmark instead of just reporting it.

**The eigenvectors are as informative as the eigenvalue.** Project the slowest eigenvectors onto
internal coordinates (bond stretches, angle bends, torsions). If the slow modes are torsions, the
model resolves local chemistry but converges slowly in conformational space; if they are bond
stretches, the short-range representation is the bottleneck — which ties directly to the radial
basis audit (D4.4) and to which of `Δr / Δθ / Δφ` is worst in the benchmark's B3.

### D2.3 Consistency along a noise ray

For a fixed `x_0` and a fixed `ε`, form `x_t(σ) = x_0 + σ ε` for a geometric ladder of σ, and
compute the one-shot estimate `x̂_0(σ) = x_t(σ) + F̂(x_t(σ))/2`. A consistent model gives
`x̂_0(σ) ≈ x_0` for every σ. Report

```
consistency(σ) = ‖ x̂_0(σ) − x_0 ‖ / √(3N)        [Å per coordinate]
```

as a curve, plus the cross-σ spread `‖x̂_0(σ_i) − x̂_0(σ_j)‖`. This is precisely the
consistency-model property that `direct_denoising` assumes when it jumps straight to `x̂_0`. The
curve's shape is diagnostic: a monotone decrease is healthy; a bump at intermediate σ localises
where the single time-agnostic network is being pulled in two directions (and D5.3 confirms it
from the gradient side).

---

## D3. σ-conditioning: the implicit clock

`experiment=gpff` is time-agnostic, so σ is a *latent* the network must read off the geometry.
Everything downstream — the sampler schedule, the convergence test, the adaptive step size —
depends on how well it does that.

### D3.1 σ̂ calibration

Plot `σ̂ = std(F̂/2)` against the true σ on log–log with the identity line, for σ across the full
training range. Report bias and spread per σ decade and **per molecule size**. The estimator's
rationale is exact: `F*/2 = σ ε` with `ε` standard normal, so `std(F*/2) → σ`. But the sample std
over `3N` values has relative standard error `≈ 1/√(2·3N)` — about 10% at `N = 18` (QM9) and 4% at
`N = 120` (GEOM-Drugs). So report the N-dependence, and separate *estimator noise* (which shrinks
with N and is harmless) from *model bias* (which does not).

This matters because `AdaptiveHeunSampler` and the `f_max` convergence test steer on `σ̂`. A 20%
low bias at small σ makes molecules freeze early with residual force still present — which shows
up in the benchmark as elevated `ΔE_relax` with a *short* NFE tail, a signature you can look for.

### D3.2 A σ probe on frozen features

Freeze the backbone, train a small linear/MLP read-out from `scalar_representation` to `log σ`.
Report its accuracy against `σ̂`'s.

* probe ≈ `σ̂` → the force magnitude is already using all the σ information in the features;
* probe ≫ `σ̂` → **the information is in the representation but the head is not using it.** That is
  a concrete architectural fix with a measured justification: feed the probe's estimate back in
  (which is what `heads.Atomwise(time_conditioning=True, time_head=DiffusionTime(...))` and
  `TimeAwareEquivariant` already implement), rather than guessing that time conditioning helps.
* probe also poor → σ is genuinely not recoverable from the features; the architecture or the
  cutoff/basis is losing it.

### D3.3 The effective loss weighting — likely the highest-value plot in this document

Plot, on one log-σ axis, four curves:

1. `p(σ)` — the sampler: `LogNormal(−0.7, 1.2)` truncated at 30 (median σ ≈ 0.50).
2. `w(σ) = clip(1/σ², 0, 1)` — the loss weight actually used.
3. `E‖F*‖² ∝ 4 σ² · 3N` — the target's own scale.
4. **the product** `p(σ) · w(σ) · E‖F*‖²` — what each σ contributes to the gradient.

Work out curve 4 by hand first, because the answer is a finding:

```
σ ≥ 1 :  w = 1/σ²   ⇒  w · E‖F*‖²  ∝  (1/σ²) · 12Nσ²  =  12N        (flat)
σ < 1 :  w = 1      ⇒  w · E‖F*‖²  ∝  1 · 12Nσ²       =  12N σ²     (→ 0 as σ → 0)
```

**The `max_clipping: 1` in `sigma_weights.yaml` makes the small-σ regime contribute to the loss
like σ², i.e. essentially nothing — exactly where bond lengths, bond angles and ring planarity are
determined.** A model that output zero at σ = 0.05 would incur ~1/400 of the loss of the same
relative error at σ = 1. The aggregate `val/loss` is therefore almost blind to the quality of the
final geometric refinement, which is precisely what the GEOM-Drugs benchmark measures (B1–B4).

This is a prediction, and the benchmark can test it: if it is right, GPFF's `Δr` and `Δθ` will be
poor relative to its validity, and `ΔE_relax` will be dominated by stiff-coordinate error. The
candidate fixes are ordered and cheap to try:

1. **Remove `max_clipping`** (use the true `1/σ²`). This is the EDM-style weighting that makes the
   loss σ-uniform in relative terms. The clipping exists to tame large weights at tiny σ, so pair
   it with a floor `σ_min` on the sampler instead of a ceiling on the weight.
2. **Reparameterise to `x_0` prediction** with EDM preconditioning (`c_skip`, `c_out`, `c_in`,
   `c_noise`), which makes the effective target unit-variance at every σ by construction. The
   `bounded_forces` option in `heads.GatedEquivariant` is already a hand-rolled step in this
   direction (a skip term `−2α(x − x̄)` plus a softly clamped network term); EDM's version is the
   principled one.
3. **Reweight the σ sampler** rather than the loss, so the two are not fighting.

Alongside the plot, report **per-σ-bin relative validation error** `‖F̂ − F*‖ / ‖F*‖`, which is the
σ-resolved quality that the aggregate loss hides. This should be a standard logged metric, not a
one-off study.

### D3.4 Training-σ support versus the sampler's starting σ

`p(σ) = LogNormal(−0.7, 1.2)` puts almost no mass near 30, yet `GaussianPrior(sigma=30)` starts
every sampling run there. The numbers are easy to compute and they are stark:

```
median σ = e^(−0.7) = 0.50
P(σ > 1)  = 0.280
P(σ > 10) = 0.0062        ← 1 training sample in 160
P(σ > 20) = 0.0010        ← 1 in 1000
P(σ > 30) = 0.0003        (rejected by the sampler anyway)
```

So the sampler's entire first phase runs in a σ range the model saw in well under 1% of training
steps. Report:

* the training-σ histogram with the quantiles, and `P(σ > 10)`, `P(σ > 20)`, against the schedule
  the sampler actually traverses;
* `σ̂` evaluated on fresh prior draws — if `σ̂ ≪ 30`, the model does not recognise its own starting
  point;
* `‖F̂‖` on prior draws versus the exact `‖F*‖ = 2σ‖ε‖` at σ = 30.

If the first few denoising steps run outside the training support, the fix is one of: raise
`sigma_mean`/`sigma_std`, sample σ log-uniformly (`sample_uniformly` exists but is *linear* in σ —
log-uniform is the better option and is not currently implemented), or start sampling at a σ the
model has actually seen. This is cheap to measure and cheap to fix, and on GEOM-Drugs — where
`sigma_max` has to grow with molecule size (benchmark §2.3) — the mismatch will get worse, not
better.

---

## D4. The learned PES: landscape and smoothness

### D4.1 Reconstructing the surface

With no energy head, the "learned PES" has to be recovered by line integration:

```
Ê(x) = − ∫_{x_ref}^{x} F̂ · dl        along a fixed path (straight line from a reference x_0)
```

Fix the path convention and report D1.2's circulation as the **error bar on `Ê`** — that is the
honest way to present a potential reconstructed from a non-conservative field.

### D4.2 Two-dimensional slices

Pick a test molecule and two directions: (a) a noise direction `ε`, (b) either a specific internal
coordinate (a torsion, swept properly through the dihedral) or the slowest eigenvector from D2.2.
Grid them and plot `Ê` and `‖F̂‖`.

What to look for, stated as a hypothesis so the plot can refute it: **corrugation with a period
near 0.25 Å**, which is the `GaussianRBF` centre spacing at `n_rbf=600, cutoff=150`. If the
surface ripples at that scale, the fixed uniform radial grid is imprinting artificial structure on
the PES, and D4.4 quantifies it.

### D4.3 Lipschitz constants and feature growth **(free)**

Four complementary measurements, cheapest first:

**(a) Architectural upper bound (free, log every epoch).** The product of the spectral norms of the
weight matrices, and the per-layer norms. Loose, but it is *exactly* the quantity `BlockAdamW`'s
`block_update=spectral` option controls — so logging it closes the loop with the optimizer work
already in `gpff/optim.py` and turns "spectral updates helped/did not help" into a mechanism.

**(b) Local Lipschitz by power iteration.** `L(x) = ‖J(x)‖₂` via power iteration on `JᵀJ` using
JVP/VJP pairs — no `J`. Report the distribution of `L` versus σ and molecule size. The exact field
satisfies `‖J*‖ = 2‖Cov/σ² − I‖ ≤ 2 max(1, ‖Cov‖/σ²)`, which is bounded; a measured `L` far above
that means the model is over-sharp, and over-sharp fields need small steps.

**(c) Multiscale Lipschitz profile.** `sup ‖F̂(x) − F̂(y)‖ / ‖x − y‖` over sampled pairs at
separations `h ∈ {10⁻³ … 1} Å`. The `h → 0` end is `‖J‖`; the large-`h` end is what the sampler
actually experiences. A fixed-point iteration `x ← x + F̂/2` is a contraction when the field's
effective Lipschitz constant at the relevant scale is below 2, so this profile gives the
**maximum admissible step size** directly — i.e. it converts smoothness into "how few steps can
this model support", which is GPFF's headline claim.

**(d) Feature-growth audit.** Log per-block RMS of `q` and `μ` versus depth, molecule size and σ.
This is the failure mode `prenorm`, `degree_norm` and `bounded_forces` were added for: with an
all-to-all neighbour list, the message an interaction block sums grows with the number of
neighbours, so features that behave at QM9's 4–28 neighbours can diverge at GEOM-Drugs' ~180.
Pair this with the benchmark's size-extrapolation curve (D4 there) — the audit says *where* in the
network the growth happens, the curve says whether it matters.

### D4.4 Radial-basis resolution audit **(free, no training needed)**

`GaussianRBF(n_rbf=600, cutoff=150, trainable=False)` places centres `150/599 ≈ 0.25 Å` apart with
matching widths. Report:

1. the **effective number of active basis functions in the 0.9–1.8 Å bond window** (sum of
   normalised activations) — expect a single-digit number;
2. the **Gram-matrix spectrum / condition number** of the basis evaluated on the *data's actual
   pair-distance distribution*, which on a fully connected graph is dominated by long distances —
   so the short-range directions are poorly conditioned;
3. `∂φ(d)/∂d` at bond distances, i.e. how much the filter input moves for a 0.01 Å change — the
   precision the benchmark's `Δr` asks for.

Then repeat for `gpff.nn.radial.ExpGaussianRBF` (exponential distance mapping; dense at short
range, sparse at long) and for a Bessel basis. This is an architectural change with a *measured*
justification rather than a hunch, it needs no training run to motivate, and it is the most likely
single explanation for poor stiff-coordinate accuracy on GEOM-Drugs.

---

## D5. Loss landscape and optimisation

### D5.1 Filter-normalised loss surfaces

Two-dimensional slices around the final weights using filter-normalised random directions
(Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018). Specifically worth
checking here: **do the EMA weights sit in a flatter basin than the raw weights?** The repo uses
`ExponentialMovingAverage` by default and `ModelCheckpoint` saves a best *inference* model; this
plot verifies that choice rather than assuming it.

### D5.2 Hessian diagnostics

Top-`k` eigenvalues (Lanczos) and the trace (Hutchinson) of the training loss Hessian; report
`λ_max`, the bulk edge, and the negative-eigenvalue mass. **Compute them per σ-bin**, because the
loss is a mixture over σ and the regimes plausibly have different curvature — an aggregate Hessian
would average that away.

### D5.3 Gradient conflict between σ regimes — the test for time conditioning

Partition a large batch into σ-bins, compute the per-bin gradient `g_σ`, and report the pairwise
cosine-similarity matrix.

**This is the decisive experiment for whether `experiment=gpff` (time-agnostic) or
`experiment=gpff_t_aware` is the right architecture**, and it costs one batch. If small-σ and
large-σ gradients are anti-aligned, one network is being asked to do two conflicting jobs, and the
fixes are: condition on time, or give the output a σ-dependent scaling (EDM preconditioning), or
split heads. If they are aligned, time-agnosticism is free and the conflict hypothesis is dead —
which is equally useful to know, since it redirects effort to D3.3 and D4.4.

Cross-check against D2.3: a bump in the consistency curve at intermediate σ and a block structure
in the cosine matrix are the same phenomenon seen from two sides.

### D5.4 Optimisation hygiene **(free)**

* Per-parameter-group update RMS and spectral norm — `BlockAdamW` already computes these internally
  (`rms_norm`, `spectral_norm`); log them.
* Gradient-noise scale / critical batch size, to justify `batch_size` rather than inherit it.
* **Loss-spike forensics.** `AtomisticTask` already counts `train/skipped_nan` and can dump the
  offending batch to `debug_folder`. Correlate the spikes with σ, molecule size, and whether
  permutation alignment moved many atoms. Spikes concentrated at small σ with large weights would
  be a direct consequence of the `1/σ²` weighting and would argue for a σ floor instead of the
  weight ceiling (D3.3).
* **Model-selection audit.** `ModelCheckpoint` *and* `EarlyStopping` both monitor `val/loss`
  (`callbacks/checkpoint.yaml`, `callbacks/earlystopping.yaml`, patience 500). Track `val/loss`
  *and* a cheap subset of the Tier-B metrics over training (the `SamplerCallback` hook exists). If
  they decouple — val loss flat while `ΔE_relax` still improves, or vice versa — then two things
  are wrong at once: the checkpoint being shipped is not the best model under the metric anyone
  cares about, and early stopping may halt a run that was still improving on geometry. Given
  D3.3's analysis (the loss is weighted away from small σ, which is what the geometry metrics
  measure), decoupling is the *expected* outcome, and the monitor should probably become a
  geometry metric or a small-σ validation loss.

---

## D6. Diagnostics for GPFF's distinctive design choices

### D6.1 Permutation alignment

`align_permutation=true` Hungarian-matches `x_0` to `x_t` before forming the label, which means
**the regression target is not `E[x_0 | x_t]`** — it is a matched variant. `pseudo_pes.py` already
acknowledges a related subtlety by dropping `keys.noise` when alignment is on, because the drawn
noise is no longer the target; the force label is still exact for the *matched* `x_0`, but the
matched `x_0` is a different random variable. Measure the consequences:

1. **Permutation rate versus σ** — fraction of atoms moved by the matching. Should be ~0 at small
   σ and large at big σ. If it is non-negligible at small σ, alignment is corrupting exactly the
   regime that sets the geometry.
2. **Target bias.** For a handful of `x_t`, Monte-Carlo estimate both the unaligned conditional
   mean (average `x_0` over many draws consistent with `x_t`) and the aligned one, and compare the
   trained `F̂` against each. This says which target the model actually learned, and whether
   alignment should be σ-dependent (on at large σ, off at small σ) rather than global.
3. **Throughput.** `linear_sum_assignment` is `O(N³)` on an `N × N` cost matrix built per sample in
   the dataloader. At `N ≈ 180` that is ~100× the QM9 cost per sample, inside the data pipeline.
   Measure samples/s with and without it before committing to it on GEOM-Drugs.

### D6.2 Parity: the model is achiral, and it is worth proving

**Check:** for a reflection `P` (e.g. `diag(1, 1, −1)`), evaluate `F̂(Px)` and `P F̂(x)`. They will
agree to numerical precision.

**Why:** PaiNN's scalar channel is built from distances and from parity-even combinations of vector
features — the interaction's `dmuR · dir_ij + dmumu · μ_j` and the mixing block's `‖μ‖` and
`⟨μ, Wμ⟩` are all invariant under reflection. PaiNN is therefore **O(3)-equivariant, not just
SO(3)-equivariant**. A field with that symmetry assigns identical energies to enantiomers, so the
model **cannot produce an enantiomeric excess and cannot learn chirality-dependent conformational
preference.**

QM9 hides this (few stereocentres). GEOM-Drugs does not. The matching output-side metric is in the
benchmark (§3.3 B9: R/S and E/Z ratios, CIP assignability). The fix is parity-odd features: cross
products of vector channels (Megalodon's "EGNN + cross product" baseline is exactly this), higher
irreps via `so3net`, or explicit pseudoscalar inputs such as signed volumes of neighbour triples.

This is a case where a three-line diagnostic establishes a hard capability ceiling — worth running
before any training on GEOM-Drugs.

### D6.3 Prior / training mismatch at the top of the schedule

`GaussianPrior` draws `randn_like(R) * sigma` **without centring**, while training noise is
mean-free (`mean_free_noise: true`), so a prior draw has a centre-of-mass offset with std
`σ/√N ≈ 7 Å` at `N = 18`. Because PaiNN is translation-invariant and the forces are
translation-equivariant, this is *equivalent to a global translation* and should be harmless — but
it is worth confirming empirically rather than assuming, since it is one line to check
(`‖F̂(x + c) − F̂(x)‖` for constant `c`) and a silent failure here would look like a sampler bug.
Combine with D3.4: the σ mismatch at the top of the schedule is the real issue, not the centring.

### D6.4 Convergence and freezing behaviour

The samplers freeze molecules once `‖F̂‖` drops below `f_max` and `Calculator` then skips them.
Report:

* the NFE distribution (not the mean) and the per-molecule freeze step;
* the distribution of `‖F̂‖` at the moment of freezing, and at the end;
* **whether frozen-early molecules are systematically worse** on Tier-B metrics. They will be if
  `σ̂` is biased low (D3.1), which is the mechanism to look for: a short NFE tail *and* elevated
  `ΔE_relax` in the same molecules is the signature.

### D6.5 Shape-prior transfer

The `ShapeGMM` is fitted per molecule size on the train split. GEOM-Drugs' radius-of-gyration
distribution is much wider and more multi-modal than QM9's. Report the fit quality (held-out
log-likelihood per size band) and whether `prior=shape` still improves anything, before carrying
the option over.

---

## Priority order

Ranked by (expected explanatory power for the QM9 → GEOM-Drugs gap) ÷ (cost). The first four need
no training run.

| # | Diagnostic | Cost | Implicates |
| --- | --- | --- | --- |
| 1 | **D3.3** effective loss weighting | pen and paper + one plot | `sigma_weights.yaml` `max_clipping: 1` — the loss is ~blind to small σ, where stiff geometry is set |
| 2 | **D4.4** radial-basis resolution | no training | `n_rbf` / `cutoff` / `GaussianRBF` → `ExpGaussianRBF`; 0.25 Å resolution vs 0.01 Å requirement |
| 3 | **D6.2** parity check | three lines | PaiNN is achiral ⇒ a hard ceiling on stereochemistry, which GEOM-Drugs exposes and QM9 hides |
| 4 | **D3.4** σ support vs prior | one histogram | `sigma_mean` / `sigma_std` / `sigma_max` — sampling may start off-support |
| 5 | **D2.1 / D2.2** force residual on data, contraction spectrum | one forward pass / Lanczos | whether `n_steps=50` is converged at all; which coordinates are slow |
| 6 | **D1.1 / D1.2** symmetry defect, circulation | JVP/VJP probes | direct force head vs energy head + autograd; predicts Heun ↔ direct-denoising disagreement |
| 7 | **D5.3** σ-regime gradient conflict | one batch | `gpff` vs `gpff_t_aware`, or EDM preconditioning |
| 8 | **D4.3(d)** feature growth with size | logging | `prenorm` / `degree_norm` / `bounded_forces` on 180-atom molecules |
| 9 | **D6.1** permutation alignment | MC study + timing | `align_permutation` — bias at small σ, `O(N³)` cost at `N ≈ 180` |
| 10 | **D3.1** σ̂ calibration | one sweep | the adaptive samplers and the `f_max` freeze criterion |

Items 1–4 are the ones to run before committing GPU time to a GEOM-Drugs training run: each is
hours of work, needs no trained model (or only the existing QM9 one), and each could change the
configuration you would train with.

---

## Implementation sketch

```
src/gpff/diagnostics/
  field.py        D1  symmetry defect, loop circulation, PSD defect (JVP/VJP, Lanczos)
  fixedpoint.py   D2  force residual on data, contraction spectrum (rigid-mode projection),
                      consistency-along-ray curves
  sigma.py        D3  sigma-hat calibration, frozen-feature sigma probe, effective-weighting
                      plot, training-support vs prior histograms
  landscape.py    D4  path-integrated PES, 2D slices, Lipschitz (power iteration + multiscale),
                      feature-growth hooks, radial-basis audit
  optim.py        D5  filter-normalised loss surfaces, Lanczos/Hutchinson Hessian,
                      per-sigma gradient cosine matrix
  symmetry.py     D6  parity and translation checks, permutation-alignment statistics
cli/diagnose.py   `gpff_diagnose` — a model path, a split, a target list; writes
                  diagnostics/<name>.npz + .png, mirroring the prepare_metadata/
                  calculate_metrics split so the expensive probes are cached
```

Reuse what is there: `Calculator` already handles model/device/dtype placement and the fully
connected neighbour list, so the Jacobian probes should take a `Calculator` rather than a raw
model. `SamplerCallback` is the hook for the cheap during-training diagnostics (D2.1, D3.1,
D4.3(a), D5.4).

One caveat on the Jacobian probes: `Calculator._predict` calls `.detach()`, and `denoise` runs
under `torch.no_grad()`. The diagnostics need a gradient-enabled path — add a
`Calculator.calculate(..., create_graph=True)` route rather than working around it at the call
site.
