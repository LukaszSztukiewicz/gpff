# The UniGEM route for GPFF: per-molecule σ, atom count, and the Hungarian step

Design analysis for `ccr-916e702f-ut0j4o`. Companion to
`joint_type_diffusion_survey.md`, which covers the literature; this document is
about what to build.

---

## 0. Two facts about the current code that reframe the question

### (a) Positions-only GPFF is *composition-conditioned* generation

`gpff_generate` iterates the **test dataloader** and calls
`prior.sample_from_batch(batch)`, which overwrites `properties.R` only
(`sample/priors.py`). `n_atoms`, `idx_m` and `properties.Z` all survive from the
real test molecule — and `cli/generate.py` refuses `n_samples > len(test_dataset)`,
i.e. **one generated molecule per test structure, reusing its exact molecular
formula.**

So the baseline is sampling `p(x | formula)` with the true H count, the true
heavy-atom composition, and the true N. `experiment=gpff_types` replaces that
with an argmax over a noisy `h` — it samples `p(x, composition | N)`.

**The branch's regression therefore conflates two separate changes:** losing
composition conditioning, and the continuous-one-hot mechanism. Validity and
stability are extremely sensitive to the first (a correct H count essentially
hands you valence saturation). Before attributing anything to the mechanism,
this needs separating — see Stage 0 in §5.

### (b) Per-molecule σ already exists, in both halves of the pipeline

- **Training:** `PseudoPESTransform` runs per sample in the dataset, and
  `VarianceExplodingGeometries` draws `sigma_t` with `size=(1,)`
  (`pseudo_pes/geometries.py:88`) before broadcasting to `(N,1)`. σ is already
  one value per molecule.
- **Inference:** `Calculator.calculate(estimate_sigma=True)` computes
  `_compute_std_per_molecule(idx_m, forces/2)` — a **per-molecule σ read off the
  model's own predicted force magnitude** — and `AdaptiveHeunSampler` /
  `AdaptiveAncestralSampler` use it to set a per-molecule `step_size`, with
  `update_convergence` freezing molecules independently. `n_steps: 50` against
  `n_steps_target: 10` shows how much this buys.

This is the payoff of GPFF being time-unaware: there is no global clock, so each
molecule can sit at its own point on the trajectory. **It is also the piece that
makes the UniGEM route natural rather than awkward** (§2).

Gap worth noting: the type code landed only in `HeunSampler` and
`StochasticHeunSampler`. The *adaptive* samplers have no `keys.h` handling at
all, so `gpff_types` models currently cannot be sampled with the per-molecule-σ
samplers.

---

## 1. Is the UniGEM route possible here? Yes — and it fits better than in UniGEM

UniGEM diffuses coordinates only and attaches an atom-type **prediction** head
active in the last ~1–3% of the trajectory (`t_n = 10` of `T = 1000`). Porting it:

| UniGEM piece | GPFF equivalent | Status |
| --- | --- | --- |
| Coordinate diffusion | the pseudo-PES force field | unchanged |
| Type prediction head | `TypeProbabilities` + CE loss | **already on this branch** |
| Nucleation time `t_n` | a nucleation **σ**, per molecule | needs a gate |
| Phase boundary at inference | `estimate_sigma` per-molecule σ | **already exists** |
| Type labels per atom | the Hungarian match | needs §4 |

Three things make this *cleaner* in GPFF than in UniGEM:

1. **No new modality in the energy.** The branch currently extends the
   pseudo-PES to `E = |x_t − x_0|² + |h_t − h_0|²`, which forces a type channel
   through the whole force-field formalism. In the UniGEM route the energy is
   untouched and the type head is a pure auxiliary classifier. Strictly less
   machinery than what is already written.
2. **The phase boundary becomes measurable instead of a hyperparameter.** UniGEM
   has to guess `t_n`. GPFF estimates σ per molecule from its own forces, so the
   gate can be `σ̂ < σ_n` — and σ_n itself is derivable (§3).
3. **The Hungarian step already defines the labels.** UniGEM's atom types come
   from a fixed atom ordering. GPFF has to match noisy atoms to reference atoms
   anyway, and that match *is* the type label (§4).

### But the naive port is inferior, and here is why

If types are not an input, the model is **type-blind**: the force head loses
element conditioning that the baseline gets for free from the test molecule's
`Z`. UniGEM still beat EDM this way, because EDM's model also never sees a clean
`Z`. **GPFF's baseline does.** So naive-UniGEM-GPFF trades one handicap for
another and will likely still lose on geometry.

The fix is not to keep types out of the input — it is to keep them out of the
*diffusion* while letting them back into the *conditioning* as they become
known. That is the design in §2.

---

## 2. The recommended design: decoupled types, discrete commitment, per-molecule gate

> Element identity leaves the diffusion entirely. It becomes an **absorbing
> discrete channel**: every atom starts as `MASK`, and atoms are **committed**
> to a real element in confidence order, gated by the per-molecule σ the force
> field already estimates. Once committed, an atom feeds the force field as a
> genuine element.

Concretely:

- **Embedding.** Keep `nn.Embedding`; add one row for `MASK` (and later one for
  `ABSENT`, §4). Delete the `GaussianRBF` expansion of `h_t` and the
  `embedding_key: h` plumbing. The network's type input is always a legal
  element or `MASK` — never an unbounded real vector. No overflow at
  `sigma_max`, no `type_scale`, no `type_range` clipping.
- **Type head.** The existing `TypeProbabilities` + CE loss, trained only where
  the label is meaningful: `σ < σ_n` (§3). Above that the head is not trained
  at all (UniGEM's hard gate), so it never learns to be confident where the
  Hungarian label is noise.
- **Labels.** From the positions-only Hungarian match (§4).
- **Commitment at inference.** While `σ̂ > σ_n`: all `MASK`, pure type-blind
  position denoising — exactly today's force field with one extra embedding row.
  Once `σ̂ < σ_n`: commit the k most confident atoms per step (lowest predicted
  entropy), optionally subject to a composition constraint (§4). Allow
  **re-masking** of an atom whose committed element becomes inconsistent — early
  wrong commitments are otherwise irreversible, which is the known failure mode
  of confidence-ordered unmasking.

Why this beats both alternatives:

- vs. the **current branch**: no continuous relaxation, so no scale/schedule/
  weighting knobs to tune, no `h_t` magnitude problem, and no indecisive
  between-category states (the oscillation UniGEM measured). The commitment is a
  discrete sample, which is where the whole field landed — EQGAT-diff +2.2 pts
  mol. stability on QM9, KLDM +20.6 pts stable on MP-20, and in KLDM's table the
  continuous-one-hot variant had the worst **coordinate** RMSD too.
- vs. **naive UniGEM**: element conditioning returns progressively, so the force
  field is element-aware over exactly the low-σ stretch where bond lengths and
  valences are decided. That is the part GPFF is good at and the part that
  determines the metrics.

This is also, read from the other direction, absorbing-state discrete diffusion
with the unmasking schedule driven by the model's own confidence — i.e. the
family B design from the survey, reached by way of UniGEM's decoupling. The two
routes converge, which is a good sign.

---

## 3. Per-molecule σ: superior, and load-bearing here

**At training:** already per-molecule, and should stay. Per-batch σ would reduce
σ diversity within a gradient step for no benefit.

**At inference:** per-molecule adaptive σ is GPFF's distinctive capability and is
*required* for a principled type gate. A global step index means nothing when
molecules sit at different σ at the same step.

### σ_n is derivable, and it is genuinely per-molecule

The Hungarian assignment — and therefore the element label — becomes ambiguous
when σ is comparable to **half the molecule's nearest-neighbour distance**. For
QM9 the shortest contact is C–H ≈ 1.09 Å, so the scale is

```
σ_n ≈ d_min / 2…3  ⟹  σ_n ≈ 0.3–0.5 Å for QM9
```

and it is **smaller for denser, more H-rich molecules**. A fixed global σ_n
would be wrong for both tails of the size distribution. This is the strongest
argument for per-molecule σ in this design: the gate should scale with the
molecule's own `d_min` (or its radius of gyration, which the shape machinery
already computes per molecule).

Better still, σ_n is *measurable* rather than guessed: sweep σ over the training
set, run the positions-only Hungarian match, and record the σ at which the
assigned element matches the true element ~95% of the time. That is a short
script over existing code and it fixes a hyperparameter the UniGEM paper had to
ablate.

### The one real caveat

`_compute_std_per_molecule` estimates σ from `3N` values, so its relative error
is ≈ `1/sqrt(2(3N−1))` — about **10% for a typical 18-atom QM9 molecule**, worse
for small ones. A hard commit decision on a 10%-noisy σ will misfire sometimes.
Mitigation: use σ̂ only as a *gate* (open/closed) and let the **type head's own
entropy** choose which atoms to commit and when. Confidence is the better signal;
σ̂ just prevents premature commitment.

### Per-*atom* σ: not now

Once elements are committed at different times, a per-atom notion of "how
resolved is this atom" exists, and asynchronous per-atom denoising is the logical
extension (the dormant `per_atom_noise` / `MixedGeometries.t_sigma` paths point
this way). But a single atom's σ can only be estimated from 3 force components
(±40%), so it would need neighbour smoothing, and it multiplies the state space
of things that can go wrong. **Lowest payoff-to-risk of everything here. Skip it
until the rest works.**

---

## 4. The Hungarian step: three distinct jobs

This is where the design is most underappreciated. `align_permutation` currently
does one thing; in the target design it does three.

### Job 1 — defining the regression target (existing)

GPFF's pseudo-PES target `2(x_0 − x_t)` is ill-defined under atom exchange, so
the Hungarian match picks the assignment. This is minibatch/equivariant optimal
transport in all but name, and the usual justification applies: an OT coupling
straightens the paths and lowers the variance of the regression target.

### Job 2 — supplying the type label (new, and it is what makes the route work)

In a type-blind model atoms are **fully exchangeable**, so the match must be
unconstrained but the cost must be **positions-only**:

```
cost[i,j] = |x_0,i − x_t,j|²        # no type term
```

and the element label for noisy atom `j` is inherited from the reference atom it
matched. Note this is strictly better than the branch's current cost: the branch
minimises `|Δx|² + |Δh|²` over the concatenated 8-vector
(`data/transform/pseudo_pes.py:84-91`), and because `h`'s noise is the *same* σ
as positions and independent of `type_scale`, the type block contributes
`~2·n_types·σ² = 10σ²` against the positions' `~6σ²`. **At large σ the assignment
is dominated by matching type noise — roughly 60/40 — so the permutation defining
the pseudo-PES target is chosen substantially at random with respect to the
geometry.** Dropping the type term fixes a target-quality bug and supplies the
labels in one move.

The consistency this buys is the nicest property of the whole design: **the σ at
which the assignment stops being ambiguous is the σ at which types become
predictable.** The label and the gate come from the same object, so they cannot
disagree.

### Job 3 — composition-constrained decoding at inference (new)

If you want to *keep* the formula conditioning the baseline enjoys, the
Hungarian algorithm does that too. Given per-atom element probabilities `p_ij`
and target counts `n_j` from the formula, solve a **balanced assignment** with
cost `−log p_ij` and column capacities `n_j`. The generated molecule then has
exactly the target formula while the model still decides *which* atom is which
element. This is strictly better than argmax, which cannot respect a composition
constraint at all, and it gives the clean ablation in Stage 0 below.

### Job 4 — variable atom count (later)

With over-provisioning (§5) the match becomes **rectangular**: `N_max` noisy
atoms against `N_true` real reference atoms plus `N_max − N_true` `ABSENT` slots.
`scipy.optimize.linear_sum_assignment` handles rectangular costs natively, so
this is a cost-matrix change, not an algorithm change.

---

## 5. Atom count: do it last, and never by "diffusing N"

**N is not a diffusable variable in a fixed-dimension Cartesian state space** —
it *is* the dimension of the state being diffused. The only options are:

1. **Sample N ancestrally** from the empirical `p(N)` (EDM, and what GPFF would
   do to stop borrowing N from the test set). Trivial, and it is the honest
   baseline for unconditional generation.
2. **Over-provision + an `ABSENT` class** — FlowMol3's "fake atoms", which it
   reports as the most impactful of its three tricks for chemistry/topology
   metrics. Draw `N_max`, let the model mark the surplus absent, drop them at the
   end. This rides entirely on the type channel, so **it only works once the type
   channel works.**
3. **Diffuse in a fixed-dimension latent space** (ADiT, and the fixed-dimensional
   latent-flow line of work). A different project.

Option 2 is the right target, with two GPFF-specific details:

- **Mask absent atoms out of the pseudo-PES.** A fake atom has no reference
  position, so it must be excluded from the energy and the force loss — not given
  a zero-force target, which would teach the field to pin it in place.
- **Exclude believed-absent atoms from the σ estimator.** `_compute_std_per_molecule`
  takes the std over all atoms; surplus atoms wandering in the prior would bias
  σ̂ and therefore corrupt the adaptive step size *and* the commitment gate. This
  coupling is easy to miss and would be diagnosed as "the adaptive sampler broke".

---

## 6. Which combination — and in what order

**The combination I would build:** decoupled discrete element channel with
progressive commitment (§2) + per-molecule σ as the commitment gate (§3) +
positions-only Hungarian for labels and balanced assignment for decoding (§4),
with atom count deferred to fake atoms (§5) once the rest is green.

Not "UniGEM" exactly: take its **decoupling** and reject its **type-blindness**.

The staging matters more than the design, because the branch currently changes
four things at once:

**Stage 0 — measure the real gap (do this first, it may change everything).**
Keep the formula fixed exactly as the baseline does, and generate only the
*assignment* of elements to positions: type head + balanced-assignment decoding,
no type diffusion. This isolates "must the force field infer element identity?"
from "did we lose composition conditioning?". If this alone loses badly to
baseline, the problem is element conditioning and no mechanism change rescues it;
if it matches baseline, the mechanism is the whole story and the rest of the plan
is worth building.

**Stage 1 — the discrete channel.** `MASK` row in the embedding, CE head gated at
`σ < σ_n` with σ_n measured from Hungarian stability, confidence-ordered
commitment with re-masking, positions-only Hungarian labels. Port the type
handling into the adaptive samplers so per-molecule σ is available. Still
composition-conditioned, so it stays comparable to baseline.

**Stage 2 — free the composition.** Drop the balanced-assignment constraint; `N`
still from `p(N)` or the test set. This is the first number that is honestly
comparable to EDM/MiDi/EQGAT-diff rather than to GPFF's own conditional baseline.

**Stage 3 — free N.** `ABSENT` class, rectangular Hungarian, masked pseudo-PES,
σ estimator restricted to present atoms.

**Cheap and orthogonal, worth adding at Stage 1:** self-conditioning on the type
belief (Analog Bits; FlowMol3 ranks it top for validity) — concatenate the
previous step's type distribution, zeroed 50% of the time in training, no
gradient through the first pass, <25% overhead. It is the mechanism that makes
progressive commitment stable.

### Honest risks

- **Stage 0 may show the gap is mostly composition conditioning.** Then the
  headline comparison was never mechanism-limited, and the right move is to
  report conditional and unconditional numbers separately rather than to keep
  engineering the type channel.
- **Irreversible commitment.** Confidence-ordered unmasking is greedy; without
  re-masking, one early wrong element propagates into the geometry through the
  force field. Keep re-masking available from the start.
- **σ̂ noise at small N.** 10% at N=18 and worse below; gate on σ̂, order on
  entropy.
- **Three Hungarian jobs sharing one function.** `align_permutation` is about to
  be asked to do training-time matching, label assignment, rectangular slots and
  constrained decoding. Split it rather than add flags.
