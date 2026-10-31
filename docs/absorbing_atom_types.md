# Absorbing-state discrete diffusion over atom types

Design note for extending GPFF from "geometry given the composition" to "geometry **and**
composition", with a single token vocabulary

```
V = {H, C, N, O, F, P, S, Cl, Br, I, ...}  ∪  {ABSENT}  ∪  {MASK}
```

and absorbing-state (masked) discrete diffusion over it. Written as a step towards
pocket-conditioned generation. **No implementation yet** — this is the survey plus the list
of decisions to settle first.

---

## 1. Where GPFF stands today, concretely

The composition is never generated. It is copied from a real molecule:

- `cli/generate.py:155` loops over the **test dataloader**, and `prior.sample_from_batch`
  (`sample/priors.py:53`) overwrites only `properties.R`. So `properties.Z`, `n_atoms` and
  `idx_m` all come from a held-out QM9 molecule. Every reported sample has the exact element
  multiset and atom count of a real test structure.
- The model is a pure force field: one equivariant head (`GatedEquivariant`,
  `model/heads.py:16`) emitting a 3-vector per atom, trained on the analytic pseudo-force
  `2(x_0 - x_t)` (`pseudo_pes/labels.py:36`) with weight `1/σ²` (`pseudo_pes/weights.py:56`).
- It is time-free: no `t` enters the network; at sampling, `σ` is *estimated* from the spread
  of the predicted forces (`Calculator._compute_std_per_molecule`, `sample/base.py:103`) and
  `x̂_0 = x_t + F/2` (`sample/base.py:189`).
- Atom types enter only through PaiNN's `nn.Embedding(100, n_atom_basis)` indexed by `Z`
  (`model/painn.py:71,141`), and through the type-constrained Hungarian matching in
  `align_permutation` (`utils/alignment.py:486`).

So the extension needs: (a) a second, **invariant categorical** output; (b) a corruption
process for it; (c) a way for the atom **count** to be generated rather than copied; (d) a
sampler that interleaves the two. (c) is what `ABSENT` buys, and it is the part that matters
most for pocket conditioning — ligand size must become a model output.

Two useful pieces already exist and should be reused rather than reinvented:

- **Frozen context machinery.** `ScaffoldDirectDenoisingSampler` + `ScaffoldPrior` +
  `keys.scaffold_mask` already implement "these atoms keep their positions, those denoise
  around them". A protein pocket is exactly a frozen context. Consider renaming the concept
  to `context_mask` when touching it.
- **Reproducible per-sample noise.** `_PinnedNoise` (`callbacks.py:111`) seeds the CPU torch
  and python RNGs from `(seed, epoch, db index)` inside the transform. A mask draw made
  inside `PseudoPESTransform` inherits that for free.

---

## 2. The process, stated precisely

Joint state per molecule: positions `x_t ∈ R^{N×3}` and types `a_t ∈ V^N`, on a canvas of
`N` slots.

**Forward, types.** With `t ∈ [0,1]`, `t=0` clean, each token is independently replaced by
`MASK` with probability `1 - α_t`:

```
q(a_t^i | a_0^i) = Cat( α_t · onehot(a_0^i) + (1 - α_t) · onehot(MASK) )
```

`α_t` decreasing, `α_0 = 1`, `α_1 = 0`. Linear (`α_t = 1 - t`) is the usual default.

**Reverse posterior** (MDLM's SUBS, [Sahoo et al. 2024](https://arxiv.org/html/2406.07524v2)):
for `s < t`, if `a_t^i ≠ MASK` it is **carried over unchanged**; otherwise

```
q(a_s^i | a_t^i = MASK, a_0^i) = Cat( (α_s - α_t)/(1 - α_t) · onehot(a_0^i)
                                    + (1 - α_s)/(1 - α_t) · onehot(MASK) )
```

The network predicts `p_θ(a_0^i | x_t, a_t)` with **zero probability on MASK**, substituted
into the above. Two consequences worth internalising: unmasking is irreversible, and the
model never has to predict `MASK`.

**Loss.** The continuous-time NELBO collapses to a weighted masked cross-entropy
([MDLM](https://arxiv.org/html/2406.07524v2),
[MD4 / Shi et al.](https://arxiv.org/html/2406.04329v4)):

```
L_type = E_t E_q [ (−α'_t / (1 − α_t)) · Σ_{i : a_t^i = MASK} −log p_θ(a_0^i | x_t, a_t) ]
```

For `α_t = 1 − t` the weight is simply `1/t`. Only masked positions contribute.

**Why absorbing fits GPFF unusually well.**
[Ou et al., *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions
of Clean Data*](https://arxiv.org/html/2406.03736v2) (RADD, ICLR 2025) show that for an
absorbing process the concrete score factorises into a *time-independent* clean-data
conditional times a scalar function of `t`. The network therefore **needs no time input at
all** — and caching its output while `a_t` is unchanged cuts function evaluations (they
report up to 3.5× faster sampling). GPFF's time-freeness for coordinates is a heuristic
(σ is estimated from force magnitudes); for the absorbing type head it is *exact*. The same
paper, and [ARDM](https://arxiv.org/abs/2110.02037), establish the equivalence with
any-order autoregressive models: the objective is an expected NLL over random atom orders,
so the type head is literally learning *"given this partial molecule and its geometry, which
element sits at this site?"* — a clean story for a thesis chapter.

This is the main argument for absorbing over uniform here. Uniform-state transitions make
`p(a_0 | a_t)` genuinely `t`-dependent, which would force a time input back into a model
whose entire design point is not having one.

---

## 3. Survey

### 3.1 Discrete diffusion machinery (the part to copy)

| Work | What to take |
|---|---|
| [D3PM](https://arxiv.org/abs/2107.03006) (Austin et al. 2021) | Original absorbing/uniform/Gaussian transition matrices; the `[MASK]` absorbing state. |
| [ARDM](https://arxiv.org/abs/2110.02037) (Hoogeboom et al. 2022) | Absorbing diffusion ≡ any-order AR; parallel decoding of `k` tokens per step with the correct loss reweighting. |
| [MDLM](https://arxiv.org/html/2406.07524v2) (Sahoo et al. 2024) | **SUBS** parameterization: zero mass on `MASK`, carry-over unmasking. Rao–Blackwellised continuous-time ELBO = weighted masked CE. The cleanest reference implementation. |
| [MD4](https://arxiv.org/html/2406.04329v4) (Shi et al. 2024) | Same simplified ELBO, derived generally; code at `google-deepmind/md4`. |
| [SEDD](https://arxiv.org/abs/2310.16834) (Lou et al. 2024) | Score-entropy objective; its "scaling trick" is explained by RADD as a reparameterization. |
| [RADD](https://arxiv.org/html/2406.03736v2) (Ou et al. 2025) | **Time-free** network, output caching, AO-ARM equivalence. Code: `ML-GSAI/RADD`. |
| [UDLM](https://arxiv.org/pdf/2412.10193) (Schiff et al. 2025) | For **small vocabularies the masked-vs-uniform gap is negligible**, and uniform is sometimes ahead (incl. on QM9); uniform is clearly better for classifier-free / classifier-based *guidance*. Relevant because `|V| ≈ 12` here and because pocket conditioning may later want guidance. |
| ["Uniform Diffusion Models Revisited"](https://icml.cc/virtual/2026/74360) (ICML 2026) | The masked/uniform gap is driven by *parameterization and sampling design*, not the marginals. Read before concluding anything from a masked-vs-uniform ablation. |
| [Discrete Flow Matching / MultiFlow](https://proceedings.mlr.press/v235/campbell24a.html) (Campbell et al. 2024) | CTMC formulation; **joint continuous + discrete generation** (protein backbone + sequence) — the closest structural template for "coordinates + types in one model", including the ability to condition on one modality and generate the other. |
| [ReMDM](https://arxiv.org/html/2503.00307v4) (NeurIPS 2025) | **Remasking** sampler applied to a pretrained masked model; fixes the incoherence from independent parallel unmasking; a predictor-corrector, strictly more general than earlier correctors. Reported to help on molecule design. |
| [MaskGIT](https://arxiv.org/pdf/2202.04200) (Chang et al. 2022) | Confidence-based parallel decoding; cosine mask schedule `γ(r) = ½(1 + cos πr)`; 8–12 iterations is the usual sweet spot. [A 2025 ICLR paper](https://iclr.cc/virtual/2025/poster/29658) notes confidence ordering picks *spatially clustered* tokens and proposes a Halton-sequence scheduler instead — a real concern in 3D, where clustering means "decode one end of the molecule first". |
| [Edit Flows](https://papers.neurips.cc/paper_files/paper/2025/hash/cb43f46154e750746602faaffd65fbbb-Abstract-Conference.html) (NeurIPS 2025) | Insert/delete/substitute edits over variable-length sequences, with **no mask or pad tokens at all**. The principled alternative to a padded canvas; much heavier to implement (auxiliary edit-path variables, Monte-Carlo training). Worth knowing exists; not for v1. |

### 3.2 Atom types in 3D molecular generative models

- **[EDM](https://arxiv.org/abs/2203.17003)** (Hoogeboom et al. 2022) — one-hot types treated
  as *continuous* extra channels with Gaussian noise. `N` is drawn from the empirical size
  distribution. This is the near-zero-effort baseline for GPFF (see D-A1).
- **[DiGress](https://arxiv.org/abs/2209.14734)** (Vignac et al. 2023) — discrete diffusion
  over atom types *and bonds*, with **marginal** (not uniform, not absorbing) transitions,
  i.e. the stationary distribution is the dataset's element frequency. Plus auxiliary
  structural features. Graph-only.
- **[MiDi](https://arxiv.org/abs/2302.09048)** (Vignac et al., ECML PKDD 2023) — joint 2D+3D:
  continuous coordinates, discrete types/bonds, and an **adaptive noise schedule** with
  *different noise rates per modality*. Directly relevant to D-C3. Also the sharpest warning
  against bond-free generation at drug scale: on GEOM-Drugs MiDi reports **92% stable
  molecules vs 6% for EDM with distance-based bond inference** (40% with a bond-order
  optimiser bolted on). See D-A3.
- **[EQGAT-diff](https://arxiv.org/html/2309.17296v1)** (Le et al., ICLR 2024) — systematic
  design-space study: continuous coordinates + **categorical** elements, charges and bonds.
  Credits **time-dependent loss weighting** for much of the gain, and reports that extra
  chemically motivated features (hybridisation) improve validity. Also demonstrates
  pretrain-on-PubChem3D → fine-tune transfer.
- **[FlowMol](https://arxiv.org/pdf/2404.19739)** / **[FlowMol-CTMC](https://arxiv.org/html/2411.16644v1)** /
  **[FlowMol3](https://arxiv.org/pdf/2508.12629)** (Dunn & Koes) — the most honest comparison
  of type parameterizations. FlowMol found a plain Gaussian prior on one-hots "equivalent or
  superior" to simplex-based (SimplexFlow / Dirichlet) treatments; FlowMol-CTMC then moved to
  **masked** discrete flow matching (types jump between `MASK` and real elements) and reports
  SOTA with fewer parameters. The lesson: do not assume discrete > continuous; measure it.
- **[SemlaFlow](https://arxiv.org/abs/2406.07266)** (Irwin et al.) — fast flow matching with
  categorical types; good engineering reference for few-step sampling.

### 3.3 Variable atom count — the `ABSENT` question

Nobody in the 3D-molecule literature gets size for free; the approaches are:

1. **Sample `N` from a prior.** EDM/DiffSBDD/TargetDiff draw `N` from the training
   distribution, conditioned on pocket size for SBDD. Simple, but size is not a learned
   function of the context.
2. **Learn `N` from the context.** [Prior-Guided Flow Matching](https://arxiv.org/pdf/2509.01486)
   predicts the ligand atom count from pocket volume and surface area, improving on
   size-from-pocket-diameter heuristics.
3. **Existence masks on an over-provisioned canvas.** [READ](https://arxiv.org/pdf/2506.14488)
   bins a pocket statistic, samples a heavy-atom count, and uses existence masks over atom
   indices as a "coarse size controller" at initialisation.
4. **Virtual / dummy nodes that can be deleted — this is the `ABSENT` design.**
   **[DrugFlow](https://arxiv.org/html/2508.17815v1)** (Cremer et al.) is the closest prior
   art: *"we further introduce a virtual node type to allow the model to dynamically add or
   remove atoms and thus learn about the distribution of ligand sizes rather than requiring
   it to be pre-specified"*, with virtual nodes removed during sampling and the distribution
   of removals shown to capture the ligand/pocket size dependency. Also combines continuous
   flow matching with discrete Markov bridges and adds an OOD uncertainty estimate.
   [SILVR](https://arxiv.org/pdf/2304.10905) uses unguided dummy atoms for linker design and
   found they help.
5. **True insertion/deletion** — Edit Flows, FlexMDM. Principled, heavy.

**Read this as validation plus a warning**: `ABSENT`-as-a-token is an established, published
mechanism (4), so the project is not speculative — but the published versions pair it with a
size *prior* that gets the canvas into the right ballpark. Expecting `ABSENT` alone to carry
the whole size distribution from a fixed global canvas is the risky version.

### 3.4 Engineering tricks worth lifting

- **SUBS**: never let the head put mass on `MASK`; never overwrite an unmasked token. Both
  are two-line changes that remove whole classes of bug and tighten the ELBO.
- **Weighted masked CE**: loss only on masked positions, weight `1/t`. Note the `1/t`
  blow-up as `t → 0`; clamp `t ≥ t_min` (this mirrors what the existing `floored-sigma`
  worktree does for `1/σ²`).
- **Carry-over caching (RADD)**: if `a_t` did not change between steps and the model is
  time-free, the type logits are unchanged — skip the forward pass. GPFF's sampler runs 50
  geometry steps; if types only change at ~10 of them, this is a direct NFE saving.
- **Cosine count schedule** for how many tokens to unmask per step (MaskGIT), not linear.
- **Confidence ordering**, but beware spatial clustering; Halton/stratified ordering is the
  documented fix.
- **Remasking corrector** (ReMDM) retrofits onto a trained model with no retraining — a cheap
  late-stage experiment, not an architectural commitment.
- **Token masking for disallowed elements**: `-inf` on logits for elements absent from the
  active dataset, at both train and sample time. This is the direct analogue of the
  **documented fluorine pitfall** (`CLAUDE.md`: old-DB models never trained the F embedding
  and emitted 1e26 forces on CF3): an untrained *logit* is worse than an untrained
  embedding, because it can be sampled.

---

## 3.5 What the `feat/atom-types-port` runs already establish

`feat/atom-types-port` (5 commits, snapshot `59a4012-atom-types-port`) already implements
EDM-style continuous one-hot types, and 20 runs on hydra have evaluated it. It is a more
developed design than the "cheap control" of D-A1b, and it settles several decisions below
empirically. All numbers are QM9, 10 000 samples, seed 42069, `sdd256` unless stated.

**What it does.** Scaled one-hot `h_0 = s·onehot`, noised with the **same σ as the positions**;
pseudo-energy `E = |x_t − x_0|² + |h_t − h_0|²`; PaiNN reads `h_t` instead of `Z` via a new
`embedding_key`. Three things in it I had not anticipated:

- **`h_t` goes through a GaussianRBF before the linear embedding**, like the interatomic
  distances — because `h_t` grows with σ and overflows PaiNN at `σ_max = 30`.
- **The RBF window is an implicit `MASK`.** `gpff_types_categorical` reads `h_t` only over
  `[−type_range, type_range]`; outside it the basis is exactly zero, so *fully noised types
  read as unknown*. That is a continuous surrogate for an absorbing state, reached from the
  other direction.
- **`align_permutation` was already fixed** for unknown types: it matches on the joint cost
  `|Δx|² + |Δh|²` with the element constraint disabled, which is a better answer to D-C7 than
  the "align on clean types first" I proposed.

**Established results.**

| finding | evidence |
|---|---|
| **`type_scale` is the dominant hyperparameter, with an interior optimum near 0.3.** | validity 88.2 / 91.4 / **95.3** / 94.5 / 80.9 at `s` = 0.05 / 0.15 / 0.3 / 0.5 / 1.0. A 14-point swing. |
| **Categorical CE ≈ continuous MSE.** The head is *not* where the gains are. | at identical `s = 1.0`: `caring-dove` (CE) 83.2 vs `great-kitten` (MSE) 80.9 @2000; 84.9 vs 87.8 @5000. The apparent "CE is much worse" is purely a `type_scale` confound — every `cat-range` run inherits `s = 1.0`. |
| **The RBF window width barely matters.** | `type_range` 1/3/5 at fixed spacing 0.25 → 80.1 / 81.0 / 81.0 @2000. Finer spacing helps slightly (81 rbf: 72.5–74.5 vs 70.5–72.4 @1000). So the implicit-`MASK` threshold is not the bottleneck. |
| **Weight the type loss *up*, and keep the `1/σ²` weighting on it.** | at `s = 0.5`: `loss_weight` 4 → 94.5, 0.5 → 92.4, 0.5 with `weights_key=null` → 91.6. |
| **Predicting the type *force* fails; predicting clean types works.** | `loving-leech` (`TypeForces`) 58.9 @1000 vs `great-kitten` (h₀) 74.2 @1000. Already superseded on the branch. |
| **Types cost little at high NFE and a lot at low NFE.** | baseline − best typed, @2000: `sdd256` **2.9**, `sdd16` 10.0, `dd256` 17.5, `dd16` 18.7. The continuous type channel needs hundreds of steps to anneal. |
| **Stochastic sampling matters far more once types are generated.** | `ts=0.3` @2000: `dd256` 71.0 → `sdd256` 95.3 (+24). Untyped: 88.5 → 98.2 (+10). The deterministic sampler gets stuck in bad type assignments. |

**The unfixed failure: systematic heteroatom suppression.** This is the headline, and it is
invisible in the validity column.

| | baseline (true types) | typed, across *every* setting |
|---|---|---|
| molecules containing N | **61.3%** | 43.8 – 47.7% |
| N at atom level | **5.68%** | 2.75 – 4.04% |
| molecules containing O | **84.7%** | 75.3 – 79.8% |
| uniqueness | **92.4%** | 80.8 – 91.1% |
| elongation s.d. | **0.794** | 0.68 – 0.72 |

Nitrogen is produced at ~70–75% of its true molecule-level rate and ~55–70% of its true
atom-level rate, and **no hyperparameter in the sweep moves it** — the deficit is flat across
`type_scale`, across `loss_weight`, across MSE-vs-CE, and across the RBF window. Carbon and
hydrogen absorb the difference (C 35.1% → 36.7–37.6%). Uniqueness and shape variance drop
with it. What *is* tunable is validity *given* a heteroatom: `valid | has N` rises 81.5 → 94.0
from `s = 0.05` to `s = 0.3`, against the baseline's 98.3. So the `type_scale` optimum is a
real improvement in handling heteroatoms, laid on top of a frequency deficit it never touches.

**The readout looked like the mechanism.** Types are read out by `Sampler.update_types` as
`Z = atom_types[h.argmax(-1)]`, where `h` is the model's clean-type estimate
`s · softmax(logits)`. The final element is therefore an **argmax of a mean-field marginal** —
fully greedy, never sampled. Argmax of an expectation collapses onto the majority class
wherever the model is uncertain, and C-vs-N is exactly the geometrically confusable pair
(C–C 1.54 Å vs C–N 1.47 Å), while fluorine survives at roughly the right rate (0.14% true,
0.07–0.19% generated) because CF₃ geometry is unmistakable. Rare classes suppressed, majority
class inflated, diversity down: the signature of greedy decoding.

**It is not.** §3.6 tested it directly and the readout accounts for none of the deficit.

---

## 3.6 Tested: the readout is not the cause

`ReadoutDirectDenoisingSampler` (branch `feat/typed-type-readout`, snapshot
`59a4012wt-typed-readout`) keeps the trajectory of `direct_denoising` and only changes how the
element is taken from the marginal: `argmax`, `sample` (draw from it), or `commit` (draw and
replace `h` by the scaled one-hot). The draws use a generator of their own, so the positions
see identical noise and the readouts are paired. On `caring-dove:5000`, 10 000 molecules each:

| readout | at% N | mol% N | at% O | elemTV |
|---|---|---|---|---|
| untyped baseline (true types) | **5.68** | **61.27** | **7.75** | 0.0000 |
| `sdd16`, as evaluated before | 2.74 | 34.77 | 6.50 | 0.0427 |
| `sdd16argmax` (control, new code) | 2.73 | 34.57 | 6.54 | 0.0424 |
| `sdd16sample` | 2.77 | 34.85 | 6.52 | 0.0422 |
| `sdd256`, as evaluated before | 3.07 | 38.62 | 6.61 | 0.0375 |
| `sdd256argmax` (control, new code) | 3.10 | 38.68 | 6.64 | 0.0368 |

The `argmax` controls reproduce the original configs to within 0.03 pp, which is the
run-to-run noise of the nondeterministic scatter-add in PaiNN, so the new code path is
faithful. And **sampling changes nothing**: 2.73 → 2.77 at% N is inside that same noise, with
180 000 atoms behind each number.

**What that implies.** `argmax` and `sample` can only differ where the marginal is
non-degenerate. They agree to three significant figures, so by the end of sampling the
per-atom marginals are **essentially deterministic**: the model is not hedging between C and
N and losing the coin flip, it is *confident* and confidently wrong about how often N occurs.
The deficit is therefore set before the readout — in training, or in the sampling dynamics —
and no decoding rule can repair it.

**The surviving hypothesis: self-reinforcement through the continuous channel.** The type loss
carries the positions' `1/σ²` weight, so it is dominated by low-σ samples, where `h_t` already
encodes the element. The model is trained largely to *copy* `h_t`. At sampling time `h` is
self-generated — `h ← s · softmax(model(h))` is a fixed-point iteration — so whichever element
the channel drifts to early is read back as evidence and reinforced, and the drift favours the
abundant elements. The regime the sampler actually starts in, "infer the element from the
geometry while the type channel carries no information", is the regime the training objective
weights least. That also explains why `sdd16` is worse than `sdd256` (2.74 vs 3.07 at% N):
fewer steps give the channel less chance to escape its initial drift.

This is testable in **one forward pass, no sampling loop**: take true test geometries, set `h`
to pure noise at `σ_max`, and compare the predicted marginal averaged over atoms with the
dataset's element frequencies. If it already under-weights N, the loss weighting and
parameterization are at fault; if it is calibrated, the fault is in the sampling dynamics.
Either answer points at a different part of the absorbing design, so run it before building
anything. The `commit` readout (draw *and* write the one-hot back into `h`) is the matching
sampling-side probe, and is queued.

**What this does to the case for absorbing diffusion.** It removes the cheap win and replaces
it with a sharper argument, about **train/sample consistency** rather than decoding:

- Masked diffusion's training task *is* the sampling-time task — predict a masked token from
  unmasked context — and its `1/t` weight concentrates on **masked** positions, where the
  answer has to come from the geometry. That is the exact opposite of `1/σ²` concentrating on
  positions where the element is already revealed.
- There is no continuous channel to drift: a token is either `MASK`, carrying no information
  at all, or committed to a value that was correct in training. No fixed-point to fall into.
- Commitment is explicit and scheduled, so the number of irreversible decisions per step is a
  knob rather than an emergent property, and remasking correctors (ReMDM) can undo one.

None of that is guaranteed to fix the heteroatom frequencies, and this document should stop
claiming a predicted win. The honest framing: the continuous parameterization has a
reproducible composition bias that survives every hyperparameter tried, more training, and
both decoding rules; absorbing diffusion changes the mechanism that plausibly causes it. **The
calibration probe above decides whether that is the right story before a single model is
trained.**

---

## 4. Design decisions

Grouped; `★` marks a recommendation. Decisions that are cheap to reverse are flagged.

### A. Framing and scope

**D-A1. Is the type model a classifier bolted onto the force field, or does it stay inside
the pseudo-PES framing?**
GPFF's identity is "the generative model *is* a force field". A softmax head is not a force.
Options: (a) **hybrid** — one shared PaiNN backbone, one equivariant force head, one
invariant logits head; (b) **continuous one-hot** — append `|V|` channels to the per-atom
target and let the *existing* pseudo-energy `|x̃ − x_0|²` cover them, i.e. EDM-style types
with zero new machinery and the framing fully intact; (c) a pseudo-energy over the type
simplex with `logits = −∂E/∂a`.
★ **Do (a). (b) already exists and is already tuned — do not rebuild it** (§3.5). The control
is `typed-ostrich-type-scale-0.3` at 95.3% validity, 2.9 points behind the untyped baseline at
`sdd256`. Skip (c). Note that §3.5 also answers the FlowMol-vs-EQGAT-diff question *for this
codebase*: at matched `type_scale` the categorical CE head and the continuous MSE head are
within ~2 points of each other, so the head is not the lever. The lever is how types are read
out at sampling time.

**D-A2. Which axes does the thesis actually vary?** Two independent axes, and conflating them
will make the results unreadable:
- *type corruption*: continuous one-hot / uniform categorical / **absorbing-masked**;
- *size handling*: copied from data (status quo) / size prior / `ABSENT` on a canvas.
★ Fix the 2×2 that matters: {continuous, absorbing} × {size given, `ABSENT`}, with current
GPFF as the (—, given) corner. Three of the four corners' continuous row is already run, so
the marginal cost of the grid is low. **The hypothesis to test is narrow and falsifiable:
absorbing types restore the heteroatom frequencies that the continuous parameterization loses**
(N in 61.3% of molecules, not 46%), at equal or better validity. State it that way in the
thesis — it is a sharper claim than "discrete is better". Note that the first mechanism this
document proposed for that deficit (greedy readout) was measured and rejected in §3.6; the
claim above survives, the explanation for it is open, and the §5 step 1 probe is what narrows
it down.

**D-A3. Bonds, charges, hydrogens.** GPFF is bond-free by design — RDKit perceives bonds from
geometry (`analysis/structure.py`), which is a genuine differentiator from DiGress/MiDi/
EQGAT-diff.
★ **Stay bond-free in v1** (QM9, where GPFF's existing numbers live). But treat it as a known
cliff, not a settled choice: MiDi measured 6% stable molecules on GEOM-Drugs for
EDM-with-distance-inferred-bonds against 92% for joint bond generation. Bond-free plausibly
survives QM9 and plausibly collapses on GEOM-Drugs and CrossDocked. Since the same absorbing
machinery over a pair vocabulary `{none, single, double, triple, aromatic} ∪ {MASK}` would
extend to bonds, the vocabulary and transform design should not foreclose it — and EQGAT-diff
and FlowMol3 both report formal charges help validity too, with CrossDocked ligands including
charged species. Decide *now* whether bonds are out of scope for the MSc or a planned phase 2;
it changes how §5 step 1 is written.

### B. Vocabulary and state representation

**D-B1. Vocabulary contents and whether it is global.** QM9 needs `{H,C,N,O,F}`; GEOM-Drugs
adds `{S,Cl,Br,P,I}` (+B, Se in some splits); CrossDocked ligands `{C,N,O,F,P,S,Cl,Br,I}`.
★ Define **one global, explicitly ordered vocabulary** now, in a single module
(`src/gpff/vocab.py`), with a per-dataset **allowed-token mask** applied to the logits at
train and sample time. Rationale: cross-dataset transfer (EQGAT-diff's pretrain→fine-tune
result) needs a stable index space, and the mask is what prevents the fluorine-pitfall
failure mode in its logit form. Hard to reverse later — a vocabulary change invalidates every
checkpoint.

**D-B2. What carries the discrete state through the batch?** (a) Reuse `properties.Z` with
reserved slots (`Z=0` → `ABSENT`, which is already ASE's dummy "X"; `Z=99` → `MASK`): zero
change to PaiNN's `nn.Embedding(100, …)` and to collation. (b) A new batch key plus a
separate `nn.Embedding(|V|, …)` summed into `q`: cleaner, but needs threading through
`compute_neighbors` (`utils/neighbors.py:64`, which rebuilds the per-molecule dict from a
hardcoded key list — it does re-add unknown keys via `skipped_keys`, so this works, but check
it) and through `clone_batch` / `batches_to_relaxed`.
★ **(a)**, with the mapping owned by `vocab.py` so it is stated once. Then audit every place
that treats `Z` as a physical atomic number: `dict_to_atoms` / `batches_to_relaxed`,
`analysis/structure.py`, `align_permutation`, `AllToAllNeighborList`, and the ASE db write in
`cli/generate.py:201`. **`ABSENT`/`MASK` atoms must be stripped before anything chemical
sees them.**

**D-B3. Is `ABSENT` a vocabulary class, or a separate binary existence channel?** As a class
it shares the softmax with elements, so "is there an atom here" and "which element" are one
decision and are automatically consistent. As a separate Bernoulli channel the two can be
weighted and thresholded independently (useful if `ABSENT` dominates the canvas and swamps
the CE).
★ **Vocabulary class**, as the user's formulation has it and as DrugFlow does — but *measure
the class balance*, because on a global canvas `ABSENT` can easily be the majority class, and
plan D-D3's weighting with that in mind.

**D-B4. Explicit or implicit hydrogens.** QM9 is explicit; the repo already carries a
heavy-atom GEOM-Drugs dataset (`geom_drugs/5b56717d5c63eea0`). With `ABSENT` on a fully
connected `O(N²)` graph, hydrogens dominate the cost and the canvas size.
★ Explicit for QM9 (comparability with existing runs and the paper); plan heavy-atom-only for
the pocket work.

### C. Forward process and the canvas

**D-C1. Absorbing vs uniform vs marginal transitions.**
★ **Absorbing**, for the RADD time-freeness argument in §2 — it is the one choice that is
*specifically* motivated by GPFF's architecture rather than by generic benchmark numbers.
Record, honestly, that UDLM finds the gap negligible at `|V| ≈ 12` and that uniform is better
for guidance; keep uniform as a named ablation, and note that DiGress's *marginal* transitions
are a third option that encodes element frequencies for free.

**D-C2. Mask schedule `α_t`.** Linear (`1 − t`) gives loss weight `1/t`; cosine concentrates
steps where it matters.
★ Linear for training (simplest correct ELBO), cosine for the *sampling* count schedule.
Under a time-free network the training schedule only enters through the loss weight and the
`t` distribution, so this is cheap to change later.

**D-C3. How are the geometry noise level `σ` and the type mask rate `t` coupled during
training?** The pivotal decision. (a) **Tied** (`t = σ/σ_max`, reusing the existing `t`): one
schedule, matches MiDi's single-time view, but the model never sees "clean geometry + fully
masked types", which is exactly the regime a geometry-first sampler and type-infilling need.
(b) **Independent**: draw `σ` and `t` from independent marginals, so the model learns the
whole 2D corruption grid. (c) **Correlated** with a copula / shared latent.
★ **(b), independent.** It costs nothing (two draws instead of one), it is the honest
extension of GPFF's time-agnostic premise — the model must infer *both* corruption levels
from its input — and one trained model then serves three evaluation modes: joint generation,
type infilling at fixed geometry, and geometry refinement at fixed types (the last reproduces
current GPFF, giving a free apples-to-apples control). It also opens the sampler's path
through the `(σ, t)` plane as a tunable, which is a genuine research contribution in its own
right. MiDi's adaptive per-modality schedule is the prior art to cite.

**D-C4. Per-molecule or per-atom mask rate?** The standard masked-diffusion forward process
masks each token independently given one molecule-level `t`, which already yields a spread of
per-molecule masked *fractions*. A per-atom `t` (as `VarianceExplodingGeometriesOld.per_atom_noise`
once did for σ) would decouple further.
★ Per-molecule `t`, independent Bernoulli per atom. Standard, matches the ELBO.

**D-C5. Canvas policy — how many slots, and where do `ABSENT` atoms sit in `x_0`?** The
hardest practical decision. The pseudo-energy `|x_t − x_0|²` is **undefined** for a slot with
no true position.
Options for the canvas: global `N_max` (29 for QM9, ~180 for GEOM — ruinous for an `O(N²)`
graph); `N_true + slack` per molecule; or `N ~ size prior + slack`.
Options for `x_0` of an `ABSENT` slot: (i) exclude from the force loss entirely (weight 0) —
their positions are then unconstrained and they wander; (ii) **self-target** (`x_0 := x_t`,
hence zero force) — well-defined label, "stay put", trivially implementable; (iii) place at
the centroid — *don't*: coincident atoms give `d_ij → 0` in a fully connected graph and the
`1/(d_ij + norm_epsilon)` in `painn.py:130` will produce exactly the kind of explosion the
`skip_exploding_batches` machinery exists to paper over.
★ **`N ~ size prior + small slack`** (the DrugFlow/READ hybrid — the prior gets the ballpark,
`ABSENT` corrects it), with **(ii) self-target and zero force weight**. Starting from a global
`N_max` is the version most likely to fail, both on cost and on class balance.

**D-C6. Do `MASK` and `ABSENT` atoms participate in message passing?** `MASK` atoms *must*
(they are atoms-to-be, and their geometry must be denoised). `ABSENT` atoms must not corrupt
real atoms' environments. A learned `ABSENT` embedding emitting near-zero messages is not
enough: at training time the network sees `ABSENT` with its type known, while at sampling
time the same slot is `MASK` with its type undecided, so ghosts sitting inside the molecule
will perturb everything.
★ **Hard-mask edges incident to `ABSENT` atoms** (and exploit §E5: once decoded, `ABSENT` is
irreversible, so the atom can simply be deleted). Do not rely on the network learning to
ignore them.

**D-C7. `align_permutation` ordering.** `utils/alignment.py:486` sets a `1e12` cost on
type-mismatched pairs. If types are masked *before* alignment, every pair mismatches and the
Hungarian assignment degenerates. Order of operations in `PseudoPESTransform.forward` must
be: noise geometry → align permutation **using the clean types** → *then* mask types, with
`ABSENT` slots excluded from the matching.
★ Add a test for this; it is a silent failure, not a crash.

### D. Model and loss

**D-D1. Logits head.** Types are invariant → an `Atomwise`-style scalar MLP on
`scalar_representation` → `|V|` logits, as a second entry in `model.output_modules` beside
the existing `GatedEquivariant`. ★ Shared backbone, separate head (as MiDi/EQGAT-diff).

**D-D2. Time conditioning of the type head.** ★ **None**, per RADD — and say so explicitly in
the thesis, because it is the point where GPFF's design and absorbing diffusion's theory
coincide. Keep `TimeConditionedGatedEquivariant` as the ablation.

**D-D3. Loss construction and balance.** Masked-position-only CE, weight `1/t` clamped at
`t_min`, combined as `L = L_force + λ · L_type`. Known pain point: EQGAT-diff attributes much
of its gain to loss weighting. Also note `task.py:22`'s `loss_fn` is **hardcoded to
`(pred − target)²`** — a `CategoricalModelOutput` subclass is needed; the existing
`weights_key` plumbing (`task.py:91`) can carry the per-atom `1/t · [masked]` weight tensor
unchanged.
★ Normalise both terms to `O(1)` at init, log them separately from day one, sweep `λ` over
~3 values, and clamp `1/t`. Also decide whether `ABSENT` positions are down-weighted in the
CE (see D-B3).
Two priors from §3.5, both from runs at `s = 0.5`: **weight the type loss generously**
(`loss_weight` 4 beat 0.5 by 2.1 points) and **keep a σ-dependent weight on it** (dropping it
cost a further 0.8). The second is worth re-examining rather than copying, though: `1/σ²`
concentrates type learning where `h_t` *already reveals* the element, i.e. the trivial regime,
whereas absorbing diffusion's `1/t` on masked positions concentrates it where the answer must
come from the geometry. Same direction, opposite rationale — so measure it again, do not
inherit the setting.
Also reuse the branch's `cross_entropy_loss` and `ModelOutput(cross_entropy=True)`
(`task.py`): the `CategoricalModelOutput` this decision calls for is already written.

**D-D4. Force loss on `MASK` and `ABSENT` atoms.** `MASK` atoms: keep full force weight — the
model *should* learn to denoise geometry without knowing the element. `ABSENT`: zero weight
(D-C5). ★ As stated; expose as config, because "does geometry denoising degrade when types
are unknown?" is a headline result.

**D-D5. Embedding `MASK` and `ABSENT`.** ★ Reuse the `nn.Embedding(100, …)` rows per D-B2.
Both tokens will be seen constantly, so no untrained-row risk — unlike the element rows,
which still need D-B1's logit mask.

**D-D6. Does the force head see type information, or vice versa?** Single forward pass, both
heads off the shared representation. A two-pass scheme (predict types → re-embed → predict
forces) doubles cost for unclear gain. ★ Single pass; defer.

### E. Sampler

**D-E1. Step budget and interleaving.** The geometry loop is 50 `DirectDenoisingSampler`
steps. How many unmasking events, and when? (a) all at the last step — "geometry first, then
type"; (b) all at the first step — "type first", which reduces to current GPFF after step 1;
(c) spread over a cosine count schedule; (d) **driven by the per-atom force norm** — unmask
the atoms whose local geometry has settled.
★ Implement (c) as the default and **(d) as the novel contribution**: GPFF already estimates
σ from force spread (`sample/base.py:103`), so a per-atom force norm is a ready-made
confidence signal for *when* a site is ready to be typed. That is a GPFF-native coupling with
no analogue in the papers above, and it is the most defensible "my own idea" in this design.
Also run (a) and (b) as interpretable extremes.

**D-E2. Unmasking order.** Random / confidence (MaskGIT) / geometry-driven (D-E1d).
★ Random as the correct-by-construction default; confidence as an ablation, with the
[clustering caveat](https://iclr.cc/virtual/2025/poster/29658) — in 3D, confidence ordering
means "decode one end of the molecule first", which is plausibly good *or* a diversity
killer. Measure it.

**D-E3. Sampling vs argmax, temperature, top-k.** ★ **Sample from the logits**, with a
temperature knob and D-B1's allowed-token mask: it is the principled choice and it costs
nothing. But **do not expect it to buy anything.** §3.6 measured exactly this swap on the
continuous port and it moved the nitrogen share by 0.04 pp, i.e. not at all, because the
marginals it decodes are already near-deterministic. Keep reporting the histogram under both
rules — it is one extra evaluation, not a training run — and treat a *difference* as the
signal that the model is genuinely hedging, which the continuous port is not.

**D-E4. Carry-over only, or a remasking corrector?** Independent parallel unmasking produces
incoherent compositions at low step counts — the chemical version of ReMDM's "She sell". ★
Carry-over (SUBS) for v1; ReMDM as a late, retrofittable experiment requiring no retraining.

**D-E5. Delete `ABSENT` atoms during sampling, or keep them with masked edges?** Deletion is
*correct* under absorbing diffusion (unmasked tokens never change), it shrinks the `O(N²)`
graph monotonically, and it matches DrugFlow. But it churns `idx_m`, `n_atoms` and the
neighbour list mid-loop, and the samplers keep per-atom/per-molecule state (`sigma_curr`,
`converged`, `nfe`, `x_prior`) that `update_batch` indexes positionally
(`sample/base.py:288`), plus it breaks `torch.compile`'s shapes.
★ **Keep-with-masked-edges first** (no index churn, constant shapes), add a batch-compaction
utility later as a performance optimisation. Cheap to reverse in this order; expensive in the
other.

**D-E6. Where does the generation batch come from?** `cli/generate.py:155` iterates the test
dataloader, so composition and count leak from real molecules. Generating composition means
the batch must be *synthesised*: `N` from the size prior, all types `MASK`, positions from
`GaussianPrior(σ = σ_max = 30)`.
★ A new `Prior` that **constructs** a batch rather than overwriting one, plus a branch in
`generate.py`. Note that `n_samples > len(test_dataset)` is currently an error
(`generate.py:130`) — that limit disappears, which is itself a win. Keep the
iterate-the-test-set path for the types-given control.

**D-E7. Convergence and NFE accounting.** `f_max` convergence freezes a molecule when forces
are small (`direct_denoising.py:117`). A molecule with masked types is *not* converged no
matter how small the forces. ★ Convergence must require "all types decoded **and** forces
below threshold"; NFE reporting should stay comparable to existing runs, so decide explicitly
whether cached (RADD-style skipped) type passes count.

### F. Evaluation

**D-F1. New metrics are mandatory.** Current metrics assume the composition is given. Once it
is generated, report at minimum: atom-type distribution TV/KL vs the dataset, **size
distribution** match, molecule stability (EDM's per-atom valency check), and bond-length/angle
JS divergences. ★ Non-negotiable — without the size and type histograms, validity numbers are
uninterpretable (a model that emits only methane scores 100% valid).

**D-F2. Paired comparison breaks.** `CLAUDE.md` prescribes paired McNemar against a control
on the same split and noise. With generated composition there is no molecule to pair *on*.
★ Pair on the **prior seed** (same canvas, same initial noise, same mask order → two models'
outputs are paired), and add distributional two-sample tests for the histogram metrics.
Decide this *before* running, or the comparisons will not be analysable.

**D-F3. Keep a types-given mode.** Setting the mask rate to 0 at sampling must reproduce
current GPFF behaviour. ★ Guard it with a test and use it to compare directly against
existing runs — otherwise this branch has no baseline.

**D-F4. Required controls.** Current GPFF (types given, size given); continuous one-hot types
(D-A1b); size-prior-without-`ABSENT`; uniform instead of absorbing. ★ Budget for these in the
plan; each is a training run, and the 48 h `gpu-2d` limit means each is also a resume chain.

### G. Infrastructure

**D-G1. Cost.** The graph is fully connected (`AllToAllNeighborList`), so pairs scale as `N²`.
Canvas slack is quadratically expensive: QM9's mean `N ≈ 18` → a global canvas of 29 is ~2.6×
the pairs. ★ Settle D-C5 on this basis; re-tune `batch_size` (default 512) and re-measure
before launching long runs.

**D-G2. `loss_threshold` recalibration.** `AtomisticTask(loss_threshold=1e3)` skips
"exploding" batches. Adding a CE term changes the loss scale, so the threshold silently
changes meaning. ★ Re-derive it, and keep watching `train/skipped_nan` (`gp run` surfaces NaN
skips).

**D-G3. Reproducibility.** Drawing the mask with the torch CPU RNG *inside*
`PseudoPESTransform` makes it a function of `(seed, epoch, db index)` via `_PinnedNoise` —
free, and required for `per_sample_train_noise` + persistent workers. ★ Draw it there, not in
a collate function or a callback.

**D-G4. `torch.compile`.** `compile={dynamic:true}` already handles variable `N`. Atom
deletion mid-sampling (D-E5) would trigger recompiles — one more reason to defer it.

**D-G5. No new dependencies or datasets.** CE is `torch`; the three dataset families already
exist on hydra. ★ Same containers; the only new asset is the size prior (fit on the train
split, cached like `ShapeGMM` already is in `sample/shape_predictor.py`).

**D-G6. Config surface.** New Hydra pieces: `vocab`, a type-corruption block under
`generative_transform/pseudo_pes/`, a second `output_modules` entry, a
`task.outputs` entry for the categorical loss, and sampler options for D-E1/E2/E3. ★ Add an
`experiment/gpff_types.yaml` rather than mutating `gpff.yaml`, so every existing run stays
reproducible.

---

## 5. Suggested order of work

0. **The metrics, first.** Element histogram, molecule-level heteroatom presence, size
   distribution, uniqueness (D-F1) wired into the eval pipeline, and the seed-paired protocol
   settled (D-F2). The existing `results.json` records only `n_molecules` and
   `rdkit_validity`, which is why the heteroatom deficit went unnoticed through 20 runs;
   §3.5 had to be reconstructed from `samples.db` and `metadata/*.npy` after the fact. Do this
   before anything else — it is cheap and it is what makes every later run readable.
1. ~~The argmax-vs-sampled readout test~~ — **done, §3.6: the readout is not the cause.**
   What replaces it is the **calibration probe**: true test geometries, `h` set to pure noise
   at `σ_max`, one forward pass, predicted marginal averaged over atoms against the dataset's
   element frequencies. No training, no sampling loop. It separates "the model's marginals are
   miscalibrated" (loss weighting and parameterization at fault — then absorbing's `1/t` on
   masked positions is the fix, and D-D3 is the decision that matters) from "the marginals are
   fine, the sampling dynamics drift" (then the fix is discrete commitment, and D-E1/D-E5 are
   the decisions that matter). Those two answers lead to different designs, so this runs
   before anything is built.
2. `vocab.py` + `ABSENT`/`MASK` plumbing through `Z`, with the downstream audit (D-B1, D-B2).
   The `align_permutation` problem (D-C7) is already solved on the branch; port that fix
   rather than reinventing it.
3. Absorbing types at *fixed* size: masking in the transform, logits head, reuse
   `ModelOutput(cross_entropy=True)`, independent `(σ, t)` draws (D-C3). Validate the
   types-given mode reproduces current GPFF (D-F3), and compare against
   `typed-ostrich-type-scale-0.3` as the continuous control — it already exists at 6
   milestones and 6 sampler configs, so no baseline run is needed.
4. `ABSENT` + size prior + canvas slack (D-C5, D-C6, D-E5, D-E6).
5. Sampler study: cosine vs force-norm-driven unmasking (D-E1), ordering (D-E2), then ReMDM
   (D-E4). Note from §3.5 that the continuous channel needed ~256 steps to close to within 3
   points of the baseline and was 10–19 points behind at 16 — so **low-NFE performance is the
   natural place to claim a win**, since masked diffusion resolves types in a chosen number of
   discrete events rather than by annealing.
6. Pocket conditioning: reuse the scaffold/`context_mask` path, swap QM9 → CrossDocked,
   condition the size prior on pocket geometry.

Steps 2–3 are where the irreversible choices live (vocabulary indices, state carrier,
coupling of the two corruption levels). Everything from 4 onward is additive.

---

## 6. The three things most likely to go wrong

1. **Ghost atoms.** `ABSENT`/`MASK` slots with meaningless coordinates inside a fully
   connected graph, no edge masking, and an undefined force target — this is the one that
   produces the `1e26`-force failures the repo has already been bitten by once. D-C5, D-C6.
2. **Train/sample mismatch on masked geometry.** At training the model sees `MASK` atoms in
   *nearly correct* positions (they carry the true `x_t`, only the label is hidden). At
   sampling, early steps have `MASK` atoms in pure `σ=30` noise. If `σ` and `t` are tied
   (D-C3a), that mismatch is never trained on. Independent draws are the defence.
3. **Uninterpretable results — already happened once.** Twenty runs of the continuous port
   were evaluated on `rdkit_validity` alone, and a flat ~25% nitrogen deficit survived the
   entire `type_scale`, `loss_weight` and MSE-vs-CE sweep without showing up in any reported
   number. Validity *rose* across part of that sweep while composition fidelity did not move.
   Settling D-F1/D-F2 before the runs is the cheapest lesson available here.
