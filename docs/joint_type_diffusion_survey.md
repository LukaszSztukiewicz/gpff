# Jointly generating atom types and positions: what the literature does

Survey written against the `ccr-916e702f-ut0j4o` branch, which adds joint
diffusion of atom types to GPFF (`experiment=gpff_types`,
`experiment=gpff_types_categorical`) and underperforms positions-only GPFF.

The short version: **the branch implements the one design that every controlled
comparison in the literature reports as the weakest option** — Gaussian noise on
a scaled one-hot vector, on the same noise schedule as the coordinates, with the
one-hot scale left at 1.0. Three of the knobs the literature identifies as decisive
(relative feature scale, per-modality noise schedule, per-modality loss weight)
are currently set at or near the values that papers ablate as the bad end.

---

## 1. The four families

### A. Gaussian diffusion on scaled one-hot vectors (what this branch does)

[EDM](https://arxiv.org/abs/2203.17003) (Hoogeboom et al., 2022) is the origin.
A molecule is `[x, h]`; `h` is one-hot; both get the same VP schedule
`q(z_t|h) = N(z_t | α_t h_onehot, σ_t² I)`; the type is decoded by argmax, and
the likelihood term integrates the Gaussian over `[−½, +½]` around the active
class.

The paper's own ablation is the single most relevant result for this branch:

| EDM input scaling | NLL | Atom stable | **Mol stable** |
| --- | --- | --- | --- |
| `[x, 1.00·h_onehot, 1.0·h_charge]` | −103.4 | 95.7% | **46.9%** |
| `[x, 0.25·h_onehot, 0.1·h_charge]` | −110.7 | 98.7% | **82.0%** |

Molecule stability nearly doubles from a scale factor. EDM's stated reason:

> when the features h are defined on a smaller scale than the coordinates x, the
> denoising process tends to first determine rough positions and decide on the
> atom types only afterwards

and in the appendix:

> it seems that it is easier to learn a denoising process where the atom type is
> decided later, when the atom coordinates are already relatively well defined.

Note the scale factor only shrinks the *signal* (`α_t·0.25·h_0`); the noise
`σ_t` is shared with the coordinates. That is exactly what `type_scale` does in
`PseudoPESTransform` — so `type_scale: 1.0` is EDM's 46.9% row, and `0.25` is
its 82.0% row.

Supporting tricks in this family:

- **x₀-parameterization over ε.** [EQGAT-diff](https://arxiv.org/abs/2309.17296)
  (Le et al., 2023): on GEOM-Drugs, x₀ gives 90.46% vs 85.19% molecule
  stability for ε. The branch already predicts `h_0` directly — correct choice.
- **Self-conditioning.** [Analog Bits](https://arxiv.org/abs/2208.04202)
  (Chen, Zhang & Hinton, 2022), the canonical reference for continuous diffusion
  over discrete data: feed the model's own previous `x̃_0` estimate back in
  alongside `x_t`, concatenated; zero it 50% of the time during training; no
  gradient through the first pass; < 25% training overhead.
- **Asymmetric time intervals** (same paper): evaluate the denoiser at
  `t' = t + ξ` at sampling time. Matters most at low step counts (BLEU-4 17.1 →
  31.6 at 5 steps in their captioning experiment).

### B. Genuinely discrete state spaces (the current mainstream)

D3PM-style categorical diffusion or CTMC-based discrete flow matching for types,
Gaussian/CFM for coordinates. Used by [MiDi](https://arxiv.org/abs/2302.09048),
[EQGAT-diff](https://arxiv.org/abs/2309.17296),
[MatterGen](https://arxiv.org/abs/2309.04475)-style crystal models,
[SemlaFlow](https://arxiv.org/abs/2406.07266),
[FlowMol3](https://arxiv.org/abs/2508.12629),
[FLOWR](https://arxiv.org/abs/2504.10564),
[TABASCO](https://arxiv.org/abs/2507.00899).

The engineering payoff that matters most for a force-field-shaped model:
**the network's type input is always a legal element** (or an explicit MASK /
"fake atom" token) — never a blurred real vector of unbounded magnitude. A
standard `nn.Embedding` nuclear embedding keeps working; no Gaussian-RBF
expansion of `h_t`, no magnitude overflow at `sigma_max`, no out-of-distribution
inputs.

Two controlled comparisons:

**EQGAT-diff, QM9** (identical architecture, x₀ parameterization, only the type/bond
state space changes):

| | discrete | continuous one-hot |
| --- | --- | --- |
| Mol. stability | **98.68%** | 96.45% |
| Validity | **98.96%** | 96.79% |
| Atom stability | **99.92%** | 99.79% |

> We discover that EQGAT^x0_disc outperforms EQGAT^x0_cont in all evaluation
> metrics on the QM9 and GEOM-Drugs dataset. Hence, employing the categorical
> diffusion for discrete state-space in the x0-parameterization is the preferred
> choice.

**[KLDM](https://arxiv.org/abs/2507.03602), MP-20** — a clean three-way comparison
of exactly the options available here, everything else fixed:

| atom-type model | coord. RMSD ↓ | E above hull ↓ | Stable % ↑ |
| --- | --- | --- | --- |
| (C) continuous one-hot | 0.371 | 0.269 | 38.62 |
| (C-AB) continuous analog bits | 0.296 | 0.187 | 49.84 |
| (D) discrete absorbing state | **0.283** | **0.155** | **59.21** |

Note the first column: the continuous-one-hot variant has the worst **coordinate**
accuracy too. A badly modelled type channel degrades the geometry — which is the
effect being observed on this branch.

### C. Don't diffuse the types at all

[UniGEM](https://arxiv.org/abs/2410.10516) (ICLR 2025) is the closest analogue to
GPFF's situation and the most actionable result here. It diffuses **only**
coordinates and treats atom type as a *prediction* task with its own head,
activated only in the low-noise "growth" phase (`t ∈ [0, t_n]`, with `t_n = 10`
of `T = 1000`, i.e. the last ~1–3% of the trajectory):

> Unlike traditional generative models that typically perform joint diffusion of
> atom types and coordinates, our approach focuses solely on the diffusion of
> coordinates, treating atom type as a separate prediction task. This separation
> is justified by the observation that atom types can often be inferred based on
> the determined scaffold.

Their measured error decomposition on QM9 vs EDM:

| | UniGEM | EDM |
| --- | --- | --- |
| Atom type error | 1.3e-5 | 0.0077 |
| Coordinate error | 0.2247 | 0.2430 |

And the mechanism they identify, which reads like a description of this branch's
sampler:

> even at the later stage of generation, the predicted atom type (determined by
> the highest probability) oscillates between two categories, indicating a
> suboptimal learned distribution. Since the coordinates in the next step depend
> on both the current coordinate and atom type, this oscillation introduces
> additional errors in coordinate estimation.

They also argue the dimensionality reduction itself helps: the prior and
discretization error terms in their TV bound both shrink when only coordinates
are diffused. Reported effect: ~10% improvement in molecule stability over EDM.

### D. Unified parameter / latent spaces

- [GeoBFN](https://arxiv.org/abs/2403.15441) (ICLR 2024 oral): Bayesian Flow
  Networks — run the generative process in the *parameter space of
  distributions* (means for coordinates, class probabilities for types) rather
  than on the data. The discrete/continuous mismatch disappears because
  everything is a distribution parameter. 90.87% molecule stability on QM9.
- [ADiT](https://arxiv.org/abs/2503.03965): joint VAE latent, then a plain
  latent DiT. Their motivation for abandoning the EDM-style setup directly:
  two independent diffusion processes for categorical types and continuous
  coordinates mean the trajectory "passes through unrealistic intermediate
  states during the denoising trajectory."

### The useful negative result

[FlowMol](https://arxiv.org/abs/2404.19739) built **SimplexFlow**, which keeps
the categorical trajectories on the probability simplex — the "theoretically
principled" fix for continuous relaxation — and found it did *not* help:

> we find that ... it yields superior performance to SimplexFlow

(referring to the unconstrained variant that does not respect the categorical
structure). FlowMol3 later abandoned simplex flows for CTMC discrete flow
matching. **Conclusion: don't spend effort projecting `h_t` onto the simplex.**
Either keep the unconstrained relaxation and fix the scaling/schedule/weighting,
or go fully discrete.

---

## 2. The three cross-cutting knobs, and their literature values

### (i) Relative feature scale / resolution order

Every paper in families A and B arranges for **geometry to resolve before types**.
EDM does it with the 0.25 scale factor. MiDi does it with per-modality noise
schedules. UniGEM does it by only predicting types at the very end.

MiDi's justification is the deepest statement of why:

> while the 2D connectivity structure can be predicted relatively well from the
> 3D conformation, the converse is not true as the conformation is not unique for
> a given structure

The information flows one way. Geometry determines types; types do not determine
geometry. So resolve geometry first and read the types off it.

### (ii) Per-modality noise schedules

[MiDi](https://arxiv.org/abs/2302.09048) modifies the cosine schedule with a
per-component exponent ν:

```
ᾱ_t = cos²( (π/2) · ((t/T + s)/(1 + s))^ν )
```

QM9 values: **ν_r = 2.5 (coordinates), ν_y = 1.5 (bonds), ν_x = ν_c = 1 (atom
types, charges)**. Coordinates retain signal longer; types are destroyed faster
in forward time and therefore resolve last in reverse time. EQGAT-diff adopts
the same schedule and reports it "significantly" outperforms the plain cosine
schedule. FlowMol defines separate "interpolant schedules" per modality for the
same reason.

**On this branch both modalities share one σ_t exactly** (`pseudo_pes.py:78`:
`h_t = h_0 + sigma_t * torch.randn_like(h_0)`).

### (iii) Per-modality loss weights

Remarkably consistent across the field — **types always weighted *below*
coordinates**:

| Model | coordinate weight | atom-type weight | ratio |
| --- | --- | --- | --- |
| MiDi | λ_r = 3 | λ_x = 0.4 | 0.13 |
| EQGAT-diff (adopts MiDi's) | λ_x = 3 | λ_h = 0.4 | 0.13 |
| SemlaFlow (QM9) | λ_x = 1.0 | λ_a = 0.2 | 0.20 |
| KLDM (discrete types) | 1.0 | λ_a = 0.33 | 0.33 |
| TABASCO | 1.0 | λ_discrete ∈ (0, 1] | ≤ 1 |
| **`gpff_types`** | **1.0** | **4.0** | **4.0** |

The branch's `loss_weight: 4.` is derived correctly — `4|ĥ_0 − h_0|²` is the
squared error on the type force `2(h_0 − h_t)` — but a derivation that fixes the
*scale* consistent with the pseudo-PES is not an argument that this is the
*optimal* training weight. It lands ~20–30× above every published value.

### (iv) Time-dependent weighting

EQGAT-diff found this "crucial", against the usual `w(t) = 1`: a truncated-SNR
weighting (clamping `SNR(t)` between 0.05 and 1.5) lifted GEOM-Drugs molecule
stability from 87.59% to 91.60%, and a model trained with it at T = 100 steps beat
a uniformly-weighted model at T = 500.

GPFF's `SigmaWeights` (`1/σ²` clipped to `[0, 1]`) is already a truncated-SNR
weighting — good for the force head. The problem is that **the same weight is
applied to the type loss**, which has a different information profile (below).

---

## 3. Why this branch loses to positions-only GPFF

Four mechanisms, in rough order of expected size. The first two are directly
addressable by config.

### (a) `type_scale: 1.0` puts the types *ahead* of the geometry

With a scaled one-hot of entry `s` and per-entry noise `σ`, argmax over 5
classes is right ~90% of the time once `s/σ ≳ 2.7`, i.e. `σ ≲ s/2.7`.

- `s = 1.0` → types legible from `h_t` alone below **σ ≈ 0.37**
- `s = 0.25` → below **σ ≈ 0.09**

QM9 geometry only pins down the bonding pattern at roughly **σ ≈ 0.3–0.5 Å**
(well under half a bond length). So at `s = 1.0` the types become readable *at
the same time as, or before,* the geometry settles — the opposite of the order
EDM, MiDi and UniGEM all engineer for. `s ≈ 0.15–0.3` puts the type commitment
after the geometry, and independently coincides with EDM's 0.25.

### (b) The type loss is concentrated where the answer is copyable

`sigma_mean = -0.7`, `sigma_std = 1.2` (lognormal, `geometries.py:88`) with
`1/σ²` clipped at 1.0 means the type cross-entropy is weighted ~flat over the
whole low-σ region. Fraction of training samples where the element is already
legible in `h_t`:

- `s = 1.0` → **≈ 40%** of samples (σ < 0.37)
- `s = 0.25` → **≈ 8%** of samples (σ < 0.09)

At `s = 1.0`, a large share of the type loss is spent on a task solvable by
`argmax(h_t)` — a shortcut that teaches the head nothing about inferring element
identity from local geometry, which is the only thing that matters at the σ
where the decision is actually made. Meanwhile the hard regime is down-weighted
by `1/σ²`. An absorbing/MASK state (family B) makes the shortcut structurally
impossible: a masked atom's input carries *zero* information about its element,
so the head must use the neighbourhood.

### (c) The force field lost its exact element conditioning

This is the fundamental cost and the reason some degradation was always
expected. Positions-only GPFF reads **ground-truth** `Z` through
`nn.Embedding`; the joint model reads a noisy `h_t` instead
(`representation.embedding_key: h`). An MLIP-shaped model deprived of element
identity is strictly weaker at every σ. On top of that comes UniGEM's feedback
term: an oscillating/indecisive type input corrupts the position prediction,
which is visible as the degraded coordinate RMSD in both the UniGEM table and
KLDM's row (C).

Family B is the resolution that preserves the inductive bias: with discrete
diffusion the input is always a real element or an explicit MASK, so the model
remains a force field evaluated on a well-defined (if partly unknown) chemical
system.

### (d) The permutation alignment changed semantics

GPFF-specific and easy to miss. Baseline `align_permutation`
(`utils/alignment.py`) **forbids cross-element matching** —
`cost_matrix[mismatch] = 1e12`. The types branch passes
`no_numbers = torch.zeros_like(atom_types)` (`pseudo_pes.py:84-90`) so every
pair is allowed, and minimizes `|Δx|² + |Δh|²` over the concatenated 8-vector.

Because the noise on `h` is `σ` — the *same* σ as positions, independent of
`type_scale` — the type block contributes `~2·n_types·σ² = 10σ²` to the cost
against the positions' `~6σ²`. **At large σ the Hungarian assignment is
dominated by matching type noise, not geometry** (roughly 60/40). The
permutation that defines the pseudo-PES target is therefore chosen substantially
at random with respect to the geometry. Lowering `type_scale` does *not* fix
this, since it only shrinks the signal. Options: weight the `h` block in the
cost matrix, run the assignment on positions only, or restore element-matching
by using the noisy argmax types.

### (e) Minor: deterministic ODE in type space

The Heun samplers step `h` with the probability-flow slope toward the posterior
mean `s·p` (`heun.py`). A posterior mean never reaches a simplex vertex, so an
under-confident atom can sit between categories for the whole trajectory — the
oscillation UniGEM describes. The categorical variant's RBF clipping to
`[−10, 10]` is a sound instinct (it makes a fully-noised type read as "unknown")
and is effectively a *soft* mask; family B is that idea taken to its conclusion,
with the commitment made by sampling rather than by an argmax at the end.

---

## 4. Suggested experiments, cheapest first

1. **`type_scale` sweep: 0.1 / 0.15 / 0.25 / 0.5** against the current 1.0.
   One config override, no code change, and it is the knob with the largest
   published effect (EDM: 46.9% → 82.0% molecule stability). Do this before
   anything else.
2. **Drop the type loss weight to ~0.2–0.5× the force weight.** Second config
   override. Log the two loss terms' gradient norms separately to see what the
   effective ratio currently is — especially in the categorical variant, where
   CE starts at `ln 5 ≈ 1.61` and is not comparable to the force MSE.
3. **Fix the permutation alignment** — weight or drop the `h` block in the cost
   matrix (§3d). This is a target-quality bug, not a hyperparameter, and it
   silently degrades the force labels the whole model is trained on.
4. **Decouple the type noise schedule** — give `h` its own σ, e.g.
   `σ_h = σ_x^ν` or a separate scale, following MiDi's ν_r = 2.5 / ν_x = 1.
   A few lines in `PseudoPESTransform`.
5. **Separate the type loss weighting from `forces_weight`** — the force head
   wants truncated `1/σ²`; the type head wants its mass where the decision is
   actually made (around `σ ≈ s`), not where the answer is copyable.
6. **Self-conditioning** (Analog Bits; FlowMol3 reports it as the most impactful
   of its three tricks for validity): concatenate the previous `ĥ_0` to the
   embedding input, zeroed 50% of the time in training. Cheap, and it directly
   targets the oscillation in (e).
7. **The UniGEM route** — stop diffusing types; keep the type head as an
   auxiliary predictor trained only for `σ` below a threshold, and read the
   elements off the final geometry. This gives up joint generation in the strict
   sense but is the best-matched published result (atom-type error 1.3e-5 vs
   EDM's 0.0077, *and* better coordinates), and it leaves positions-only GPFF
   exactly intact — so it cannot regress the current best model.
8. **The discrete route** — absorbing-state/MASK categorical diffusion for
   types, keeping Gaussian VE for positions: extend the `nn.Embedding` by one
   MASK row, train the existing CE head, sample types with a CTMC step. This is
   where the field has converged (EQGAT-diff +2.2 pts on QM9; KLDM +20.6 pts on
   MP-20) and it is the only option that preserves GPFF's force-field inductive
   bias, since the model then always sees a legal element. Biggest change, best
   evidence.

A reasonable plan: run 1–3 as a batch (config-only plus one bug fix), and if the
gap to positions-only narrows but does not close, choose between 7 (cheap, safe)
and 8 (expensive, best-supported).

---

## Sources

- Hoogeboom, Satorras, Vignac & Welling — [Equivariant Diffusion for Molecule Generation in 3D](https://arxiv.org/abs/2203.17003) (EDM)
- Vignac, Osman, Toni & Frossard — [MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation](https://arxiv.org/abs/2302.09048)
- Le, Cremer, Noé, Clevert & Schütt — [Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation](https://arxiv.org/abs/2309.17296) (EQGAT-diff)
- Feng et al. — [UniGEM: A Unified Approach to Generation and Property Prediction for Molecules](https://arxiv.org/abs/2410.10516)
- Dunn & Koes — [Mixed Continuous and Categorical Flow Matching for 3D De Novo Molecule Generation](https://arxiv.org/abs/2404.19739) (FlowMol / SimplexFlow)
- Dunn & Koes — [FlowMol3](https://arxiv.org/abs/2508.12629)
- Irwin, Tibo, Janet & Olsson — [SemlaFlow](https://arxiv.org/abs/2406.07266)
- Cremer et al. — [FLOWR](https://arxiv.org/abs/2504.10564)
- [TABASCO: A Fast, Simplified Model for Molecular Generation with Improved Physical Quality](https://arxiv.org/abs/2507.00899)
- [Kinetic Langevin Diffusion for Crystalline Materials Generation](https://arxiv.org/abs/2507.03602) (KLDM; three-way atom-type ablation)
- Song et al. — [Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks](https://arxiv.org/abs/2403.15441) (GeoBFN)
- Joshi et al. — [All-atom Diffusion Transformers](https://arxiv.org/abs/2503.03965) (ADiT)
- Chen, Zhang & Hinton — [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202)
- Jiao et al. — [Crystal Structure Prediction by Joint Equivariant Diffusion](https://arxiv.org/abs/2309.04475) (DiffCSP)
