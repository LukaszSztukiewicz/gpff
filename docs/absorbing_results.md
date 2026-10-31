# Absorbing atom types: what we built, measured and learned

Companion to `absorbing_atom_types.md` (the design note). That one says what we
*intended*; this one says what actually happened. Written 2026-10-28.

Everything below is QM9, 10 000 generated molecules per number, seed 42069.
Raw tables, scripts and job ids live in `hydra:~/gpff/scratch/absorbing/README`
and `hydra:~/gpff/scratch/typed-composition/README`.

---

## 1. The problem we started from

GPFF generates geometry but not composition: at sampling time it copies the atom
types and the atom count from a real test molecule. Only the positions are
generated. Making composition a model output is the prerequisite for
pocket-conditioned generation, where ligand size and elements have to be decided
by the model.

A previous branch (`feat/atom-types-port`) had already tried this the EDM way:
represent types as a scaled one-hot vector, add the same Gaussian noise as the
positions, and let the existing pseudo-energy cover them. Twenty runs existed,
evaluated on RDKit validity alone.

**The first thing we found was a bug in how those runs had been read.** Validity
looked fine (up to 95%), but nobody had checked *which* elements were produced.
Every single setting under-produced heteroatoms:

| | untyped baseline (types copied) | continuous port, every setting |
|---|---|---|
| molecules containing N | **61.3%** | 38.6–48.5% |
| N at atom level | **5.68%** | 2.75–4.04% |
| O at atom level | **7.75%** | 6.27–6.82% |

The deficit was flat across `type_scale`, across the loss weight, across
MSE-vs-cross-entropy heads, and across 10× more training. Carbon and hydrogen
absorbed the difference. A model that quietly drops nitrogen scores *better* on
validity, because N is where valences break — so validity alone had been
rewarding the wrong thing.

---

## 2. What we built

Absorbing-state (masked) discrete diffusion over a single token vocabulary:

```
{ABSENT} ∪ {H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br, I, Hg, Bi} ∪ {MASK}
```

- Each atom's type is replaced by `MASK` with probability `rate`, drawn per
  molecule and **independently of the position noise level**.
- The loss is cross-entropy on the masked atoms only, weighted by `1/rate` — the
  standard masked-diffusion bound.
- At sampling, every still-masked atom is unmasked independently with the correct
  reverse-process probability on a cosine or linear schedule, its element drawn
  from the predicted distribution, and never revisited.

One design choice paid off repeatedly: **`MASK` is carried as atomic number 99
and `ABSENT` as 0**, so PaiNN embeds them through its existing embedding table and
the model configuration is otherwise untouched. The continuous port had needed a
Gaussian basis expansion to cope with an unbounded one-hot channel; the discrete
version needs nothing. The absorbing implementation is strictly *simpler* than the
one it replaces.

`ABSENT` exists in the vocabulary but is disabled (its logit is forced to −10⁴).
With no padded slots in the training data it would never be trained, and an
untrained logit that can still be sampled is worse than an untrained embedding.
**So atom count is still copied from the data.** Turning `ABSENT` on is the
remaining piece of the original plan.

---

## 3. Where it ended up

At epoch 5000, 256 sampling steps, composition fully generated:

| | validity | vun | uniqueness | element TV | at%N | at%O |
|---|---|---|---|---|---|---|
| untyped baseline (types **copied**) | 98.2 | 47.9 | 92.4 | — | 5.68 | 7.75 |
| best continuous (`ts=0.15`) | **93.6** | 42.3 | 84.1 | 0.0335 | 3.97 | 6.34 |
| absorbing, cosine schedule | 90.3–90.7 | **45.3–46.0** | 90.0 | 0.0240 | 3.88 | 7.17 |
| absorbing, **linear schedule** | 88.6 | 46.0 | **92.5** | **0.0166** | 4.23 | **7.57** |

(`vun` = valid, unique and novel; `element TV` = total variation distance of the
element distribution from the data, lower is better.)

The honest summary: **absorbing loses ~3–5 points of raw validity and wins on
everything else.** Element distribution is 2× better with the linear schedule,
`vun` is 3–4 points higher, and uniqueness is 8 points higher. Its oxygen share
(7.57) is within 0.18 of the data.

Two further results matter as much as the table:

**The continuous approach plateaus; absorbing does not.** The continuous models
stop improving on composition at element TV ≈ 0.034 by epoch 1000 and are still
there at epoch 10 000, while validity and bond lengths keep improving. Absorbing
went 0.0293 → 0.0249 → 0.0240 (cosine) over epochs 500 → 2000 → 5000 and had not
flattened. So the composition error is structural for one parameterization and
merely slow for the other.

**The cost of generating composition is collapsing.** Running the same checkpoint
twice, once generating types and once with the real types supplied:

| | generating types | types given | cost |
|---|---|---|---|
| epoch 1000 | 72.1 | 91.6 | **19.5 points** |
| epoch 5000 | 90.3–90.7 | 96.2–97.1 | **5.5–6.8 points** |

And the force field itself is now at parity: with composition supplied, its bond
lengths (0.00601) **beat** the untyped baseline (0.00658), angles still trail
(1.064 vs 0.962), and overall validity is 1.1 points behind. Almost all of what
remains is the type generation.

---

## 4. The main insight: it is exposure bias, and it is in the input

We spent most of the effort finding out *why* heteroatoms are under-produced. The
answer took four measurements, each ruling out the obvious suspect.

**The decoding rule is not the cause.** The samplers read the element off as the
`argmax` of a mean-field marginal, which classically collapses onto frequent
classes. We implemented sampling instead of `argmax` and measured it on paired
trajectories. It moved nitrogen by **0.04 percentage points** — nothing. The
marginals being decoded were already near-deterministic.

**The type head is not the cause either — it is essentially perfect.** Feeding
real test geometries with every type masked, one forward pass:

- on the **true** geometry it recovers the element of 99.65–100% of atoms, and
  its aggregate distribution matches the data to a total variation of 0.0003;
- on **pure noise** its expected distribution *is* the dataset prior (N 5.76 vs
  5.68), so its prior is right too;
- only its `argmax` on pure noise collapses (94% hydrogen, never N/O/F), which is
  why the sampler must draw from the categorical — it does.

This holds at every loss weight we tried. The head is a near-perfect element
classifier given a correct geometry and a correctly calibrated guesser given none.

**Element identification needs sub-0.25 Å geometry.** Perturbing real geometries
by a known amount, nitrogen recall is 99.9% at σ=0, 84.8% at σ=0.25, **15.8% at
σ=0.5** and 0% at σ=1.0 — while the *expected* distribution stays calibrated at
every σ, falling back to the prior when the geometry says nothing.

**The geometries the model generates are off-distribution.** Running the real
sampler and probing its own intermediate structures:

| | at%N of the head's marginal |
|---|---|
| real geometry perturbed to σ≈0.25 | **5.46** (correct) |
| the model's own geometry at σ=0.255 | **3.98** |

Same model, same nominal noise level, same quantity. At the very first step, when
the structure is pure prior noise, the marginal is correct (5.76). The moment the
model's own partially-denoised geometry appears it collapses to ~4 and stays
there. A partially denoised synthetic structure apparently reads as a *confident*
C or H environment where a noised real molecule reads as ambiguous.

This is textbook **exposure bias**: the model is trained on inputs derived from
ground truth and runs on its own imperfect outputs. It is well documented for
image diffusion ([Ning et al., ICML 2023](https://arxiv.org/abs/2301.11706);
[Li et al., ICLR 2024](https://arxiv.org/abs/2305.15583)) and for language
diffusion, but we found **nothing** in the literature on exposure bias for a
*discrete head reading a continuously diffused state*, which is exactly GPFF's
situation. That gap is the interesting part of the result.

There is also a **second, compounding effect**: already-committed types bias what
is left. With the real partial context, the marginal for the remaining atoms is
worse than when everything is re-masked (N 2.86 vs 3.67 at the same step).
Committed atoms are mostly C and H, and they pull the rest the same way.

---

## 5. Two fixes tried at inference time; both fail, informatively

**Adding noise to the head's input.** Since the head is calibrated on real
geometries at every noise level, we perturbed the generated geometry before the
head read it. It does restore the marginal — but only from σ≈1.0 upward, which is
**four times** the σ at which an element can still be identified. There is no
setting that fixes the aggregate while keeping the per-atom signal: at σ≥1 the
head is simply sampling its prior, giving the right element *frequencies* attached
to the *wrong atoms*.

**Remasking (ReMDM / targeted-remasking style).** We added a corrector that puts
doubted atoms back to `MASK` so they are decided again with more context. Result:

| | validity | element TV | radical failures |
|---|---|---|---|
| no remasking | 79.9 | **0.0249** | 12.5% |
| remasking | **96.5** | 0.0474 | **2.0%** |

A spectacular validity fix — +16.7 points, valence failures down 84%, within 1.7
points of the baseline that is *handed* the composition — but composition gets
**worse**, and uniqueness drops 14 points. The reason is specific to our setting:
each re-decision is made from the *same off-distribution geometry*, so iterating
redraws from the same biased marginal and converges to its fixed point. Remasking
buys validity precisely *by* settling into safe chemistry. In language models the
re-decision gets a better context and a calibrated model, which is why the
published versions help.

**Together these establish something useful:** no inference-time intervention on
the type decision can fix this, because each one either destroys the information
that identifies the element or redraws from the same biased marginal. The bias
lives in the input distribution, so the fix has to change what the model is
trained on.

Remasking is still worth keeping as a switch: it is free, needs no retraining, and
whether to use it depends on the metric. Element TV prefers it off, validity and
failure counts strongly prefer it on, `vun` marginally prefers it off.

---

## 6. Conclusions we reached and then had to withdraw

Listed because each was stated confidently and measurement overturned it, and
because the pattern is worth noticing: **every one came from generalising a single
operating point.**

1. *"Sampling instead of argmax will restore the heteroatoms."* Moved nitrogen by
   0.04 pp. The marginals were already sharp.
2. *"Lower loss weight is monotonically better, so try 0.06."* It is not; 0.06 is
   worse than 0.25 on validity and `vun`. The monotone run we saw was one side of
   a peak.
3. *"The optimum is at loss weight 0.25."* True at epochs ≤2000 only. By epoch
   5000, weights 0.25 and 1 are indistinguishable. The weight controls
   **convergence speed, not final quality** — so it is not worth sweeping further.
4. *"The type head inherently costs force-field quality."* An artifact of one loss
   weight. At a low weight, bond lengths with composition supplied **beat** the
   untyped baseline.
5. *"Absorbing beats both continuous parameterizations at matched epoch."* Only
   against the two *untuned* continuous runs, which happened to be the only ones
   evaluated at that epoch. The tuned continuous runs are much stronger.
6. *"Hold an atom's type until its local σ drops below 0.25 Å."* Unusable: σ is
   estimated from force magnitudes and already reports 0.23 while the marginal is
   still wrong. It reports "converged" on a structure that is numerically settled
   and chemically wrong.
7. *"It is a nitrogen deficit."* It is an **oxygen** deficit that absorbing largely
   fixes (7.17–7.57 vs the continuous 6.27–6.34, true 7.75) plus a nitrogen
   deficit it does **not** (3.86–4.23 vs the continuous 3.93–3.97). Worth stating
   precisely in any write-up.

---

## 7. What is ruled out, and what is left

**Ruled out by measurement** (not by argument): more loss-weight values; more
type-head training; a different type loss; the head's prior; the argmax-vs-sampling
readout; a σ-threshold commitment schedule; and inference-time noise injection.

**The one surviving candidate** is input perturbation during training, in the style
of Ning et al.: add extra noise to the geometry the model reads while keeping the
clean type as the label, so the training inputs resemble the imperfect geometries
the model will actually condition on. It is the only option that attacks the cause.
One training run, a few lines in the transform.

**Other open work**, roughly in order of value:

- Turn on `ABSENT` so the model generates the atom *count* — the original point of
  the vocabulary, and the prerequisite for pocket conditioning. Needs a padded
  canvas, a policy for where absent atoms sit, and edge masking so they cannot
  perturb real atoms.
- Report `abs256lin` at epoch 5000 as the headline configuration; carry a run to
  epoch 10 000 to see whether the composition trend really keeps going.
- Try remasking on a late, good checkpoint, where it has far less to fix (4.4%
  bond-order and 7.0% radical failures rather than 7.6% and 12.5%) and may
  therefore cost much less composition.
- Pocket conditioning, reusing the existing frozen-scaffold machinery.

---

## 8. Practical traps, all of which cost us time

- **`sdd256` means different things for the two model families.** For the
  continuous models it is the *generative* config; for absorbing models it leaves
  `properties.Z` untouched and is the *types-given* config. The tell is at%N:
  5.66 means the types were copied, ~3.9 means they were generated. This produced
  one wrong conclusion before we noticed.
- **Validity alone is not a usable metric** once composition is generated. A model
  that drops heteroatoms scores better. Always report the element distribution
  beside it.
- **Two QM9 databases.** The `novelty` metric silently fails on the paper database
  if `ASSETS` points at the old 121 651-row SMILES list. Going through `hy eval`
  sets it correctly; submitting the evaluation script by hand does not.
- **`MASK` and `ABSENT` leak into the output database** for molecules whose
  positions diverge: the sampler deliberately lets them converge so they stop
  burning function evaluations, and they freeze with undecided types, which are
  then written out as einsteinium and as ASE's dummy element. Harmless at 0.01%
  and never in a valid molecule, but it pollutes all-molecule statistics and
  should be dropped at write time.
- **One non-finite atom poisons a mean.** A handful of molecules diverge at
  σ≈30; averaging without dropping them turns an entire diagnostic table into
  NaN, which happened once and cost a rerun.
