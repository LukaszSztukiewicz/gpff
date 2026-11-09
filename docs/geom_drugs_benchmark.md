# A GEOM-Drugs evaluation suite for GPFF

**Status: proposal.** Nothing here is implemented yet. This document specifies (1) how to
prepare GEOM-Drugs, (2) which metrics to report and why each one is informative, and (3) the
protocol rules that make cross-model numbers comparable. Model-internal diagnostics are in
[`gpff_diagnostics.md`](gpff_diagnostics.md).

The design goal is a **small benchmark that is hard to game**: every metric either has a known
ceiling (the training data itself), a calibrated ruler (a noise ladder), or a decomposition that
says *which* part of chemistry broke. Metrics that only go up when a model gets better at one
thing, and that can be moved by changing an evaluation flag, are explicitly rejected.

---

## 0. Why the current QM9 evaluation does not transfer

`gpff_prepare_metadata` / `gpff_calculate_metrics` currently report: RDKit validity, PoseBusters
validity (one boolean), uniqueness, novelty, mean Tanimoto similarity, and Jensen–Shannon
divergences on predicted `energy_U0`, predicted `gap`, and `elongation`. On QM9 that is a
reasonable suite. On GEOM-Drugs each piece breaks in a specific way, and the breakages drive the
design below.

| Current component | What breaks on GEOM-Drugs |
| --- | --- |
| `analyze_with_rdkit` → `rdDetermineBonds.DetermineBonds(charge=0, allowChargedFragments=False)` | This is xyz2mol. It is tuned for small neutral organics. On 40–180-atom drug-like molecules with hypervalent S/P, halogens and charged groups it fails often and slowly (the 5 s `SIGALRM` will fire). Worse: it is used *both* as the dataset filter (`build_filtered_qm9`) and as the validity metric, so a perception failure is indistinguishable from a geometry failure. |
| one boolean `posebusters_valid` | Collapses ~8 independent physics checks into one number. "73% PB-valid" does not say whether rings are non-planar, bonds are long, or atoms clash — i.e. it gives no gradient to the researcher. |
| `js_divergence_energy_U0`, `js_divergence_gap` | Need QM9-trained property predictors (`assets/best_model_gap`, `best_model_energy_U0`). No equivalent exists for GEOM-Drugs, and a surrogate predictor's error would be folded into the metric. |
| `js_divergence_elongation` | Max pairwise distance is a crude global shape proxy. It is nearly blind to local geometry, which is what fails on drug-like molecules. |
| `calculate_js_divergence` | Bins on the *reference* range with under/overflow bins. A badly scaled model dumps everything into overflow and the metric saturates at 1, losing all resolution exactly where you need it. JSD is also bin-dependent and unitless. |
| `novelty` / `uniqueness` on isomeric SMILES | Graph-level only. A model that reproduces training *conformers* of novel graphs scores perfectly. There is no 3D memorization check. |
| no reference row, no CIs, no stratification | GEOM-Drugs validity is strongly size-dependent, so an unstratified mean is partly a molecule-size statistic. Without a data reference row there is no ceiling to compare against. |

Three design principles follow:

1. **Decouple bond perception from geometry quality.** GPFF emits coordinates only; bond
   perception is a post-hoc front-end and therefore a confound. It must be a *named, shared,
   reported* component, not an implementation detail.
2. **Use GFN2-xTB as the reference level, never MMFF.** GEOM-Drugs conformers are GFN2-xTB
   optimised. Under MMFF94 the *data itself* shows a mean relaxation energy of ≈16 kcal/mol,
   versus ≈0 under GFN2-xTB ([Nikitin et al.][geomrev]). Scoring against MMFF makes the ground
   truth look strained — a benchmark bug, not a model result.
3. **Every table carries a ceiling row and a ruler.** Test-split conformers (the ceiling), an
   MMFF-relaxed row (the classical-force-field bar), and a noise ladder (§3.6) that converts any
   metric value into "equivalent coordinate noise in Å".

---

## 1. Where GPFF sits in the taxonomy (and why the protocol must say so)

The de-novo 3D generation literature has two model classes, and comparing across them without
saying so is the single most common unfairness in current tables:

* **Joint graph+coordinate models** — MiDi, EQGAT-Diff, SemlaFlow, Megalodon, FlowMol/FlowMol3,
  JODO. They *generate* atom types, formal charges and bond orders alongside coordinates. Their
  "validity" is a property of a generated graph that happens to carry coordinates.
* **Coordinate-only models** — EDM, voxel models, and **GPFF**. Atom types are given (GPFF takes
  the atom multiset of a test molecule, see `gpff_generate`: it walks `test_dataloader()` and the
  prior overwrites positions only). Bonds must be perceived afterwards.

The gap is large: EDM + OpenBabel reaches 0.363 connected validity on GEOM-Drugs where MiDi
reaches 0.705 and Megalodon-flow 0.948 ([Megalodon][megalodon] Table 1). A naive table placing
GPFF next to those numbers measures the bond-perception front-end as much as the model.

**Protocol rule P1 — common perception front-end.** Every model is evaluated through the *same*
bond-perception step applied to coordinates + elements, with as-generated bonds discarded. Report
two columns where applicable:

| column | meaning |
| --- | --- |
| `as-generated` | the model's own bond orders (N/A for GPFF) |
| `perceived` | the shared front-end's bonds, for every model |

The `perceived` column is the apples-to-apples column and should be the headline. The
`as-generated` column is kept so that joint models can be cited against their own papers. Making
this explicit is, by itself, a contribution: it is the only way a coordinate-only model like GPFF
can be compared faithfully.

**Protocol rule P2 — matched composition.** GPFF conditions on a real molecule's atom multiset;
most baselines sample the atom count from the training histogram and then generate types. These
are different tasks. Fix a single list of 5,000 target compositions drawn from distinct test-split
molecules (`composition_list.json`, shipped with the benchmark). Then:

* models that can condition on composition (GPFF) consume that list verbatim;
* models that cannot sample `N` from the empirical `N` histogram of that same list;
* GPFF additionally reports an **unconditional variant** where compositions are drawn from the
  *training* formula distribution rather than the test one, because conditioning on a held-out
  molecule's exact formula is a real advantage and must not be hidden.

Every row in every table is tagged `cond=formula` or `cond=N`.

---

## 2. Dataset preparation

### 2.1 Which GEOM-Drugs

Use the **MiDi/Vignac split** of GEOM-Drugs (the de-novo-generation split, downloadable from
`https://bits.csb.pitt.edu/files/geom_raw/`), then apply the [GEOM-Drugs Revisited][geomrev]
corrections on top. Rationale: SemlaFlow, EQGAT-Diff, FlowMol, Megalodon and GEOM-Drugs Revisited
all use it, so the numbers compose with the literature.

> Do **not** use the GeoMol / Torsional-Diffusion split (243,473 / 30,433 / 1,000 molecules). That
> is the *conformer-generation* split and its test set is 1,000 molecules — far too small for the
> 5×1,000 protocol below, and not what the de-novo papers report.

### 2.2 The pipeline, in order

Each step is a filter with a logged count. The counts are part of the benchmark: a reader must be
able to see exactly how much of GEOM-Drugs a given track covers.

**Step 1 — sanitize and kekulize.**
`Chem.SanitizeMol(mol, SANITIZE_ALL)` then `Chem.Kekulize(mol, clearAromaticFlags=True)`.
Kekulizing is not cosmetic: it is what makes valency well-defined. Aromatic bonds have bond order
1.5, and the widely copied MiDi valency code rounds that to 1, which is why the legacy tables
permit neutral carbon at valency 3 and neutral nitrogen at valency 2. That bug propagated into
EQGAT-Diff, SemlaFlow, Megalodon and FlowMol through code reuse ([GEOM-Drugs Revisited][geomrev]).
Retraining on kekulized data raised validity by ≈5% for 4 of 6 models in that paper, so this is a
*data* decision with a real effect, not only an evaluation one.

**Step 2 — single fragment.** Reject `len(Chem.GetMolFrags(mol)) > 1`. This removes the molecules
that rare GFN2-xTB optimisation failures fractured (≈0.18% of the dataset, 272 molecules in the
reference analysis). These are the entries that contaminated the legacy valency tables with
nonsense like neutral C at valency 2 and H at valency 0.

**Step 3 — topology validation.** For every conformer, for every bonded pair, require

```
| d_ij − (r_i + r_j) |  ≤  tolerance · (r_i + r_j),      tolerance = 0.4
```

with `r` the covalent radius (H 0.31, C 0.76, N 0.71, O 0.66, F 0.57, P 1.07, S 1.05, Cl 1.02,
Br 1.20, I 1.39 Å). Reject the molecule if *any* conformer fails. This catches poor conformers and
impossible geometries that sanitisation lets through.

**Step 4 — element alphabet.** Restrict to `{H, C, N, O, F, P, S, Cl, Br, I}` (FlowMol's
`atom_map`). Drop B, Al, Si, As, Se, Hg, Bi: counts are negligible, they break xyz2mol and the
valency tables, and keeping them makes every element-stratified statistic noisy. Log the drop
count per element.

**Step 5 — charge policy. This is the decision that most affects GPFF, and it needs two tracks.**

GPFF has no formal-charge channel. Its head emits a 3-vector per atom and nothing else; its
validity check calls `DetermineBonds(charge=0, allowChargedFragments=False)`. So GPFF cannot
*represent* a zwitterion, let alone be scored on one. Two tracks, both reported:

* **`drugs-neutral` (primary for GPFF today).** Keep molecules in which **every atom has formal
  charge 0**. Note the precision: this is strictly stronger than net charge 0 — a zwitterionic
  amino acid has net charge 0 but two charged atoms, and xyz2mol with
  `allowChargedFragments=False` will reject it. Stating "we dropped charged molecules" without
  this distinction is the kind of ambiguity that makes benchmarks irreproducible.
* **`drugs-full` (secondary, for when a charge head exists).** Keep everything that survived steps
  1–4, charges included, and model formal charge as an extra output. This is what MiDi, SemlaFlow,
  FlowMol and Megalodon do.

Report `|drugs-neutral| / |drugs-full|` prominently. If the neutral track covers, say, 85% of
drug-like space, then "GPFF matches FlowMol3 on drugs-neutral" is an honest claim with visible
fine print; without the ratio it is not.

**Step 6 — do *not* re-relax the data.** GEOM-Drugs conformers are already GFN2-xTB optimised.
Re-relaxing would be wasted compute and would silently shift the reference distribution. What is
needed instead is a **reference-row re-relaxation on a 5,000-conformer test subset only**, run
through the exact same xTB driver the models are scored with, so that the data row's ΔE_relax and
RMSD are ≈0 *by construction of the same code path* rather than by assumption. This is how you
prove the harness is not biased.

**Step 7 — conformers per molecule.** Cap at `K` conformers per molecule and state `K`. This is
not a detail for GPFF: the denoising target is `E[x_0 | x_t]`, a *posterior mean over the
conformers compatible with* `x_t`. The conformer multiplicity therefore changes the regression
target itself. Recommend `K = 5` as the default (matching FlowMol's `geom_5_kekulized`) with
`K = 30` as a secondary run, and report both — the difference is a measurement of how much
conformational averaging the model is being asked to absorb.

**Step 8 — splits.** **Molecule-level, never conformer-level.** All conformers of a molecule go
into one split. `SeededRandomSplit` in `gpff/data/splitting.py` permutes *dataset rows*; on a
multi-conformer database that leaks the same molecule into train and test and inflates every
novelty number. A `GEOMSplit` that permutes molecule ids and then expands to rows is required.

Ship two splits:

* `random-molecule` (80/10/10 by molecule) — the comparable split.
* `scaffold-disjoint` (Bemis–Murcko, greedy largest-scaffold-first into train) — the
  generalization split. The *gap* between the two is the generalization number (§3.7).

**Step 9 — no size cap, but stratify.** SemlaFlow and Megalodon-flow were trained on ≤72 atoms
while Megalodon was not, and GEOM-Drugs Revisited flags this as a comparability trap because
validity is strongly size-dependent. So: keep every size, **stratify every metric by heavy-atom
count** in bands (≤20, 21–30, 31–40, 41–55, 56–72, >72), and additionally publish a ≤72-atom
subset so numbers can be cited against the papers that used it.

### 2.3 GPFF-specific configuration that the data forces

Three `globals` in `configs/experiment/gpff.yaml` are QM9 constants and must be rederived:

| global | QM9 value | why it changes |
| --- | --- | --- |
| `sigma_max` | 30 | Chosen relative to QM9's ~7.4 Å molecular radius. GEOM-Drugs molecules reach ~181 atoms, so the radius-of-gyration distribution is far wider. Set `sigma_max` from the data: large enough that `N(0, sigma_max²)` is indistinguishable from the prior-limit of the forward process for the *largest* molecules. Report the chosen value and the radius distribution it came from. |
| `cutoff` | `5 · sigma_max` = 150 | Only meaningful as "comfortably larger than the noisiest molecule". Recompute from the new `sigma_max`. |
| `n_rbf` | `4 · cutoff` = 600 | **This is a latent bug at GEOM-Drugs precision.** `GaussianRBF(n_rbf=600, cutoff=150)` places centres 150/599 ≈ **0.25 Å** apart with matching widths. The xTB benchmark asks for bond-length agreement at 0.006–0.03 Å. A fixed uniform basis with 0.25 Å resolution cannot encode that. `gpff.nn.radial.ExpGaussianRBF` already exists and concentrates resolution at short range — it is the obvious fix. See [`gpff_diagnostics.md` §D4.4](gpff_diagnostics.md) for the audit that quantifies this before you change anything. |

Also: the all-to-all neighbour list (`AllToAllNeighborList`, fully connected) at 181 atoms is
32,580 directed pairs per molecule, 16,290 after PaiNN's half-pair optimisation. The filter net is
`Dense(n_rbf, n_interactions · n_atom_basis · 3)` = `600 → 3072`, so the filter tensor alone is
`16290 × 3072 × 4 B ≈ 0.20 GB` **per molecule** in float32 — versus ~1.9 MB for an 18-atom QM9
molecule, a factor of ~100. Batch size, `n_atom_basis` and `n_rbf` therefore have to be chosen
together. Record the resulting memory and throughput in the benchmark's cost table (§3.8): a model
that only fits at batch size 4 is a different research object than one that fits at 128.

### 2.4 Reference assets to build

By analogy with the QM9 `assets/`, `gpff_reference_assets` must produce, from the *cleaned train
and test splits*:

```
drugs-<track>_valency_tuple.json        aromatic-aware valency table, built from THIS train split
drugs-<track>_smiles_train_val.txt      novelty reference (one canonical SMILES per molecule)
drugs-<track>_scaffolds_train_val.txt   Bemis-Murcko scaffold novelty reference
drugs-<track>_n_atoms_histogram.pt      size histogram, for baselines that sample N
drugs-<track>_composition_list.json     the 5,000 shared target compositions (P2)
drugs-<track>_geom_reference.npz        per-type bond-length / angle / torsion reference samples
drugs-<track>_ring_reference.npz        Cremer-Pople puckering reference distributions
drugs-<track>_reos_train.pkl            REOS flag rates on train, for the deviation metric
drugs-<track>_xtb_reference.sdf         the 5,000-conformer ceiling row (pre/post relaxation)
drugs-<track>_noise_ladder/             the calibration ladder (§3.6)
```

Build the valency table from your own cleaned split rather than importing one. The published
tables exist precisely because imported tables were wrong.

---

## 3. The metric suite

Six tiers. Each entry gives the definition, what it diagnoses, and the failure it is designed to
catch. Tiers A–B are the core; C–E are what turn a table into a benchmark.

### 3.1 Tier A — Can the molecule be read at all? (granular, decomposed)

**A1. Validity cascade.** Replace the single `try/except` in `analyze_with_rdkit` with
**mutually exclusive, ordered failure buckets whose fractions sum to 1**:

| bucket | trigger |
| --- | --- |
| `perception_error` | `DetermineBonds` raised |
| `perception_timeout` | the `SIGALRM` fired |
| `fragmented` | `len(GetMolFrags) > 1` |
| `radical` | any atom with `GetNumRadicalElectrons() > 0` |
| `valence_invalid` | `AtomValenceException` on sanitize |
| `kekulize_failed` | `KekulizeException` |
| `sanitize_other` | any other `MolSanitizeException` |
| `valid` | everything passed |

Why this and not one number: these buckets point at *different* fixes. `fragmented` means the
model is scattering atoms — a prior/σ-schedule problem. `valence_invalid` means local geometry
implies the wrong bond multiplicity — a short-range-resolution problem. `perception_timeout` means
the harness is the bottleneck, not the model. FlowMol already does a coarse version of this
(`disconnected / valence / kekulization / other / valid`); making it exhaustive and exclusive is
cheap and makes the metric diagnostic rather than merely descriptive.

**A2. Perception-backend sensitivity.** Run A1 under (i) `rdDetermineBonds` (xyz2mol, primary),
(ii) OpenBabel, (iii) a distance + valency-lookup builder (MiDi-style). Report the primary number
and the spread. GEOM-Drugs Revisited warns that "reliance on RDKit-sanitized molecules can hide
raw model errors"; a model whose validity moves 15 points when you swap the perceiver has not
been measured, it has been guessed at. For `drugs-full`, pass the *known* net charge of the
reference composition to `DetermineBonds` — GPFF conditions on a real molecule's atom multiset, so
that charge is legitimately available, and it makes perception far more reliable.

**A3. Connected validity (V&C).** Fraction both sanitizable and single-component. One number, for
citation against existing tables.

**A4. Valency stability, corrected.** Atom stability = fraction of atoms whose valency tuple is in
the table; molecule stability = fraction of molecules where every atom is. Use the
**aromatic-aware tuple key**

```
(element, formal_charge)  →  set of (n_aromatic_bonds, valence_from_non_aromatic_bonds)
```

built from your own cleaned train split. Count aromatic bonds separately instead of summing 1.5s;
that is the whole point of the correction. Report the legacy-table number *once*, in an appendix
column, so readers can map onto pre-2025 papers — the correction costs 1–3 points of molecule
stability, and silently reporting only the corrected number looks like a regression.

### 3.2 Tier B.1 — Geometry against the real PES (the heart of the benchmark)

This is where "did it learn chemistry" is actually decided. Everything in this subsection is
measured by relaxing each generated structure with **GFN2-xTB** (`xtb <in.xyz> --opt --charge
<q> --namespace <prefix>`) and comparing before/after. Computed over the molecules that passed A1
(`valid`), and the valid fraction is reported immediately adjacent — never alone.

**B1. Relaxation energy.**

```
ΔE_relax = E(x_gen) − E(x_gen → xTB-relaxed)        [kcal/mol]
```

Report **median and mean** (the distribution is heavy-tailed, so the mean is a different
statistic, not a noisier version of the median), the full CDF as a figure, and the exceedance
fractions `P(ΔE > 10)`, `P(ΔE > 50)` kcal/mol. Exceedances are what matter downstream: a ligand
200 kcal/mol above its own minimum is unusable regardless of the median.

**B2. Per-atom relaxation energy — `ΔE_relax / N`.** *An addition to the published protocol, and
a necessary one.* ΔE_relax grows with molecule size, so on GEOM-Drugs (20–181 atoms) a raw mean is
partly a size statistic, and a model that quietly prefers small molecules wins. Normalising makes
the number a property of geometric quality. Report both; if they rank models differently, say so —
that disagreement is itself a finding.

**B3. Bond length, angle and torsion deviation.** Per geometric feature, with periodicity handled:

```
Δr_ij   = | r_init − r_opt |                                           [Å]
Δθ_ijk  = min( |θ_init − θ_opt| , 180° − |θ_init − θ_opt| )             [deg]
Δφ_ijkl = min( |φ_init − φ_opt| , 360° − |φ_init − φ_opt| )             [deg]
```

Aggregate as a **type-weighted mean** over canonical keys — `(Z_i, bond, Z_j)` for lengths,
`(Z_i, b_1, Z_j, b_2, Z_k)` for angles, `(Z_i, b_1, Z_j, b_2, Z_k, b_3, Z_l)` for torsions, each
canonicalised as `min(key, reversed(key))`. Torsions are enumerated around bonds matching
`[!$(*#*)&!D1]~[!$(*#*)&!D1]`.

Why all three and not a single RMSD: they separate **stiff** from **soft** coordinates. Bond
lengths and angles are stiff — error there means the short-range representation is wrong. Torsions
are soft — error there means the model got local chemistry right but conformational preference
wrong. Those need different fixes, and `gpff_diagnostics.md` §D2 ties this split directly to the
slow-mode spectrum of the learned field.

**B4. Self-RMSD to own minimum.** RMSD between the generated pose and *its own* xTB-relaxed pose
(heavy-atom and all-atom), median and mean, plus `P(RMSD < 0.25 / 0.5 / 1.0 Å)`. This is the most
directly interpretable single number in the suite: *how far is the generated structure from a true
local minimum of the real potential energy surface.* Use the same aligned-RMSD convention
throughout (`AllChem.AlignMol` between the two conformers of one molecule).

**B5. The three mandatory reference rows.**

| row | how | what it means |
| --- | --- | --- |
| `data (test)` | 5,000 held-out test conformers through the identical pipeline | the ceiling: ΔE_relax ≈ 0, RMSD ≈ 0. If it is not ≈0, your harness is broken, not the model. |
| `MMFF → xTB` | MMFF94-relax the test conformers, then score them | the classical-force-field bar (≈11 kcal/mol mean ΔE_relax in the reference study). Beating it is a genuine, quotable claim. |
| noise ladder | §3.6 | converts any ΔE_relax / RMSD into "equivalent Å of coordinate noise" |

### 3.3 Tier B.2 — Distributional geometry (the orthogonal axis)

B1–B4 measure *per-molecule physical error*. They are blind to a model that produces 5,000
individually-relaxed-looking molecules all with the same bond lengths. The distribution metrics
cover that, and they are not redundant with the relaxation metrics.

**B6. W1 on geometric features — primary; JSD — secondary.** Per feature class,

```
W_feature  =  Σ_y  p(y) · W1( D̂_y , D_y )
```

where `y` ranges over canonical type keys, `p(y)` is the type's frequency in the reference, `D̂_y`
is the generated distribution for that type and `D_y` the **test-split** distribution. This is
Megalodon's definition, so the numbers compose.

Report W1 as the headline and JSD in an appendix column. The reason is concrete: `calculate_js_divergence`
bins on the reference range with under/overflow bins, so a model whose bond lengths are
systematically off dumps mass into overflow and JSD saturates near its maximum, after which it
cannot distinguish "slightly wrong" from "catastrophic". W1 keeps resolving, carries units (Å,
degrees) and needs no bin choice. Keep JSD only for backward citation.

**Compute W1 on the raw generated geometry, never the relaxed one** — relaxing first measures xTB,
not the model.

**B7. Named torsion classes — the sharpest "real chemistry" probe.** Aggregate torsion W1 is a
statistic; specific torsion classes are chemistry. Report W1 separately for:

| class | what a wrong distribution proves |
| --- | --- |
| amide ω (`C(=O)-N`) | should be sharply bimodal at 0°/180°. A broad ω means the model has not learned conjugation-induced planarity. |
| sp³–sp³ C–C | should be trimodal (anti ≈180°, gauche ≈±60°). A unimodal or flat profile means the model is emitting the *average* of conformers instead of a conformer — the classic posterior-mean collapse, and exactly the failure a denoiser trained with heavy conformational averaging is prone to (see §2.2 step 7). |
| biaryl | characteristic twist, driven by ortho sterics |
| ester / conjugated | planarity |
| ring torsions (separately from acyclic) | ring closure is a hard geometric constraint; mixing ring and acyclic torsions into one average hides which one failed |

This split is cheap and it is the single most informative block for "did it learn chemistry rather
than statistics". Recommend making it a figure, not just a table row.

**B8. Ring geometry and puckering.** Mostly absent from this literature, and highly
discriminative:

* **Aromatic planarity**: distribution of max out-of-plane distance for 5- and 6-membered
  aromatic rings (PoseBusters uses a 0.25 Å pass/fail; report the *distribution*, which is
  strictly more informative, and the pass rate for compatibility).
* **Cremer–Pople puckering** for saturated 6-rings — amplitude `Q` and angles `(θ, φ)`. The data
  shows a sharp chair population with minor boat/twist-boat. W1 on the `Q` distribution and the
  chair/boat/twist population ratios test whether the model learned a real conformational
  preference or just "roughly ring-shaped". For 5-rings, the pseudorotation `(q, φ)` envelope.
* **Ring-size histogram** vs training (TV distance), which catches invented 3- and 4-rings — the
  same quantity `analyze_with_rdkit` already computes as `ring_counts`, so this is nearly free.

**B9. Stereochemistry — and a structural prediction worth testing.** Report: fraction of molecules
with assignable CIP tags, the R/S ratio at generated stereocentres, and the E/Z ratio on double
bonds.

There is a specific reason to make this a headline metric for GPFF. Vanilla PaiNN builds its
invariants from distances and from scalar products of vector features (`dmuR * dir_ij +
dmumu * muj`, then `‖μ‖` and `⟨μ, Wμ⟩` in the mixing block). Every one of those is parity-even, so
PaiNN is **O(3)-equivariant, not merely SO(3)-equivariant**: for a reflection `P`,
`F(Px) = P F(x)` exactly. A model with that symmetry *cannot* assign different energies to
enantiomers, so it cannot produce an enantiomeric excess and cannot learn chirality-dependent
conformational preference. QM9 barely exposes this (few stereocentres); GEOM-Drugs is
stereochemistry-rich, so it becomes a first-order limitation. The metric is the measurement; the
corresponding model-side check (`F(Px) == P F(x)` to numerical precision) is in
`gpff_diagnostics.md` §D6.2, and the fix direction is parity-odd features (cross products, higher
irreps, or pseudoscalar inputs) — which is precisely what Megalodon's "EGNN + cross product"
baseline is about.

### 3.4 Tier C — Distribution-level chemistry

Validity-conditioned, graph-level. These catch "every molecule is valid and they are all the same
molecule".

| metric | definition | catches |
| --- | --- | --- |
| **uniqueness** | distinct canonical SMILES / valid | mode collapse. Note GPFF's composition conditioning: also report **distinct formulas sampled**, since uniqueness is partly inherited from the composition list. |
| **novelty** | valid SMILES not in train+val | memorization (graph level only — see Tier D) |
| **scaffold novelty** | Bemis–Murcko scaffolds not in train+val | the meaningful novelty for drug-like space; molecule-level novelty is nearly free when formulas are large |
| **FCD** | Fréchet ChemNet Distance to the test set | the standard 2D distribution metric; sensitive to collapse that validity and uniqueness both miss |
| **internal diversity** | 1 − mean pairwise Tanimoto | breadth. **Switch from the current RDKit path fingerprint (`maxPath=5`) to Morgan/ECFP4** for comparability with MOSES/GuacaMol. |
| **REOS flag rate** | fraction flagged by Glaxo + Dundee structural alerts, plus cumulative deviation vs train | "valid but chemically absurd" functional groups |
| **ring-system OOD rate** | fraction of generated ring systems with zero ChEMBL occurrences | invented ring systems that pass every valency check |
| **marginal fidelity** | TV / W1 between generated and training distributions of atom types, bond orders, formal charges, ring sizes, functional groups | **the validity↔diversity tradeoff.** FlowMol3 observed that tuning geometry distortion to maximise validity *collapses functional-group diversity*. This row is what makes that visible; without it, the validity column can be gamed. |

### 3.5 Tier D — Generalization and memorization

Usually omitted. It is where the interesting claims live.

**D1. 3D conformer-copy rate.** For each generated molecule whose canonical SMILES matches a
training molecule, compute the minimum RMSD to that molecule's *training conformers*. Report the
distribution and `P(RMSD < 0.25 Å)`. A memorizing model reproduces training conformers; a
generalizing model finds new minima of the same molecule. Graph-level novelty cannot see this
distinction at all, and for a model trained on multi-conformer data it is the relevant one.

**D2. Graph-copy rate.** Fraction of generated canonical SMILES present in train (the complement
of novelty, reported as a rate so it reads as a risk).

**D3. Scaffold-split transfer.** Run all of Tier A and Tier B on the `scaffold-disjoint` split and
report the **gap** against `random-molecule`. One number per metric. This is the generalization
result; everything else in Tier D is supporting detail.

**D4. Size extrapolation.** Sample at atom counts in held-out size bands (and beyond the training
maximum), and plot validity, ΔE_relax/N and self-RMSD against `N`. This directly exercises the
failure mode that `prenorm`, `degree_norm` and `bounded_forces` were added to
`configs/experiment/gpff.yaml` for: with an all-to-all neighbour list, an interaction block's
summed message grows with molecule size, so features that are fine at QM9's 4–28 neighbours can
diverge at GEOM-Drugs' ~180. The curve says whether those options are doing their job, and the
matching internal audit is `gpff_diagnostics.md` §D4.3.

**D5. Composition extrapolation.** A held-out set of formulas absent from train, scored on Tier A.

### 3.6 The noise ladder — a calibrated ruler

*An addition, and the thing that makes the whole suite interpretable.* For each of the 5,000
reference test conformers, form

```
x_δ = x_0 + δ · ε ,      ε ~ N(0, I) centred,      δ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2} Å
```

and push `x_δ` through the *entire* Tier A + Tier B pipeline. This gives a monotone family of rows
in the same table as the models. Near a minimum `ΔE ≈ ½ δ² tr(H)`, so the ladder is smooth and
quadratic and can be interpolated.

Three things it buys:

1. **A unit.** "ΔE_relax = 6 kcal/mol" becomes "equivalent to ≈0.02 Å of coordinate noise" — a
   quantity a chemist can judge without knowing the benchmark.
2. **A perception-brittleness measurement.** The validity-vs-δ curve says how much of a model's
   invalidity is genuinely bad geometry versus a perception front-end that is itself fragile to
   small displacements. If validity drops to 60% at δ = 0.05 Å, then no model can score above
   that at that geometric accuracy, and reporting raw validity without the ladder overstates how
   much headroom remains.
3. **A sanity check on the harness.** The ladder must be monotone in δ for every metric. If it is
   not, the metric is noise-dominated at the sample size used.

### 3.7 Tier E — Cost, and the curve that should be GPFF's headline

GPFF's claim is few-step direct denoising, so a single-NFE row understates it and a
best-quality-at-any-cost row misrepresents the competition.

* **NFE** — already tracked per molecule (`keys.nfe`, written into the samples DB). Report the
  **distribution**, not the mean: the adaptive samplers freeze molecules individually via `f_max`,
  so the mean hides a long tail.
* **Wall-clock per molecule** at fixed hardware and batch size, plus peak memory.
* **The Pareto frontier.** Sweep `n_steps ∈ {5, 10, 25, 50, 100, 250}` for each sampler
  (`direct_denoising`, `heun`, `stochastic_heun`, `ve_ddpm`, and the adaptive variants) and plot
  validity, ΔE_relax/N and self-RMSD against NFE. **This curve, not a single table row, is the
  result.** Baselines are then compared at *matched NFE*, which is the only fair axis when
  diffusion baselines run 500 steps and flow baselines 100.

### 3.8 Reporting rules

These are not decoration; they are what separates a benchmark from a results table.

1. **5 × 1,000 molecules**, mean ± 95% CI computed across the five subsets (the convention in
   GEOM-Drugs Revisited and FlowMol3). Additionally report **bootstrap CIs over molecules** for
   per-molecule metrics, which is the statistically correct interval; the 5-subset std is kept
   because prior work reports it.
2. **Every table carries** the `data (test)` row, the `MMFF → xTB` row, and at least δ = 0.01 and
   δ = 0.05 ladder rows.
3. **Every metric stratified** by heavy-atom band, with the unstratified number also given.
4. **Every row tagged** with `cond=formula|N`, the perception backend, the track
   (`neutral`/`full`), `K` (conformers per molecule), the split, `n_steps` and NFE.
5. **Validity and W1 are reported adjacent and read jointly, never separately.** W1 is conditioned
   on validity, so a model can improve W1 by becoming more selectively valid. The pair is the
   measurement.
6. **State which metrics are validity-conditioned** in the caption. Every Tier B and Tier C metric
   is.
7. **No composite score.** Publish a profile/radar instead. A scalar invites optimising the scalar,
   and the whole point of the granular decomposition is to say which thing to fix.
8. **Release artefacts**: generated SDFs, the xTB-relaxed SDFs, the asset files, the exact config,
   and the seeds. The benchmark is the artefacts; the table is a summary of them.

---

## 4. Implementation sketch in this repo

The existing two-phase design (`gpff_prepare_metadata` writes per-sample arrays →
`gpff_calculate_metrics` reduces them) is the right shape and should be kept: the expensive passes
stay cached and the cheap reductions stay re-runnable. The additions fit inside it.

```
src/gpff/
  data/datasets/geom.py          GEOMDrugs datamodule + the step 1-9 pipeline; GEOMSplit
                                 (molecule-level) and ScaffoldSplit
  analysis/
    structure.py                 + the A1 validity cascade (exclusive buckets, typed exceptions)
                                 + pluggable perception backends (xyz2mol / OpenBabel / lookup)
                                 + granular PoseBusters (per-check columns, not one boolean)
    stability.py                 NEW  aromatic-aware tuple valency table + atom/molecule stability
    geometry.py                  NEW  canonical type keys; per-type bond/angle/torsion extraction;
                                      named torsion classes; Cremer-Pople puckering
    relaxation.py                NEW  xTB driver (subprocess, --opt --charge --namespace),
                                      output parsing, MMFF comparison, paired RMSD
    distances.py                 NEW  W1 (type-weighted) and the existing JSD, side by side
    collection.py                + FCD, scaffold novelty, REOS, ring-system OOD, marginal fidelity
  cli/
    prepare_metadata.py          + targets: validity_cascade, stability, pb_checks, geom_features,
                                   xtb_energy, xtb_rmsd, scaffolds, reos, rings
    calculate_metrics.py         + the Tier A-E reductions, stratification, bootstrap CIs
    reference_assets.py          + the §2.4 asset builders (incl. the noise ladder)
    noise_ladder.py              NEW  builds and scores the §3.6 ladder
```

Two practical notes:

* **xTB is not a pip dependency.** It is a separate binary (`conda install -c conda-forge xtb`).
  It needs to be in the environment's setup script, and the benchmark should degrade gracefully
  (Tier A, B.2, C, D, E all run without it; only B.1 needs it). Budget it: ~5,000 geometry
  optimisations of 40–180-atom molecules is the dominant cost of the whole suite and should be
  parallelised the same way the structure passes already are (`ProcessPoolExecutor`, `spawn`).
* **The PoseBusters config matters and should be an explicit choice.** `get_posebusters_validity`
  currently uses `config="mol"`, which includes the `energy_ratio` check (UFF energy of the input
  versus the mean of a 50-conformer ETKDGv3 ensemble, threshold 100×). That check is slow, is the
  most frequent failure, and overlaps with B1 — which is why FlowMol's config disables it and adds
  non-aromatic ring *non*-flatness instead. Recommendation: report **per-check columns** from the
  full `mol` config, plus a `pb_valid_geometric` aggregate that excludes `energy_ratio` (because
  ΔE_relax measures that properly), and say so in the caption.

---

## 5. The headline table

One table, ~14 columns, every row tagged per §3.8:

```
row                      | valid | V&C | mol-stab | ΔE/N med | ΔE med | RMSD med | P(RMSD<.5) |
                           W1(r) | W1(θ) | W1(φ) | W1(ω amide) | pb-geom | FCD | NFE
-------------------------|-------|-----|----------|----------|--------|----------|------------|
data (test)              |       |     |          |   ~0     |  ~0    |   ~0     |            |
MMFF → xTB               |       |     |          |          |        |          |            |
ladder δ=0.01 Å          |       |     |          |          |        |          |            |
ladder δ=0.05 Å          |       |     |          |          |        |          |            |
-------------------------|-------|-----|----------|----------|--------|----------|------------|
GPFF  dd    n=10         |       |     |          |          |        |          |            |
GPFF  dd    n=50         |       |     |          |          |        |          |            |
GPFF  sheun n=50         |       |     |          |          |        |          |            |
DM baseline (eps) n=50   |       |     |          |          |        |          |            |
-------------------------|-------|-----|----------|----------|--------|----------|------------|
SemlaFlow   (perceived)  |       |     |          |          |        |          |            |
FlowMol3    (perceived)  |       |     |          |          |        |          |            |
Megalodon   (perceived)  |       |     |          |          |        |          |            |
```

with the Tier A cascade, Tier C, Tier D and the stratified breakdowns as supporting tables, and
three figures: the NFE Pareto frontier, the named-torsion profiles, and the ΔE_relax CDF with the
ladder overlaid.

---

## 6. Open questions to settle before freezing the benchmark

1. **`sigma_max` on GEOM-Drugs** — derive from the radius distribution, or keep a QM9-comparable
   value and accept that the largest molecules are under-noised? This changes the prior and
   therefore every number.
2. **Which perception backend is primary.** xyz2mol is the incumbent in this repo; the lookup
   builder is what MiDi-descended papers effectively use. Picking xyz2mol as primary is defensible
   but needs the A2 sensitivity column to back it.
3. **`K` (conformers per molecule).** `K` changes the denoising target, so it is a modelling
   choice masquerading as a data choice. It needs a short ablation (K = 1, 5, 30) before being
   fixed.
4. **Should `drugs-neutral` be the primary track?** It matches the current architecture, but if the
   neutral fraction is low the headline claim is narrow. Measure the fraction first, then decide.
5. **Whether to add the `energy_ratio` PoseBusters check to the headline** or rely on ΔE_relax.
   Recommendation: rely on ΔE_relax, report `energy_ratio` in the per-check table.

---

## References

* [GEOM-Drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation][geomrev]
  — Nikitin, Dunn, Koes, Isayev. arXiv:2505.00169; *Digital Discovery* (2025).
  Code: <https://github.com/isayevlab/geom-drugs-3dgen-evaluation>.
  Source of the kekulization/valency corrections, the fragment filter, the topology check and the
  GFN2-xTB geometry/energy benchmark.
* [FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation][flowmol3] — Dunn & Koes.
  arXiv:2508.12629; *Digital Discovery* 5, 2052 (2026). Code: <https://github.com/dunni3/FlowMol>.
  Source of the granular PoseBusters configuration, REOS structural alerts, ring-system OOD rate,
  and the validity↔functional-group-diversity tradeoff observation.
* [Applications of Modular Co-Design for De Novo 3D Molecule Generation (Megalodon)][megalodon]
  — arXiv:2505.18392. Source of the type-weighted W1 definition for angles and torsions and the
  baseline comparison table.
* [FLOWR: flow matching for structure-aware de novo, interaction- and fragment-based ligand
  generation][flowr] — *Nature Computational Science* (2026). Source for PB-validity + strain
  energy as a paired report, and the observation that strain metrics depend strongly on explicit
  hydrogens.
* [SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate
  Modeling][syncogen] — arXiv:2507.11818; ICLR 2026. Source for the paired W1/JSD report over the
  most prevalent bond lengths, angles and dihedrals.
* [PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to
  novel sequences][posebusters] — arXiv:2308.05777. Definitions of the individual checks,
  including the 100× UFF energy-ratio threshold and the 0.8× distance-geometry clash bound.
* [MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation][midi] — arXiv:2302.09048.
  Origin of the GEOM-Drugs de-novo split used here.

[geomrev]: https://arxiv.org/abs/2505.00169
[flowmol3]: https://arxiv.org/abs/2508.12629
[megalodon]: https://arxiv.org/abs/2505.18392
[flowr]: https://www.nature.com/articles/s43588-026-00998-8
[syncogen]: https://arxiv.org/abs/2507.11818
[posebusters]: https://arxiv.org/abs/2308.05777
[midi]: https://arxiv.org/abs/2302.09048
