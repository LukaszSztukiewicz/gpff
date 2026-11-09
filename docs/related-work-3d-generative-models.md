# Generative models for 3D molecules in Cartesian space: how atom types and atom count are produced

Survey compiled 2026-11-03. Scope: generative models that produce **3D molecular
structures in Cartesian coordinates** and that also have to decide **atom types**
and **how many atoms** there are — both unconditional/de-novo and
**pocket-conditioned** (structure-based) models. Models that are handed a 2D graph
and only produce a conformer (GeoDiff, Torsional Diffusion, MoreRed, …) are out of
scope, except where they appear as the second stage of a two-stage generator.

The survey is deliberately not restricted to absorbing-state / masked discrete
diffusion: it covers Gaussian diffusion, score matching, flow matching, Bayesian
flow networks, walk-jump / neural-empirical-Bayes, normalizing flows, VAEs on
density grids, neural fields and autoregressive models.

---

## 1. The two classification axes

### 1.1 Mechanism: who decides what, and when

| Code | Mechanism |
|------|-----------|
| **B1** | Geometry first with *anonymous* atoms, then infer atom types |
| **B2** | Geometry first, then infer atoms and/or bonds, then relax/refine a bit more |
| **B3** | Geometry alternated with atom types |
| **B4** | Joint diffusion of atom types + coordinates |
| **B5** | Joint diffusion of atom types + coordinates + bonds |
| **B6** | Formula / graph / fragment set first, then coordinates (and bonds) |
| **B7** | Atom types and count sampled up front (from a prior), geometry afterwards |
| **B8** | Something else (autoregressive, latent, field-valued, search-based, …) |

Two notes on how the buckets behave in practice:

* **B2 is the universal post-processing tail.** Almost every point-cloud model in
  B4 emits only coordinates + types and then calls OpenBabel/RDKit for bonds, and
  many evaluate after a force-field or GNINA minimisation. When a row is marked
  B4 (+B2) that is what is meant; a row marked **B2 as its primary bucket** is one
  where the *generative object itself* is not an atom list (a density grid or a
  field), so atoms, elements and bonds all have to be extracted afterwards.
* **B7 is likewise a modifier, not usually a standalone design.** Nearly all
  non-autoregressive models fix *N* before denoising. B7 is listed as a primary
  bucket only when the composition/count decision is the interesting part of the
  method.

### 1.2 Noise family

The requested four-way split, plus the honest caveat that two large families do
not fit it:

| Label | Definition |
|-------|-----------|
| **VP** | Variance preserving: `x_t = α_t x_0 + σ_t ε`, `α_t² + σ_t² = 1`, prior `N(0, I)` (DDPM/EDM/cosine/polynomial) |
| **pVP** | *Pseudo-VP*: linear interpolant `x_t = (1−t) x_noise + t x_0` against a **fixed-scale** Gaussian prior, with σ → 0 at the data end. OT-conditional-flow-matching / rectified flow. Variance is not literally preserved, but it is bounded and monotone, and the prior scale is fixed as in VP |
| **VE** | Variance exploding: data is **not** rescaled, `x_t = x_0 + σ_t ε` with σ_t growing; includes the degenerate single-σ case used by neural-empirical-Bayes / walk-jump |
| **Hybrid** | Two or more families in one model — either *across modalities* (Gaussian coordinates + categorical types) or *within the coordinate process* (VE and VP paths mixed) |

Two families that genuinely are none of the above and are flagged as such:

* **SL (straight line)** — `x_t = (1−t) x_0 + σ ε` with σ a **constant, small**
  value. The mean decays linearly, the noise never grows. SLDM states explicitly
  that this is neither VP (μ² + σ² ≠ const) nor VE (σ does not grow).
* **BFN** — Bayesian flow networks carry noise in an *accuracy/precision
  schedule* β(t) over distribution parameters, not in an observation-space noise
  level. GeoBFN, MolCRAFT, MolPilot, CByG, Pharmolix-FM.

And one component label used throughout:

* **DISC** — a discrete-state process on the categorical variables: `uniform`
  (multinomial/D3PM-uniform), `marginal` (transition to the training marginal),
  `absorbing`/`mask`, or `CTMC` (discrete flow matching). Whenever a model pairs
  Gaussian coordinates with DISC types, the row is **Hybrid** in the
  four-way sense — and that is the single most common answer in this field.

### 1.3 Diffusion state space

Orthogonal to the noise family, and the axis the field argues about hardest:

| Label | Definition |
|-------|-----------|
| **continuous** | Continuous-state throughout, including one-hot atom types *relaxed to continuous values* and noised under the same Gaussian kernel as the coordinates (EDM, DiffSBDD, JODO) |
| **continuous — geometry only** | The diffusion is continuous-state and covers **only the coordinates**. Atom types are never diffused: they are predicted afterwards (UniGEM, SLDM), searched over (SiMGen), fixed by an earlier stage (NExT-Mol) or supplied (GPFF) |
| **mixed — continuous + discrete** | Continuous coordinates plus a genuinely discrete process on types and/or bonds: `uniform`/multinomial (TargetDiff), `marginal` (MiDi, EQGAT-diff), `absorbing`/`mask` (MolDiff, FlowMol3) or a CTMC (SemlaFlow, discrete flow matching) |
| **continuous — parameter space** | Bayesian flow networks: categorical variables live in the *same* continuous parameter space as the coordinates, so nothing is diffused discretely at all (GeoBFN, MolCRAFT, MolPilot, CByG) |
| **N/A** | No diffusion process to classify — autoregressive, VAE or normalizing flow |

A separate, often-conflated distinction is **discrete-time vs continuous-time**.
Most Gaussian models are discrete-time (EDM T = 500, TargetDiff/MolDiff/Apo2Mol
T = 1000); continuous-time formulations are JODO (VP-SDE), EEGSDE, every
flow-matching model (ODE on `t ∈ [0,1]`), every CTMC discrete-flow model, all
BFNs, SLDM and GPFF. Continuous-*time* says nothing about the state space: a CTMC
is continuous-time and discrete-state at once.

---

## 2. De-novo 3D molecule generation

`†` = detail not re-verified against the primary text in this pass; treat as
likely-but-unconfirmed.

| Model | Venue | Bucket | Coordinates | Atom types | Bonds | Atom count N | VP/pVP/VE/Hybrid | Diffusion state space | Exact process |
|---|---|---|---|---|---|---|---|---|---|
| **G-SchNet** | NeurIPS 2019 | B8 (AR) | placed one atom at a time from predicted distance distributions | emitted with each atom | post hoc | stop token | — | N/A | autoregressive, no noise process |
| **cG-SchNet** | Nat. Commun. 2022 | B8 (AR) | as above, property-conditioned | with each atom | post hoc | stop token | — | N/A | autoregressive |
| **E-NF** | NeurIPS 2021 | B4 | equivariant CNF | one-hot, continuous | post hoc | `p(N)` empirical | pVP-like† | **continuous** | continuous normalizing flow to `N(0,I)` |
| **EDM** | ICML 2022 | B4 (+B2) | VP Gaussian, zero-CoM subspace | **continuous-relaxed** one-hot, scaled ×0.25 | OpenBabel / distance lookup | `p(N)` empirical categorical | **VP** | **continuous** | polynomial schedule `ᾱ_t = 1 − (t/T)²` |
| **EDM-Bridge** | NeurIPS 2022 | B4 | VP with informative prior bridge | continuous-relaxed | post hoc | `p(N)` | VP (non-standard prior) | **continuous** | diffusion bridge |
| **EEGSDE** | ICLR 2023 | B4 | VP SDE + energy guidance | continuous-relaxed | post hoc | `p(N)` | **VP** | **continuous** | equivariant energy-guided SDE |
| **GeoLDM** | ICML 2023 | B8 (latent) | VP in a latent point cloud (invariant scalars + equivariant vectors) | decoded from latents | post hoc | `p(N)` | **VP** | **continuous** | latent equivariant diffusion |
| **MDM** | AAAI 2023 | B4 | VP Gaussian, dual (local/global) edges | continuous-relaxed | post hoc | `p(N)` | **VP** | **continuous** | DDPM + interaction-aware edges |
| **HierDiff** | ICML 2023 | B6 → B2 | coarse **fragment-level** nodes diffused first | fragment identity at the coarse level, then atoms decoded | from fragment templates + iterative refinement | coarse node count | VP† | **continuous** † | coarse-to-fine hierarchical diffusion |
| **VoxMol** | NeurIPS 2023 | **B2** | **voxel atomic-density grid**, one channel per element | implicit in the channel, **extracted after** by thresholding + 3×3×3 max-filter peak finding, then L-BFGS position refinement | OpenBabel | **emerges from the grid** — never specified | **VE** (single level) | **continuous** | neural empirical Bayes, **fixed σ = 0.9**, walk (Langevin) + jump (one-step denoise) |
| **EquiFM** | NeurIPS 2023 | B4 | equivariant **optimal-transport** FM path | separate path, information-matched to the coordinate path | post hoc | `p(N)` | **Hybrid** (per-modality paths) | **continuous** | "hybrid probability transport" — the explicit per-modality-path design |
| **MiDi** | ECML 2023 | **B5** | Gaussian, zero-CoM, adaptive cosine `ν_r = 2.5` | **DISC marginal** `Q = α I + β 1 m'`, `ν_x = 1` | **diffused**, DISC marginal, `ν_y = 1.5` | `p(N)`† | **Hybrid** (VP + DISC) | **mixed** — continuous + discrete | per-modality exponents: coords & bonds denoise *earlier* than types & charges |
| **MolDiff** | ICML 2023 | **B5** (+B3 flavour) | VP Gaussian, sigmoid `ᾱ` | **DISC absorbing** (extra absorbing type) | **diffused**, DISC absorbing, **two-stage schedule** (bonds destroyed in steps 1–600, atoms denoised in 600–1000) | `p(N)` | **Hybrid** (VP + DISC) | **mixed** — continuous + discrete | deliberately staggered atom/bond schedules to fix atom–bond inconsistency |
| **JODO** | arXiv 2023 / TPAMI | **B5** | **VP-SDE** | continuous-relaxed | **diffused** as a continuous adjacency tensor | `p(N)` | **VP** | **continuous** | joint 2D+3D VP-SDE, Diffusion Graph Transformer |
| **MUDiff** | LoG 2023 | B5 | Gaussian | DISC† | DISC† | `p(N)`† | Hybrid† | **mixed** — continuous + discrete † | unified 2D graph + 3D geometry diffusion |
| **GeoBFN** | ICLR 2024 | B4 (parameter space) | continuous BFN variable, zero-CoM | **discretised** variable (the released design carries type through a discretised charge; fully discrete types are an ablation and are *worse*) | post hoc | `p(N)` | **neither — BFN** | **continuous** — parameter space | accuracy schedule β(t) on distribution parameters |
| **EQGAT-diff** | ICLR 2024 | **B5** | Gaussian, **explicitly called VP**, adaptive cosine `ᾱ_t = cos²(π/2 ((t/T+s)^ν/(1+s)))`, `ν_r = 2.5` | **DISC categorical** `U_t = α_t I + (1−α_t) 1 c̃'`, x₀-parameterisation | **diffused**, DISC, `ν_y = 1.5` | `p(N)`† | **Hybrid** (VP + DISC) | **mixed** — continuous + discrete | the canonical design-space study: **discrete beats continuous** (98.68 vs 96.45 mol-stability QM9), x₀ beats ε, SNR-clipped loss weighting |
| **GFMDiff** | AAAI 2024 | B4† | VP† | continuous-relaxed† | post hoc | `p(N)`† | VP† | **continuous** † | geometry-facilitated denoising |
| **END** | NeurIPS 2024 | B4 | **learned** forward process (time- and data-dependent equivariant transformation) | jointly | post hoc | `p(N)`† | VP-generalisation (learned) | **continuous** | learnable diffusion instead of a hand-set schedule |
| **FuncMol** | NeurIPS 2024 | **B2** | **neural field** (occupancy) compressed to a per-molecule modulation code; sampling happens in code space | decoded from the field | post hoc | **emerges from the field** | **VE** (single level) | **continuous** | walk-jump in latent code space, one-step jump |
| **SiMGen** | Nat. Commun. 2025 | **B1/B8** | score from a **pretrained MACE force field + time-dependent similarity kernel**; shape control via point-cloud priors | **element identities by an evolutionary swap**, not by a learned head | post hoc | prior point cloud | VE-like annealing | **continuous** — geometry only | training-free; the closest published cousin of a pseudo-force-field sampler |
| **FMG** | ICLR 2025 | **B2** | **4D multi-channel field** of isotropic RBFs (atoms *and* bonds as channels), U-Net denoiser, reference rotations instead of E(3) equivariance → **learns chirality** | channel identity, recovered by deterministic peak extraction | **field channel**, recovered by iterative bond optimisation | emerges from the field | VP† | **continuous** | field-valued diffusion; 98.8% validity on QM9 |
| **GeoRCG** | ICML 2025 | **B6** (semantic) | second stage, from the base model (EDM or SemlaFlow) | second stage | base model | base model | inherits base | **continuous** † | stage 1 generates a **geometric representation**, stage 2 generates the molecule conditioned on it; ~100 steps instead of 1000 |
| **ADiT** | ICML 2025 | B8 (latent) | **linear-interpolant** flow/diffusion over per-atom VAE latents | argmax of a linear head on the decoder | RDKit | fixed latent set size† | **pVP** | **continuous** | DiT on a joint molecule+crystal latent space |
| **NExT-Mol** | ICLR 2025 | **B6** | second stage: DMT conformer diffusion | **stage 1**: a SELFIES language model (MoLlama, 1.8B sequences) fixes the whole graph | stage 1 (100% valid by construction) | stage 1 | VP† | **continuous** — geometry only | 1D LM → 3D diffusion; the cleanest "formula first" design |
| **UniGEM** | ICLR 2025 | **B1** | **only coordinates are diffused**, EDM-style, `α_t² + σ_t² = 1` | **predicted after**: a 2-layer MLP head at `t = 0`, L1 loss on the one-hot, active only in the *growth* phase | distance-based inference | `p(N)`† | **VP** | **continuous** — geometry only | two-phase split at nucleation time `t_n = 10/1000`: nucleation forms anonymous geometry, growth attaches types + properties. ~10% better molecule stability than EDM |
| **SLDM** | NeurIPS 2025 | **B1** | **only coordinates**: `x_t = (1−t) x_0 + σ ε`, σ constant and ~2 orders of magnitude below data scale (0.05) | **predicted from coordinates, following UniGEM** | post hoc | `p(N)`† | **neither — SL** | **continuous** — geometry only | explicitly tabulates itself against VP (`μ²+σ²=1`) and VE (`σ=√t`) and rejects both; ~100× sampling efficiency |
| **SemlaFlow** | arXiv 2024 (v3 2025) | **B5** | CFM: `N(t x_1 + (1−t) x_0, σ²)`, σ = 0.2, `t ~ Beta(2,1)`, **equivariant OT** (rotation + permutation alignment) | **DISC CTMC, uniform prior**: `Cat(t δ + (1−t)/\|A\|)` | **diffused**, DISC CTMC | sampled from the **test-set size distribution** | **pVP** + DISC | **mixed** — continuous + discrete | charges predicted but not flowed |
| **FlowMol / FlowMol2** | arXiv 2024 | B5 | CFM | compares continuous / simplex / **discrete CTMC** variants for categoricals | diffused | `p(N)`† | pVP + DISC | **mixed** — continuous + discrete | the systematic "which state space for categoricals" study on the FM side |
| **FlowMol3** | Digital Discovery 2026 | **B5 (+B7)** | CFM, prior `N(0,I)`, **deterministic linear interpolant** `X_t = (1−t)X_0 + t X_1`, rigid-body + permutation alignment, endpoint prediction, loss weight `clamp(t/(1−t), 0.005, 1.5)` | **DISC CTMC with a *mask* prior** (everything starts masked; unmask prob. t; η controls re-masking) | **diffused**, same masked CTMC | **"fake atoms"**: an extra atom category, `U(0, 0.3N)` fake atoms added in training, anchored with Gaussian offsets → the model can *delete* atoms during sampling | **pVP** + masked CTMC | **mixed** — continuous + discrete | also: self-conditioning and **geometry distortion** (for t > 0.5, 20% of atoms displaced by 0.5 Å σ) so the model learns late-stage repair |
| **SynCoGen** | ICLR 2026 | B5 | flow matching on coordinates | **masked graph diffusion** over building blocks | masked graph diffusion | building-block count | Hybrid (pVP + mask) | **mixed** — continuous + discrete | joint reaction-graph + coordinate modelling for synthesizability |
| **GaUDI** | Nat. Comput. Sci. 2023 | B4 | EDM-like VP | continuous-relaxed† | post hoc | `p(N)` | VP† | **continuous** † | guided inverse design for polycyclic systems |
| **GPFF** *(this repo)* | arXiv 2026 | **B7 → B1** | **VE** pseudo-force field: the network learns forces/score of a Gaussian pseudo-PES, `x_t = x_0 + σ_t ε`; `MixedGeometries` additionally mixes VP, VE, a straight-line `ve_ds` variant and a **multivariate/shape** path, optionally with **per-atom timesteps**; a `ShapeGMM` fitted **per atom count** supplies principal-axis variances to a shape prior | **supplied, not generated** — atomic numbers go into the PaiNN embedding; composition comes from the data/condition generator | post hoc (RDKit/PoseBusters validity) | **drawn with the composition**, up front | **VE / genuinely Hybrid VE-VP** | **continuous** — geometry only | the `ve_ds` path `x_t = (1−t)x_0 + t σ ε` is SLDM's straight line; the VE+VP mixture is the rare case of hybridising *within* the coordinate process |

Also in this space but **not classified here** because the mechanism was not
verified in this pass: ControlMol, CGD, MuDM, GruM, TACS, ChemGuide, GOAT, RADM,
IDFlow, MolTD, UAE-3D, RL-Diffusion, Uni-3DAR, Megalodon.

---

## 3. Pocket-conditioned / structure-based generation

| Model | Venue | Bucket | Coordinates | Atom types | Bonds | Atom count N | VP/pVP/VE/Hybrid | Diffusion state space | Exact process |
|---|---|---|---|---|---|---|---|---|---|
| **liGAN** | Chem. Sci. 2022 | **B2** | **atomic-density grid**, per-element channels, VAE | implicit in channels, recovered by **atom fitting** | inferred (OpenBabel), then **UFF relaxation** | emerges from fitting | — (VAE) | N/A | the original density-grid-then-fit pipeline |
| **AR / 3DSBDD** | NeurIPS 2021 | B8 (AR) | atom-by-atom from a predicted occupancy field | with each atom | post hoc | stop criterion | — | N/A | autoregressive sampling of an occupancy probability field |
| **GraphBP** | ICML 2022 | B8 (AR) | local spherical coordinates via a normalizing flow | with each atom | post hoc | stop criterion | — | N/A | autoregressive flow |
| **Pocket2Mol** | ICML 2022 | B8 (AR) | equivariant frontier + position prediction | with each atom | **predicted per step** | stop criterion | — | N/A | the strongest autoregressive SBDD baseline |
| **TargetDiff** | ICLR 2023 | **B4 (+B7)** | Gaussian, **sigmoid β schedule, β₁ = 1e-7 → β_T = 2e-3**, T = 1000 (VP form) | **DISC uniform** (multinomial, cosine `ᾱ`, s = 0.01, Gumbel-argmax at training) | **post hoc, OpenBabel** (bond diffusion left as future work) | **`p(N \| pocket size)`**: pocket size = median of the top-10 farthest pairwise protein-atom distances, N drawn from the training histogram in that bin | **Hybrid** (VP + DISC-uniform) | **mixed** — continuous + discrete | the reference SBDD diffusion design |
| **DiffSBDD** | Nat. Comput. Sci. 2024 | **B4 (+B7)** | **VP**, `α_t = √(1−σ_t²)`, polynomial `ᾱ_t = 1 − (t/T)²`, T = 500 | **continuous-relaxed** one-hot, scaled ×0.25, noised under the *same* Gaussian kernel as coordinates | OpenBabel + RDKit sanitisation, largest fragment; **GNINA local minimisation before docking** | manual, or `N_L ~ p(N_L \| N_P)` from the smoothed empirical joint (σ = 1 Gaussian filter); means shifted +5 / +10 atoms to match test sets | **VP** | **continuous** | two modes: `-cond` (clean pocket every step) and `-joint`/inpainting (pocket injected by replacement, r = 10 resampling) |
| **DecompDiff** | ICML 2023 | **B5** | VP with **decomposed, data-dependent priors** (scaffold + arms, placed in subpockets) | DISC† | **diffused** (bond diffusion added end-to-end) | `p(N \| pocket)`† | **Hybrid** + non-standard prior | **mixed** — continuous + discrete † | the "informative prior" line; up to −8.39 Vina Dock, 24.5% success rate |
| **D3FG** | NeurIPS 2023 | **B6** | **functional groups as rigid bodies** (position + orientation), then single-atom linkers | group type from a CrossDock frequency repository | implied by group templates | per-stage counts | VP + SO(3) | **mixed** — continuous + discrete † | two-stage: groups first, then linkers |
| **IPDiff** | ICLR 2024 | B4 | VP with **interaction prior shifting** in both forward and reverse | DISC† | post hoc | `p(N \| pocket)` | Hybrid + shifted prior | **mixed** — continuous + discrete † | strongest Vina averages in several benchmarks, lower success rate |
| **PMDM** | Nat. Commun. 2024 | B4 | **dual** (local + global) VP diffusion | jointly | post hoc | `p(N \| pocket)`† | VP | **continuous** † | lead optimisation as well as de novo |
| **BindDM** | AAAI 2024 | B4 | VP, binding-adaptive hierarchical subcomplex | DISC† | post hoc | `p(N \| pocket)`† | Hybrid† | **mixed** — continuous + discrete † | |
| **IRDiff** | ICML 2024 | B4 | VP + retrieval augmentation | DISC† | post hoc | `p(N \| pocket)`† | Hybrid† | **mixed** — continuous + discrete † | |
| **MolCRAFT** | ICML 2024 | **B4** | **BFN** continuous parameter space (Gaussian with Bayesian updates) | **categorical in the same parameter space** — the point of the paper is to avoid the mixed continuous/discrete space of diffusion | post hoc | `p(N \| pocket)`† | **neither — BFN** | **continuous** — parameter space | noise-reduced sampling, >10× speedup, −6.59 Vina |
| **VoxBind** | ICML 2024 | **B2** | pocket-conditioned **voxel grid** | per-element channels, extracted after | post hoc | emerges from the grid | **VE** (single level) | **continuous** | walk-jump, the SBDD counterpart of VoxMol |
| **AliDiff** | NeurIPS 2024 | B4 | inherits TargetDiff | inherits | post hoc | inherits | Hybrid | **mixed** — continuous + discrete | exact-energy preference alignment |
| **FlexSBDD** | NeurIPS 2024 | B5† | flow matching, **flexible protein** | DISC† | diffused† | pocket-conditioned† | pVP† | **mixed** — continuous + discrete † | code not public, excluded from third-party benchmarks |
| **DrugFlow** | ICLR 2025 | **B5 (+B7)** | **conditional flow matching** | **discrete Markov bridge** | **discrete Markov bridge** | pocket-conditioned size model | **pVP** + discrete bridge | **mixed** — continuous + discrete | adds an **OOD uncertainty estimate** and preference alignment; a later version also samples side-chain angles |
| **MolPilot** | ICML 2025 | **B3 in effect** | BFN continuous part | BFN discrete part | diffused | pocket-conditioned | **neither — BFN**, with per-modality schedule | **continuous** — parameter space | **VLB-Optimal Scheduling**: the continuous and discrete times form a 2D plane and the test-time path through it is found by dynamic programming. The clearest statement that the *geometry/topology schedule asymmetry* matters. 95.9% PoseBusters |
| **CByG** | NeurIPS 2025 | B4 | BFN (MolCRAFT weights) + gradient-integrated guidance | BFN | post hoc | inherits | neither — BFN | **continuous** — parameter space | controllable property guidance |
| **PAFlow** | NeurIPS 2025 | **B4 + B7** | **variance-preserving flow-matching path** | **categorical CFM**: uniform prior interpolated to the one-hot | post hoc† | **learned atom-number predictor from pocket features** — explicitly removes the dependence on reference-ligand statistics | **VP/pVP** + CTMC | **mixed** — continuous + discrete | prior interaction guidance on the vector field; −8.31 Vina |
| **DiffBP** | Chem. Sci. 2025 | **B4 + B7** | VP, non-autoregressive full-molecule | jointly with coordinates | post hoc | **centre-of-mass prediction + explicit atom-number distribution modelling** — motivated by autoregressive models generating molecules that are too small | VP | **continuous** † | one of the few papers to treat N as a modelling target |
| **DiffLinker** | Nat. Mach. Intell. 2024 | B4 + B7 | EDM-style VP (linker atoms only) | continuous-relaxed | post hoc | **learned linker-size predictor** | VP | **continuous** | fragment linking; the size head is the cleanest example of a learned count |
| **UniMoMo** | ICML 2025 | B8 | unified block-based latent representation for small molecules, peptides, antibodies | within blocks | within blocks | block count | pVP† | **mixed** — continuous + discrete † | one model for several binder modalities |
| **CGFlow** | ICML 2025 | B6/B8 | compositional flows | building blocks | reaction graph | pathway length | pVP† | **mixed** — continuous + discrete † | co-designs the synthesis pathway |
| **FuncBind** | NeurIPS 2025 | **B2** | **neural field**, all-atom, pocket-conditioned | extracted from the field | post hoc | emerges from the field | **VE** | **continuous** | the FuncMol line extended to binders |
| **PocketXMol** | bioRxiv 2024 / Cell 2026 | B5 | **universal denoiser**: task-specific noise distributions mapped into one shared space, prompt-selected task | atoms + bonds as one set | jointly | task-dependent | non-standard (denoising without a data-dependent schedule) | **mixed** — continuous + discrete † | one model for docking, conformers, SBDD, linking, growing, PROTACs, peptides |
| **Apo2Mol** | arXiv 2025 | **B4** | ligand: Gaussian `N(√ᾱ_t x_0, (1−ᾱ_t)I)`, β 1e-7 → 2e-3, T = 1000 (VP form); **pocket: residue-level frames** (translation, quaternion, χ-angles) linearly interpolated apo → holo | **DISC uniform** | post hoc | not specified in the text | **Hybrid** | **mixed** — continuous + discrete | dynamic/flexible pocket; baselines Pocket2Mol, TargetDiff, DecompDiff, IPDiff |
| **Pharmolix-FM** | arXiv 2025 | B5† | all-atom multimodal, **diffusion *and* BFN variants** | jointly | jointly | task-dependent | VP and BFN variants | **mixed** — continuous + discrete † | reports beating PocketXMol on self-ranking (81.07 vs 78.97) |

---

## 4. What the table actually shows

**(1) The geometry-first bucket (B1) is real, small and very recent — and it is
where GPFF lives.** Only three point-cloud models diffuse *coordinates alone* and
attach element identities afterwards: **UniGEM** (ICLR 2025, two-phase
nucleation/growth, types from an MLP head at `t = 0`), **SLDM** (NeurIPS 2025,
which adopts UniGEM's type head and replaces the path), and **GPFF**. UniGEM's
stated motivation is exactly the one that motivates a geometry-only model:
lowering the dimensionality of the diffused state and not having to transport a
discrete variable through a Gaussian process. UniGEM reports ~10% better molecule
stability than EDM, SLDM reports ~100× sampling efficiency — so this is not a
curiosity, it is currently winning on both quality and cost.

**(2) Everything else that looks geometry-first is actually field-first (B2).**
liGAN, VoxMol, VoxBind, FuncMol, FuncBind and FMG generate a *type-channelled*
density or field, so the element identity is produced simultaneously with the
geometry and merely *extracted* afterwards by peak finding and optimisation.
Worth separating from B1: in B1 the type decision is a learned conditional
`p(type | geometry)`; in B2 it is a channel argmax.

**(3) These same B2 models are the only ones where N is not decided in advance.**
Every point-cloud model — VP, pVP, VE, BFN or flow — fixes N before the first
denoising step, because the Gaussian prior needs a fixed-size state. Grids and
fields do not, and the atom count simply emerges from how many peaks appear.

**(4) The field moved B4 → B5 and continuous-relaxed → discrete between 2022 and
2024.** EDM/DiffSBDD treat one-hot types as continuous values under the
coordinate noise kernel; MiDi, MolDiff, EQGAT-diff, JODO and everything on the
flow-matching side added bonds to the generative state and (except JODO) moved
types to a genuinely discrete process. EQGAT-diff is the controlled comparison:
discrete beats continuous, 98.68% vs 96.45% molecule stability on QM9, and x₀-
beats ε-parameterisation.

**(5) "Hybrid" in this literature almost always means hybrid *state space*, not
hybrid VE/VP.** The overwhelmingly common answer to the noise-family question is
"VP Gaussian on coordinates + a categorical process on types (and bonds)". A
genuine VE/VP *mixture inside the coordinate process* is rare — GPFF's
`MixedGeometries` is one, and **EquiFM's** "hybrid probability transport" (an
equivariant-OT path for coordinates and an information-matched path for types) is
the nearest published analogue.

**(6) Nobody in the mainstream point-cloud line uses a growing-σ VE schedule.**
VE in Cartesian molecular space is confined to the **walk-jump /
neural-empirical-Bayes** family, which uses the degenerate single-level case
(VoxMol σ = 0.9 constant; FuncMol/FuncBind the same in code space) — plus
SiMGen's annealed force-field score and GPFF. SLDM's explanation is the useful
one: molecular geometry is noise-sensitive, so what matters is keeping SNR high
for most of the trajectory, which VP does by rescaling and SLDM does by keeping σ
tiny and constant. A classic VE schedule does the opposite.

**(7) A third noise family has quietly become standard.** Linear-interpolant
paths — OT-CFM with σ → 0 (SemlaFlow σ = 0.2, FlowMol3 deterministic, ADiT,
DrugFlow) and SLDM's constant-σ straight line — are neither VP nor VE. Calling
them *pseudo-VP* is fair for the OT-CFM family (fixed-scale Gaussian prior,
monotone variance) but not for SLDM, which says so itself.

**(8) Bayesian flow networks are an orthogonal axis, not a schedule.** GeoBFN,
MolCRAFT, MolPilot, CByG and Pharmolix-FM put *all* modalities in one continuous
parameter space, which dissolves the continuous/discrete split rather than
hybridising it. Notably, released GeoBFN carries atom identity through a
**discretised charge** variable, with fully discrete types being an ablation that
performs worse — a second data point, alongside UniGEM, that taking types out of
the hard-categorical path can help.

**(9) True alternation (B3) does not exist, but *schedule asymmetry* does — and
it is load-bearing.** Four designs converge on it from different directions:
MiDi and EQGAT-diff give each modality its own schedule exponent (coordinates
ν = 2.5, bonds 1.5, types 1 — so geometry and bonds resolve *before* types);
MolDiff deliberately destroys bonds in steps 1–600 and denoises atoms in
600–1000; MolPilot searches the 2D (continuous-time × discrete-time) plane for a
VLB-optimal path and gets 95.9% PoseBusters; UniGEM is the degenerate limit, a
hard two-phase split. Anyone building a geometry-first model is already at the
extreme end of a trend the joint models have been creeping toward.

**(10) Atom count: six mechanisms, and none of them diffuses N.**
(i) empirical `p(N)` — EDM, GeoLDM, MiDi, MolDiff, JODO, GeoBFN, SemlaFlow;
(ii) pocket-conditioned empirical `p(N | pocket)` — TargetDiff (size-binned
histogram), DiffSBDD (smoothed joint, with a deliberate +5/+10 shift),
DecompDiff and most SBDD work;
(iii) a **learned predictor** — PAFlow's atom-number predictor, DiffBP's atom-
number distribution, DiffLinker's linker-size head;
(iv) an autoregressive **stop token** — G-SchNet, AR, Pocket2Mol, GraphBP;
(v) **slack slots** that make the effective N adaptive — FlowMol3's "fake atoms"
(`U(0, 0.3N)` extra atoms in training, anchored, deletable at sampling time) and
MolDiff's absorbing type;
(vi) **emergent** from a grid/field — liGAN, VoxMol, VoxBind, FuncMol, FMG.
The strict reading of bucket 7 — sample types and count from a prior, then do
geometry — is the *default* for the whole non-autoregressive literature; what is
genuinely rare is (iii) and (v).

**(11) Continuous vs discrete is the one axis with a clear empirical answer — and
two live dissents.** The counts: **23 continuous**, **23 mixed**, **5
continuous-but-geometry-only**, **4 continuous-in-parameter-space**, 6 N/A. The
2022 cohort relaxed one-hot types into the coordinate kernel; by 2024 the
leaders had moved types (and bonds) into a genuinely discrete process, and
EQGAT-diff's controlled comparison is why — discrete 98.68% vs continuous 96.45%
molecule stability on QM9, same network, same schedule. But two independent
lines dissent, and both win on their own benchmarks. **BFNs** (MolCRAFT's
explicit thesis) put categoricals back into a continuous parameter space and
argue that diffusion's mixed continuous/discrete state is the problem, not the
solution. **Geometry-only models** (UniGEM, SLDM, and GPFF) sidestep the
question: if atom types are never diffused, there is no discrete state space to
design. GeoBFN carrying type through a *discretised charge* — with fully
discrete types an ablation that performs worse — sits between the two dissents.
So "discrete beats continuous" holds only *given* that types are diffused
jointly; take them out of the diffusion and the result does not apply.


---

## 5. Where GPFF sits, and what the neighbourhood implies

GPFF is **B7 → B1 with a VE (optionally hybrid VE/VP) coordinate process**:
composition and count come from the data/condition generator, a `ShapeGMM`
fitted **per atom count** supplies principal-axis variances to the prior, and
the network learns the forces of a Gaussian pseudo-PES rather than an ε-target.
That combination has no exact precedent in the table. The nearest neighbours:

* **UniGEM / SLDM** — same geometry-first commitment, but types are *predicted by
  the model* rather than supplied, and both stay in the VP/SL regime. The obvious
  comparison to make explicit: GPFF's `ve_ds` path, `x_t = (1−t)x_0 + tσε`, **is**
  SLDM's straight line, so that ablation is already a baseline reproduction.
* **SiMGen** — training-free force-field score + point-cloud shape priors; the
  closest in spirit to a pseudo-force-field sampler, and the only other model
  that handles elements by something other than a learned joint process (an
  evolutionary swap).
* **VoxMol / FuncMol / FuncBind** — the only other VE family, via single-σ
  neural empirical Bayes. Different representation, same insight that one
  well-chosen noise level plus a good denoiser is enough.
* **ShapeMol / DiffSMol and UniGuide** — shape- and geometry-conditioned
  generation, the natural comparison for the `ShapePrior` / `ShapeConditionGenerator`
  path (not classified above; not verified in this pass).

Two gaps the survey makes visible, both cheap to occupy from this codebase:

1. **No pocket-conditioned geometry-first model exists.** Every SBDD entry is B4,
   B5, B2 or autoregressive. A pocket-conditioned pseudo-force field with
   composition drawn from `p(types, N | pocket)` would be the first B1/B7 entry in
   section 3, and the count side already has well-specified baselines to beat
   (TargetDiff's size-binned histogram, PAFlow's learned predictor).
2. **The VE-vs-VP question has never been asked properly in this setting.**
   EQGAT-diff's design-space paper ablates state spaces, parameterisations and
   loss weightings but — by its own text — runs **no noise-schedule ablation**.
   `MixedGeometries` already implements VP, VE, straight-line and multivariate
   paths behind one switch, which is exactly the experiment the literature is
   missing.

---

## 6. Sources

Primary texts consulted for this pass (details above were read from these):

- EDM — <https://arxiv.org/abs/2203.17003>
- UniGEM — <https://arxiv.org/abs/2410.10516>, ICLR 2025
- SLDM — <https://arxiv.org/abs/2503.02918>, NeurIPS 2025
- MiDi — <https://arxiv.org/abs/2302.09048>
- MolDiff — <https://arxiv.org/abs/2305.07508>, ICML 2023
- EQGAT-diff — <https://arxiv.org/abs/2309.17296>, ICLR 2024
- JODO — <https://arxiv.org/abs/2305.12347>
- GeoBFN — <https://arxiv.org/abs/2403.15441>, ICLR 2024
- EquiFM — <https://arxiv.org/abs/2312.07168>, NeurIPS 2023
- VoxMol — <https://arxiv.org/abs/2306.07473>, NeurIPS 2023
- FuncMol — <https://arxiv.org/abs/2501.08508>, NeurIPS 2024
- FMG — <https://arxiv.org/abs/2402.15864>, ICLR 2025
- SiMGen — <https://arxiv.org/abs/2402.08708>, Nat. Commun. 2025
- HierDiff — <https://arxiv.org/abs/2305.13266>, ICML 2023
- GeoRCG — <https://arxiv.org/abs/2410.03655>, ICML 2025
- ADiT — <https://arxiv.org/abs/2503.03965>, ICML 2025
- NExT-Mol — <https://arxiv.org/abs/2502.12638>, ICLR 2025
- SemlaFlow — <https://arxiv.org/abs/2406.07266>
- FlowMol3 — <https://doi.org/10.1039/D5DD00363F>, Digital Discovery 2026
- TargetDiff — <https://arxiv.org/abs/2303.03543>, ICLR 2023
- DiffSBDD — <https://arxiv.org/abs/2210.13695>, Nat. Comput. Sci. 2024
- DecompDiff — <https://arxiv.org/abs/2403.07902>, ICML 2023
- D3FG — <https://arxiv.org/abs/2306.13769>, NeurIPS 2023
- MolCRAFT — <https://arxiv.org/abs/2404.12141>, ICML 2024
- MolPilot (VOS) — <https://arxiv.org/abs/2505.07286>, ICML 2025
- DrugFlow — <https://arxiv.org/abs/2508.17815>, ICLR 2025
- PAFlow — <https://arxiv.org/abs/2509.01486>, NeurIPS 2025
- DiffBP — <https://arxiv.org/abs/2211.11214>, Chem. Sci. 2025
- Apo2Mol — <https://arxiv.org/abs/2511.14559>
- END — <https://arxiv.org/abs/2506.10532>, NeurIPS 2024
- PocketXMol — <https://doi.org/10.1101/2024.10.17.618827>, Cell 2026
- Pharmolix-FM — <https://arxiv.org/abs/2503.21788>
- Diffusion Models for Molecules survey — <https://arxiv.org/abs/2502.09511>
- Model index used for coverage — <https://github.com/AzureLeon1/awesome-molecular-diffusion-models>
