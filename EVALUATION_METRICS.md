# Molecular Generation Evaluation: Literature Survey and GPFF Priorities

Scope: representative benchmark papers and 3D generation work available through 26 October 2026. The literature does not define one universal score: metrics depend on whether the task is distribution learning, 3D conformer generation, property conditioning, or pocket-conditioned design. The suite should report metric families with their protocols and denominators, rather than collapse them into one score.

## Recommendation at a glance

For GPFF, the most informative core is: (1) chemistry and graph validity, (2) 3D geometric and energetic plausibility, (3) distribution fidelity plus coverage and novelty, and (4) uncertainty and compute needed to obtain valid samples. Add conditioning and protein interaction modules when those tasks are evaluated. Keep raw-pose and relaxed-pose results separate.

| Priority | Module | Core metrics | Usability | Impact on GPFF conclusions | Cross-paper comparability |
|---|---|---|---|---|---|
| P0 | Evaluation protocol | Fixed data split, sample budget, repeated seeds, explicit denominators, means and confidence intervals | High | Very high | Very high when protocol matches |
| P0 | Molecular identity and chemistry | RDKit validity, atom stability, molecule stability, uniqueness, novelty; counts of valid, unique, and novel molecules | High | Very high | High on the same dataset and definitions |
| P0 | 3D structure quality | PoseBusters checks, bond-length/angle/torsion and distance distributions, strain or energy, relaxation displacement | Medium | Very high for position diffusion | Medium; software, force field, and reference choices matter |
| P0 | Distribution and coverage | Atom/element/bond/size/property distributions; JSD or Wasserstein distance; diversity and nearest-neighbor diagnostics | High–medium | High | Medium–high with fixed features and estimator settings |
| P1 | Standard 2D distribution suite | MOSES FCD, SNN, IntDiv, Frag, Scaff; GuacaMol KL/FCD and benchmark scores | Medium | High for chemical-space learning | High on the benchmark's fixed data and implementation |
| P1 | Sampling efficiency | Wall time, NFE, valid molecules per second, cost per valid/unique molecule, quality-versus-compute curves | Medium | High for sampler research | Medium; hardware and software must be recorded |
| P2 | Property conditioning | Target-property MAE, target hit rate, constraint satisfaction, and diversity among hits | Medium | Very high for conditional extensions | Medium; predictor and target ranges must match |
| P2 | Pocket conditioning | Vina Score/Min/Dock, high-affinity rate, physical interaction checks, pocket fit, QED/SA and per-pocket success | Low–medium | Very high for pocket research | Medium–low unless the pocket set and docking protocol match exactly |

Usability means practical cost and input availability for this codebase. Impact means how much the metric can distinguish a meaningful model improvement or failure. Comparability means how often another paper uses a sufficiently similar definition and protocol. These are qualitative rankings, not measured scores.

## Metric inventory

### 1. Chemical validity, identity, and exploration

| Metric | What it measures | Use and caveat |
|---|---|---|
| Chemical validity | Fraction of attempted samples that parse and satisfy the stated valence, charge, aromaticity, and stereochemistry rules | Essential, but specify the chemistry toolkit, sanitization, hydrogen handling, and whether the graph is given or inferred from coordinates. Report the numerator and denominator. |
| Atom stability / molecule stability | EDM-style fraction of atoms with expected valence / fraction of molecules whose every atom passes that test, often with bonds inferred from distances | Directly relevant to joint position and atom-type generation. It is a useful heuristic, not a complete chemical or physical validity test; report bond-inference thresholds and atom set. |
| Uniqueness | Distinct canonical molecular identities among generated samples | Report uniqueness among valid samples and the raw counts. State whether stereochemistry, isotopes, charges, and tautomers distinguish identities. |
| Novelty | Fraction absent from a declared reference set | Compare against training data for memorization and optionally a broader external reference for known-chemistry novelty. Novelty alone does not imply useful or diverse chemistry. |
| Internal diversity | Pairwise fingerprint diversity, commonly 1 − mean Tanimoto similarity (`IntDiv`) | Detects collapse, but high diversity can mean samples are unlike the intended data or task. Fix fingerprint type, size, and sample count. |
| Nearest-neighbor similarity (SNN) | Average similarity of generated molecules to their closest reference molecule | A reference-set support diagnostic; it is not the same as internal diversity. High or low values are not inherently better without the task context. |
| Fragment and scaffold scores | MOSES frequency similarity for molecular fragments and Bemis–Murcko scaffolds | Interpretable distribution diagnostics; sensitive to canonicalization and dataset size. |
| Precision / recall or coverage | Embedding-space estimate of generated-sample quality (precision) and reference-distribution coverage (recall) | Helps separate fidelity from mode coverage, but results depend strongly on the molecular representation and estimator. Treat as a secondary diagnostic unless the benchmark fixes both. |

MOSES reports Valid, Unique@k, Novel, FCD, SNN, internal diversity, fragment and scaffold similarity, and filters. GuacaMol distribution learning reports validity, uniqueness, novelty, KL divergence, and Fréchet ChemNet Distance (FCD); its goal-directed track adds rediscovery, similarity, isomer, median-molecule, and multi-objective tasks. [MOSES](https://pmc.ncbi.nlm.nih.gov/articles/PMC7775580/) · [GuacaMol](https://doi.org/10.1021/acs.jcim.8b00839)

### 2. Distribution fidelity and molecular properties

| Metric family | Common metrics | Use and caveat |
|---|---|---|
| Learned molecular representation | FCD (Fréchet ChemNet Distance) | Popular whole-set comparison for 2D molecule generation. It depends on ChemNet and canonical SMILES preprocessing; it does not identify which molecular feature failed. It is not a 3D geometry metric. |
| Property distribution | KL divergence, Jensen–Shannon divergence, Wasserstein distance, and sometimes MMD for molecular-weight, atom-count, ring-count, charge, logP, TPSA, QED, SA, or QM properties | Report the compared property, reference split, binning/estimator, and sample counts. Wasserstein distance retains units; histogram divergences depend on bins and range. |
| Composition and graph statistics | Element/atom-type and atom-count frequencies; bond-order, valence, ring, fragment, and scaffold distributions | Particularly important when atom types and positions are jointly generated. A good average property score can conceal the wrong elemental or graph distribution. |
| Dataset likelihood | NLL or log likelihood, where the model defines a comparable normalized density | Useful within a fixed representation and likelihood convention. Not available or directly comparable for many diffusion implementations, and not a substitute for sample quality. |

For GPFF, preserve the existing QM9 property-distribution evaluations and extend them to model-generated atom/type and graph statistics. Compare on the held-out test/reference split, and include both a single-feature summary and interpretable per-feature distributions.

### 3. 3D geometry, stability, and conformers

| Metric family | Common metrics | Use and caveat |
|---|---|---|
| EDM molecular stability | Atom stability and molecule stability using distance-inferred bonds and valence rules | A common 3D unconditional-generation baseline. Publish the bond assignment rule and avoid presenting it as equivalent to RDKit sanitization. [EDM](https://proceedings.mlr.press/v162/hoogeboom22a.html) |
| Pose and conformation checks | PoseBusters pass rates and individual checks; GenBench3D Validity3D based on reference bond-length and valence-angle likelihoods | More sensitive to whether a generated pose is physically plausible than 2D graph validity. Report individual failure categories as well as the all-check pass rate. GenBench3D found that many generated pocket ligands failed conformation checks and that relaxation could change docking results. [PoseBusters](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/) · [GenBench3D](https://arxiv.org/abs/2407.04424) |
| Local geometry distributions | Bond lengths, bond angles, torsions, and intra-molecular pair distances; compare with JSD or Wasserstein distance | High-value diagnostics for a force-field generator. Stratify by chemical environment and element/bond type; an aggregate can hide local failures. |
| Energetics and relaxation | Force-field or learned-potential energy, strain relative to a relaxed structure, energy distributions, relaxation success, and coordinate displacement | Report the energy model, units, protonation, optimization settings, and whether reference molecules went through the same relaxation. Raw and relaxed structures answer different questions. |
| Conformer accuracy | Aligned RMSD, RMSD coverage (COV-R/COV-P), and conformer energy ranking | Appropriate when the molecular graph and reference conformers are known. Not meaningful as a direct per-sample score for unconditional generation without a matched reference. |
| Emerging geometry diagnostics | High-energy atom detection and torsional energy descriptors | Useful additional analyses for conformation rationality; they are newer and not yet as standardized across papers as validity or docking metrics. [2026 conformation assessment](https://www.nature.com/articles/s41467-026-69303-5) |

3D diffusion papers commonly report validity, uniqueness, novelty, atom/molecule stability, and sometimes NLL and interatomic-distance JSD. Work on larger molecules and pocket generation increasingly adds PoseBusters or dedicated conformation checks because inferred connectivity and valid 2D chemistry do not guarantee a plausible 3D pose. [GCDM](https://www.nature.com/articles/s42004-024-01233-z)

### 4. Conditional generation and protein pockets

| Metric family | Common metrics | Use and caveat |
|---|---|---|
| Condition satisfaction | MAE/RMSE between requested and independently predicted properties; fraction inside a target tolerance or satisfying constraints | Report predictor and its held-out error. Pair hit rate with the generated property distribution and diversity among successful samples. |
| Docking | Vina Score (generated pose), Vina Min (locally minimized pose), Vina Dock (redocked pose), and high-affinity rate relative to a stated reference | Widely reported for pocket-conditioned generation, but docking scores are approximate and can reward clashes or implausible poses. Keep the three protocols separate and document receptor preparation, protonation, box, software, and version. [TargetDiff](https://openreview.net/pdf?id=kJqXEPXMsE0) |
| Protein–ligand physical quality | Steric clashes, hydrogen bonds, hydrophobic and salt-bridge contacts, interaction fingerprints, key-contact recovery, ligand strain, and pose/binding-mode similarity | Necessary to test whether a ligand actually fits and interacts with its pocket. PoseCheck highlights clashes, missing interactions, and strain; DrugPose proposes Simbind for binding-mode similarity. [PoseCheck](https://arxiv.org/abs/2308.07413) · [DrugPose](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00076e) |
| Drug-like properties | QED, SA, Lipinski rule count, molecular weight, logP, TPSA, and sometimes synthetic/medicinal chemistry filters | Report distributions and joint pass rates, not just means. QED and SA are proxies, not evidence of experimental developability or synthesis. |
| Per-target performance | Per-pocket valid-output rate, high-affinity/success rate, property and diversity distributions, and fraction of targets with usable outputs | Aggregate per target before averaging so a few easy/high-output pockets do not dominate. Report failed targets and fixed samples per pocket. |

Pocket papers frequently report Vina score variants, QED, SA, Lipinski, diversity, validity, and high-affinity rate. Newer analyses emphasize conformation quality and protein–ligand interactions; a favorable Vina score alone is weak evidence of a useful binder. [TargetDiff](https://openreview.net/pdf?id=kJqXEPXMsE0) · [PoseCheck](https://arxiv.org/abs/2308.07413) · [GenBench3D](https://arxiv.org/abs/2407.04424)

## Rankings by decision value

### Usability: easiest to most demanding

1. RDKit validity, canonical identity counts, uniqueness, novelty, atom counts, and simple element/property histograms. These need generated structures plus a fixed reference set.
2. Pairwise fingerprint diversity, SNN, graph/fragment/scaffold statistics, and JSD/Wasserstein comparisons. They need declared fingerprints, reference data, and estimator settings.
3. Bond/angle/torsion and pair-distance distributions, EDM stability, and PoseBusters/GenBench3D. They need 3D coordinates, chemistry rules, reference geometry, and pinned toolkit versions.
4. Relaxation energies and conformer coverage. These add an optimizer or potential and require matched relaxation protocols or known conformer references.
5. Pocket docking, interaction fingerprints, and per-pocket success. These require prepared proteins, docking/scoring software, pocket-specific aggregation, and more substantial compute.

### Impact on GPFF conclusions: highest to lowest

1. 3D chemical plausibility: atom/molecule stability, geometry checks, and energetics. These test the central claim that generated positions form usable molecules.
2. Joint distribution learning: element/atom-count and graph statistics, property and geometry distributions, uniqueness, novelty, and coverage. These reveal collapse, memorization, and mismatched sampling.
3. Condition success with diversity: target hit rates/MAE, pocket interactions and conformation validity, plus per-condition success. Essential to judge future conditioning work.
4. Valid-sample efficiency and quality-versus-compute curves. Needed to compare sampler changes fairly.
5. Single mean drug-likeness, one docking score, or likelihood alone. Useful context, but each can improve while chemistry, pose quality, coverage, or other objectives degrade.

### Cross-paper comparability: strongest to weakest

1. Fixed benchmark protocols for validity/uniqueness/novelty, EDM stability, and MOSES/GuacaMol results, when dataset version, split, sample count, canonicalization, and implementation match.
2. Pocket Vina variants and QED/SA/Lipinski. Commonly reported, but receptor preparation, docking modes, selection of generated samples, and aggregation often differ; published numbers are comparable only after protocol matching.
3. PoseBusters/GenBench3D and geometric distribution metrics. Stronger measures of 3D quality, but version, config, reference conformers, element handling, and thresholds must be fixed.
4. Custom energies, learned property predictors, novel interaction scores, NLL across different representations, and bespoke success criteria. Valuable within a controlled study; weak cross-paper comparators without shared assets and code.

## GPFF coverage and the next modular boundary

The current evaluator already has a useful two-stage structure: per-sample metadata preparation and metric reduction. It covers RDKit validity, a molecule-only PoseBusters pass rate, uniqueness, novelty, mean pairwise Tanimoto similarity, QM9 property JSDs, bond-length and bond-angle Wasserstein distances, and mean relaxation strain. See `src/gpff/cli/prepare_metadata.py`, `src/gpff/cli/calculate_metrics.py`, `src/gpff/analysis/structure.py`, and `src/gpff/analysis/collection.py`.

The main gaps for the stated research direction are:

- Add explicit EDM atom/molecule stability, element and atom-count distributions, bond-order/valence distributions, torsions and pair-distance distributions. Keep coordinate-inferred connectivity distinct from a model-provided graph.
- Add MOSES/GuacaMol-style FCD, SNN, fragment/scaffold and fixed-budget internal-diversity reports where the dataset and representation fit. The current mean pairwise Tanimoto similarity is an internal-similarity statistic; it is not SNN and has the opposite direction from diversity.
- Expand geometry checks to report individual PoseBusters failures and GenBench3D-style validity. The present PoseBusters `mol` configuration does not evaluate protein–ligand clashes or pocket interactions.
- Add property-conditioning and pocket profiles only for those tasks: independent condition satisfaction; then docking, interaction, ligand-strain and per-pocket success metrics with fixed receptors and protocols.
- Save protocol metadata with every result: dataset and split fingerprints, generated/valid/unique/novel counts, seeds, identity rules, hydrogens/charges/stereochemistry, software versions, reference assets, relaxation/docking settings, and runtime/compute.

A modular implementation should let each metric declare its task profile, required metadata, reference assets, direction, aggregation, and software/version requirements. Separate calculators from reducers, and provide profiles for unconditional QM9/GEOM, property-conditioned generation, conformer generation, and pocket-conditioned generation. The common report should include raw counts, per-seed results and uncertainty, distributions or per-target rows, and a small benchmark table of established headline metrics. Avoid combining heterogeneous metrics into an unvalidated single score.

Trajectory and experiment diagnostics are described separately in [EVALUATION_DIAGNOSTICS.md](EVALUATION_DIAGNOSTICS.md); they explain how a sample was produced and where a run's aggregate metrics came from.

## Selected literature

- Brown et al., [GuacaMol: Benchmarking Models for de Novo Molecular Design](https://doi.org/10.1021/acs.jcim.8b00839), 2019.
- Polykovskiy et al., [Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC7775580/), 2020.
- Hoogeboom et al., [Equivariant Diffusion for Molecule Generation in 3D](https://proceedings.mlr.press/v162/hoogeboom22a.html), 2022.
- Guan et al., [Target-aware 3D molecular generation with conditional diffusion models](https://openreview.net/pdf?id=kJqXEPXMsE0), ICLR 2023.
- Schneuing et al., [Structure-based drug design with equivariant diffusion models](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659159/), 2024.
- Buttenschoen et al., [PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/), 2024.
- Baillif et al., [Benchmarking structure-based three-dimensional molecular generative models using GenBench3D](https://arxiv.org/abs/2407.04424), 2024.
- Harris et al., [Benchmarking Generated Poses: How Rational is Structure-based Drug Design with Generative Models? (PoseCheck)](https://arxiv.org/abs/2308.07413), 2024.
- Thölke and De Fabritiis, [Geometry-complete diffusion for 3D molecule generation and optimization](https://www.nature.com/articles/s42004-024-01233-z), 2024.
- [DrugPose: benchmarking 3D generative methods for early stage drug discovery](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00076e), 2024.
- [Assessing conformation validity and rationality of deep learning-generated 3D molecules](https://www.nature.com/articles/s41467-026-69303-5), 2026.
