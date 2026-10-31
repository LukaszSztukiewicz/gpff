# Historical relaxation study

Unmodified scripts and reports from `feat/uff-relaxation`, commit
`43858fdeb9b16f62bc1d1a5721ce6490d5867241` (2026-10-31), rebased as a
regular archive commit on top of `feat/formula-report-fixes`. The old UFF
and xTB implementation commits are superseded by that base's implementations.

These files document the fancy-monkey epoch-5000 / SOB999 study. They are
historical recipes, with hard-coded cluster paths, model assets, and an old
`gpff.analysis.get_bond_lengths` API. Run them only with their original code
and recorded inputs. They are not the current evaluation entry points.
The original single-structure relaxation module, including that helper, is
preserved verbatim in `legacy_relaxation.py` for historical reference; it is
not part of the installed package.

The JSON and Markdown reports are historical results, not a fresh validation.
They compare relaxed geometries against raw QM9 DFT geometries and use broad
bond types, unlike the current environment-conditioned metrics. Their values
are not interchangeable with current scores. Establish the exact database,
split, model and relaxation settings before reusing them in a comparison;
the reports alone do not contain all that provenance.

For new work, use `gpff_relax`, `gpff_relax_report`, `gpff_refine` and
`gpff_damping_probe`. Relax the reference with the same method and settings
as the samples before building the relaxed reference's evaluation assets.
An independent set of real molecules from the same database and protocol can
estimate sampling variability; a reference compared with itself cannot.
