"""
Composition metrics for runs that generate the atom types.

Validity says nothing about *which* elements a model produces, so a model that
suppresses the rare elements can score better than one that gets them right:
dropping a nitrogen usually turns a hard valence into an easy one. These metrics
make that visible.

Reported over a set of molecules, by ``gpff_calculate_metrics --metrics
composition`` over all samples:

    element_fraction_<symbol>   share of the atoms that are this element
    molecule_fraction_<symbol>  share of the molecules that contain it
    heteroatom_fraction         share of the atoms that are neither H nor C
    mean_n_atoms                mean number of atoms per molecule
    element_tv                  total variation distance of the atom-level
                                element distribution to the reference
    size_w1                     Wasserstein-1 distance of the molecule size
                                distribution to the reference, in atoms

The molecules are count matrices, one row per molecule and one column per
element (``gpff.analysis.formula.count_matrix`` of an element count table).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from . import formula

__all__ = ["composition_metrics"]


def composition_metrics(
    counts: np.ndarray,
    symbols: Sequence[str],
    reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Reduce the compositions of a set of molecules to the metrics above.

    Args:
        counts: (n_molecules, n_elements) atoms of each element per molecule.
        symbols: the element of each column; every one is reported.
        reference: the reference's count matrix, with the same columns, for
            element_tv and size_w1.
    """
    if not len(counts):
        return {}
    shares = formula.element_shares(counts)
    metrics: Dict[str, float] = {
        f"element_fraction_{symbol}": float(share) for symbol, share in zip(symbols, shares)
    }
    for k, symbol in enumerate(symbols):
        metrics[f"molecule_fraction_{symbol}"] = float((counts[:, k] > 0).mean())
    heteroatoms = [k for k, symbol in enumerate(symbols) if symbol not in ("H", "C")]
    total = counts.sum()
    metrics["heteroatom_fraction"] = float(counts[:, heteroatoms].sum() / total) if total else 0.0
    sizes = counts.sum(axis=1)
    metrics["mean_n_atoms"] = float(sizes.mean())
    if reference is not None and len(reference):
        metrics["element_tv"] = formula.total_variation(shares, formula.element_shares(reference))
        metrics["size_w1"] = formula.count_w1(sizes, reference.sum(axis=1))
    return metrics
