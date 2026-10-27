"""
Composition metrics for runs that generate the atom types.

Validity says nothing about *which* elements a model produces, so a model that
suppresses the rare elements can score better than one that gets them right:
dropping a nitrogen usually turns a hard valence into an easy one. These metrics
make that visible.

Reported per sample database, over all molecules or only the RDKit-valid ones:

    element_fraction_<symbol>   share of the atoms that are this element
    molecule_fraction_<symbol>  share of the molecules that contain it
    heteroatom_fraction         share of the atoms that are neither H nor C
    mean_n_atoms                mean number of atoms per molecule
    element_tv                  total variation distance of the atom-level
                                element distribution to the reference
    size_w1                     Wasserstein-1 distance of the molecule size
                                distribution to the reference, in atoms

The reference is another sample database, normally the untyped baseline's: it
was sampled on the same split, so it carries exactly the element distribution
and the atom counts the typed model should reproduce. A dataset database works
too.

Run it as

    python -m gpff.analysis.composition <samples.db>... [--reference <db>]

Written as its own module so that it can be folded into
``gpff_calculate_metrics`` later without rewriting anything: ``--json`` writes
``composition.json`` next to each database's ``results.json``, with the same
flat ``{name: value}`` shape.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.data import chemical_symbols
from ase.db import connect


__all__ = [
    "QM9_ELEMENTS",
    "read_compositions",
    "composition_metrics",
    "load_valid_mask",
]

# the elements of QM9, in the order experiment=gpff_types one-hot encodes them
QM9_ELEMENTS: Tuple[int, ...] = (1, 6, 7, 8, 9)


def read_compositions(
    db_path: str | Path,
    limit: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Read the atomic numbers of every molecule of an ASE database.

    Args:
        db_path: the database to read.
        limit: stop after this many molecules.

    Returns:
        numbers: one array of atomic numbers per molecule, in database order.
        sizes: the number of atoms of each molecule.
    """
    numbers: List[np.ndarray] = []
    with connect(str(db_path)) as db:
        for index, row in enumerate(db.select()):
            if limit is not None and index >= limit:
                break
            numbers.append(np.asarray(row.numbers))
    sizes = np.array([len(n) for n in numbers], dtype=int)
    return numbers, sizes


def load_valid_mask(db_path: str | Path) -> Optional[np.ndarray]:
    """
    The RDKit validity mask ``gpff_prepare_metadata`` cached next to a sample
    database, or None when it was not prepared.
    """
    path = Path(db_path).parent / "evaluation" / "metadata" / "rdkit_valid.npy"
    if not path.exists():
        return None
    return np.load(path).astype(bool)


def _element_distribution(
    numbers: Sequence[np.ndarray],
    elements: Sequence[int],
) -> Dict[int, float]:
    counts: collections.Counter = collections.Counter()
    for molecule in numbers:
        counts.update(molecule.tolist())
    total = sum(counts.values())
    if total == 0:
        return {z: 0.0 for z in elements}
    return {z: counts.get(z, 0) / total for z in elements}


def _size_histogram(sizes: np.ndarray, n_bins: int) -> np.ndarray:
    histogram = np.bincount(np.clip(sizes, 0, n_bins - 1), minlength=n_bins)
    total = histogram.sum()
    return histogram / total if total else histogram.astype(float)


def composition_metrics(
    numbers: Sequence[np.ndarray],
    sizes: np.ndarray,
    reference_numbers: Optional[Sequence[np.ndarray]] = None,
    reference_sizes: Optional[np.ndarray] = None,
    elements: Sequence[int] = QM9_ELEMENTS,
) -> Dict[str, float]:
    """
    Reduce the compositions of a set of molecules to the metrics above.

    Args:
        numbers: one array of atomic numbers per molecule.
        sizes: the number of atoms of each molecule.
        reference_numbers: the reference's atomic numbers, for element_tv.
        reference_sizes: the reference's molecule sizes, for size_w1.
        elements: the elements to report, by atomic number.
    """
    metrics: Dict[str, float] = {"n_molecules": float(len(numbers))}
    if not len(numbers):
        return metrics

    distribution = _element_distribution(numbers, elements)
    for z in elements:
        metrics[f"element_fraction_{chemical_symbols[z]}"] = float(distribution[z])
    for z in elements:
        present = np.mean([bool((molecule == z).any()) for molecule in numbers])
        metrics[f"molecule_fraction_{chemical_symbols[z]}"] = float(present)

    n_atoms = sum(len(molecule) for molecule in numbers)
    n_heteroatoms = sum(int(((m != 1) & (m != 6)).sum()) for m in numbers)
    metrics["heteroatom_fraction"] = float(n_heteroatoms / n_atoms) if n_atoms else 0.0
    metrics["mean_n_atoms"] = float(np.mean(sizes))

    if reference_numbers is not None and len(reference_numbers):
        # every element of either set, so an element the model invents counts
        seen = sorted(
            {int(z) for m in numbers for z in np.unique(m)}
            | {int(z) for m in reference_numbers for z in np.unique(m)}
        )
        generated = _element_distribution(numbers, seen)
        reference = _element_distribution(reference_numbers, seen)
        metrics["element_tv"] = float(
            0.5 * sum(abs(generated[z] - reference[z]) for z in seen)
        )
    if reference_sizes is not None and len(reference_sizes):
        n_bins = int(max(sizes.max(), reference_sizes.max())) + 1
        generated_sizes = _size_histogram(sizes, n_bins)
        reference_size_histogram = _size_histogram(reference_sizes, n_bins)
        metrics["size_w1"] = float(
            np.abs(np.cumsum(generated_sizes) - np.cumsum(reference_size_histogram)).sum()
        )
    return metrics


def _report(
    db_paths: Sequence[str],
    labels: Sequence[str],
    reference: Optional[str],
    valid_only: bool,
    elements: Sequence[int],
    limit: Optional[int],
    write_json: bool,
) -> List[Tuple[str, Dict[str, float]]]:
    reference_numbers = reference_sizes = None
    if reference is not None:
        reference_numbers, reference_sizes = read_compositions(reference, limit)
        if valid_only:
            mask = load_valid_mask(reference)
            if mask is not None:
                k = min(len(mask), len(reference_numbers))
                reference_numbers = [
                    m for m, keep in zip(reference_numbers[:k], mask[:k]) if keep
                ]
                reference_sizes = np.array([len(m) for m in reference_numbers], dtype=int)

    rows: List[Tuple[str, Dict[str, float]]] = []
    for label, path in zip(labels, db_paths):
        numbers, sizes = read_compositions(path, limit)
        if valid_only:
            mask = load_valid_mask(path)
            if mask is None:
                raise SystemExit(
                    f"--valid_only needs the rdkit_valid.npy of {path}; run "
                    "gpff_prepare_metadata first."
                )
            k = min(len(mask), len(numbers))
            numbers = [m for m, keep in zip(numbers[:k], mask[:k]) if keep]
            sizes = np.array([len(m) for m in numbers], dtype=int)
        metrics = composition_metrics(
            numbers, sizes, reference_numbers, reference_sizes, elements
        )
        rows.append((label, metrics))
        if write_json:
            out = Path(path).parent / "evaluation" / "composition.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2) + "\n")
    return rows


def _print_table(rows: Sequence[Tuple[str, Dict[str, float]]], elements: Sequence[int]) -> None:
    symbols = [chemical_symbols[z] for z in elements]
    columns = (
        ["n_molecules", "mean_n_atoms"]
        + [f"element_fraction_{s}" for s in symbols]
        + [f"molecule_fraction_{s}" for s in symbols]
        + ["heteroatom_fraction", "element_tv", "size_w1"]
    )
    headers = (
        ["n", "natoms"]
        + [f"at%{s}" for s in symbols]
        + [f"mol%{s}" for s in symbols]
        + ["het%", "elemTV", "sizeW1"]
    )
    width = max(len(label) for label, _ in rows)
    print(f"{'model':<{width}} " + "".join(f"{h:>8s}" for h in headers))
    print("-" * (width + 1 + 8 * len(headers)))
    for label, metrics in rows:
        cells = []
        for column in columns:
            value = metrics.get(column)
            if value is None:
                cells.append(f"{'-':>8s}")
            elif column == "n_molecules":
                cells.append(f"{int(value):>8d}")
            elif column in ("mean_n_atoms", "size_w1"):
                cells.append(f"{value:>8.2f}")
            elif column == "element_tv":
                cells.append(f"{value:>8.4f}")
            else:
                cells.append(f"{100 * value:>8.2f}")
        print(f"{label:<{width}} " + "".join(cells))
    print("\nat% / mol% / het% are percentages; elemTV in [0,1]; sizeW1 in atoms.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpff.analysis.composition",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("samples_db_paths", nargs="+", help="sample databases to report")
    parser.add_argument(
        "--reference",
        default=None,
        help="sample or dataset database for element_tv and size_w1",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated row labels, default the databases' parent folders",
    )
    parser.add_argument(
        "--valid_only",
        action="store_true",
        help="restrict to the RDKit-valid molecules (needs the prepared metadata)",
    )
    parser.add_argument(
        "--elements",
        default=",".join(str(z) for z in QM9_ELEMENTS),
        help="comma-separated atomic numbers to report",
    )
    parser.add_argument("--limit", type=int, default=None, help="molecules per database")
    parser.add_argument(
        "--json",
        action="store_true",
        help="also write evaluation/composition.json next to each database",
    )
    args = parser.parse_args(argv)

    elements = [int(z) for z in args.elements.split(",") if z]
    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(args.samples_db_paths):
            raise SystemExit("--labels needs one label per database.")
    else:
        labels = [
            "/".join(Path(p).parts[-4:-1]) if len(Path(p).parts) >= 4 else Path(p).stem
            for p in args.samples_db_paths
        ]

    rows = _report(
        args.samples_db_paths,
        labels,
        args.reference,
        args.valid_only,
        elements,
        args.limit,
        args.json,
    )
    _print_table(rows, elements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
