"""Generate the reference asset files from the reference database (``gpff_reference_assets``).

The evaluation reads its reference data from files in assets/ rather than from a
reference database (see ``gpff_calculate_metrics``). This script produces those
files from the reference QM9 database so they can be rebuilt from scratch:

    <prefix>_energy_U0_values.npy       reference energy_U0 distribution
    <prefix>_gap_values.npy             reference gap distribution
    <prefix>_elongation_values.npy      reference elongation distribution
    <prefix>_energy_U0_atomrefs_eV.npy  single-atom reference energies
    <prefix>_smiles_values.txt          one SMILES per molecule, in database order

An array stored in the database's metadata under its own name is used as it is.
Otherwise it is derived from the database, which then needs energy_U0 and gap
in Hartree and distances in Angstrom, as schnetpack's QM9 stores them:

    energy_U0_atomrefs_eV  the metadata's energy_U0 atomrefs, in eV
    energy_U0_values       each molecule's energy_U0 minus the atomrefs of its
                           atoms, in eV: the atomization energy, which is what
                           gpff_prepare_metadata computes for the samples
    gap_values             each molecule's gap, in eV
    elongation_values      each molecule's largest interatomic distance

The SMILES are computed by walking the database with RDKit (slow, one line per
molecule including the empty lines of invalid ones, so positions still line up
with a split file). --targets picks which assets to build (default: all).

The run stops before doing any work if a requested array can be neither read
nor derived, and writes no file until every requested asset is built.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.units import Hartree
from tqdm import tqdm

import gpff  # noqa: F401
from gpff.analysis import analyze_with_rdkit, compute_elongation

# assets read from the reference database's metadata or derived from its
# molecules, saved as .npy
METADATA_ARRAY_TARGETS = (
    "energy_U0_values",
    "gap_values",
    "elongation_values",
    "energy_U0_atomrefs_eV",
)
# the ones holding one value per molecule, in database order
PER_MOLECULE_TARGETS = ("energy_U0_values", "gap_values", "elongation_values")
# the asset computed by walking the database, saved as one SMILES per line
SMILES_TARGET = "smiles_values"
TARGETS = (*METADATA_ARRAY_TARGETS, SMILES_TARGET)

# the units the derived arrays are converted from, those of schnetpack's QM9
ENERGY_UNIT = "Ha"
DISTANCE_UNIT = "Ang"


def _selected(requested: list[str], known) -> list[str]:
    """Resolve a --targets selection, expanding the "all" sentinel to `known` in order."""
    return [name for name in known if "all" in requested or name in requested]


def missing_inputs(metadata: dict, n_rows: int, targets) -> list[str]:
    """Why each requested array can be neither read from `metadata` nor derived, if any can't.

    A stored per-molecule array must have one value per molecule. Deriving an
    array needs the property it comes from in Hartree (energy_U0 also needs
    atomrefs, stored in eV or in the metadata's atomrefs), or, for elongation,
    distances in Angstrom.
    """
    units = metadata.get("_property_unit_dict", {})
    has_atomrefs = "energy_U0_atomrefs_eV" in metadata or "energy_U0" in (metadata.get("atomrefs") or {})
    problems = []
    for key in (t for t in targets if t in METADATA_ARRAY_TARGETS):
        if key in metadata:
            n_values = len(metadata[key])
            if key in PER_MOLECULE_TARGETS and n_values != n_rows:
                problems.append(f"{key}: stored with {n_values} values for {n_rows} molecules")
            continue
        if key == "elongation_values":
            unit = metadata.get("_distance_unit")
            if unit != DISTANCE_UNIT:
                problems.append(f"{key}: not stored, and distances are in {unit!r}, not {DISTANCE_UNIT!r}")
            continue
        prop = "gap" if key == "gap_values" else "energy_U0"
        if units.get(prop) != ENERGY_UNIT:
            problems.append(f"{key}: not stored, and {prop} is in {units.get(prop)!r}, not {ENERGY_UNIT!r}")
        if prop == "energy_U0" and not has_atomrefs:
            problems.append(f"{key}: not stored, and there are no energy_U0 atomrefs")
    return problems


def atomrefs_eV(metadata: dict) -> np.ndarray:
    """The energy_U0 single-atom reference energies in eV, by atomic number."""
    if "energy_U0_atomrefs_eV" in metadata:
        return np.asarray(metadata["energy_U0_atomrefs_eV"], dtype=float)
    return np.asarray(metadata["atomrefs"]["energy_U0"], dtype=float) * Hartree


def _row_value(row, name: str) -> float:
    """The scalar property `name` of one database row."""
    value = np.ravel(row.data.get(name, []))
    if value.size != 1:
        raise ValueError(f"Row {row.id} has no single {name} value: {row.data.get(name)!r}")
    return float(value[0])


def derive_per_molecule_arrays(
    db_path: str | Path, keys, atomrefs: np.ndarray | None = None, verbose: bool = True
) -> dict[str, np.ndarray]:
    """The per-molecule arrays `keys`, derived in one pass over the database, in database order.

    energy_U0_values needs `atomrefs`, the single-atom reference energies in eV.
    """
    values = {key: [] for key in keys}
    with connect(db_path) as db:
        for row in tqdm(
            db.select(), total=db.count(), desc="Reading reference molecules", disable=not verbose
        ):
            if "energy_U0_values" in values:
                energy = _row_value(row, "energy_U0") * Hartree
                values["energy_U0_values"].append(energy - sum(atomrefs[z] for z in row.numbers))
            if "gap_values" in values:
                values["gap_values"].append(_row_value(row, "gap") * Hartree)
            if "elongation_values" in values:
                values["elongation_values"].append(compute_elongation(row.toatoms()))
    return {key: np.asarray(v, dtype=float) for key, v in values.items()}


def build_metadata_arrays(
    db_path: str | Path, metadata: dict, targets, verbose: bool = True
) -> dict[str, np.ndarray]:
    """The requested metadata-array targets, read from `metadata` if stored there, else derived."""
    keys = [key for key in targets if key in METADATA_ARRAY_TARGETS]
    arrays = {key: np.asarray(metadata[key], dtype=float) for key in keys if key in metadata}
    if "energy_U0_atomrefs_eV" in keys and "energy_U0_atomrefs_eV" not in arrays:
        arrays["energy_U0_atomrefs_eV"] = atomrefs_eV(metadata)
    to_derive = [key for key in keys if key not in arrays]
    if to_derive:
        atomrefs = atomrefs_eV(metadata) if "energy_U0_values" in to_derive else None
        arrays.update(derive_per_molecule_arrays(db_path, to_derive, atomrefs, verbose=verbose))
    return {key: arrays[key] for key in keys}


def compute_reference_smiles(db_path: str | Path, timeout: int = 5, verbose: bool = True) -> list[str]:
    """The isomeric SMILES of every molecule, in database order (empty if invalid).

    Walks the whole database, so it is slow. Invalid molecules keep an empty
    string so positions still line up with a split file.
    """
    smiles = []
    with connect(db_path) as db:
        for row in tqdm(
            db.select(), total=db.count(), desc="Computing reference SMILES", disable=not verbose
        ):
            smiles.append(analyze_with_rdkit(row.toatoms(), timeout=timeout)["smiles"])
    return smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference_db_path", required=True, type=Path)
    parser.add_argument("--assets_dir", type=Path, default=Path("assets"),
                        help="Where the asset files are written. Default: assets/.")
    parser.add_argument("--prefix", default="qm9",
                        help="Filename prefix of the assets. Default: qm9.")
    parser.add_argument("--targets", nargs="+", choices=["all", *TARGETS], default=["all"],
                        metavar="TARGET",
                        help=f"Assets to build: {', '.join(TARGETS)}. Default: all.")
    parser.add_argument("--timeout", type=int, default=5,
                        help="Per-molecule RDKit timeout for smiles_values.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.reference_db_path.exists():
        print(f"ERROR: --reference_db_path not found: {args.reference_db_path}", file=sys.stderr)
        return 1

    targets = _selected(args.targets, TARGETS)
    with connect(args.reference_db_path) as db:
        metadata = db.metadata
        n_rows = db.count()
    missing = missing_inputs(metadata, n_rows, targets)
    if missing:
        print(f"ERROR: cannot build these assets from {args.reference_db_path}:", file=sys.stderr)
        for problem in missing:
            print(f"  {problem}", file=sys.stderr)
        return 1

    verbose = not args.quiet
    print(f"Reference DB : {args.reference_db_path}")
    print(f"Assets       : {', '.join(targets)}")
    print(f"Assets dir   : {args.assets_dir}")

    # build everything before writing anything, so that a run failing in the
    # slow SMILES walk leaves the asset files of an earlier run as they were
    arrays = build_metadata_arrays(args.reference_db_path, metadata, targets, verbose=verbose)
    smiles = None
    if SMILES_TARGET in targets:
        smiles = compute_reference_smiles(args.reference_db_path, timeout=args.timeout, verbose=verbose)

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    for key, values in arrays.items():
        out_path = args.assets_dir / f"{args.prefix}_{key}.npy"
        np.save(out_path, values)
        print(f"  wrote {out_path}")

    if smiles is not None:
        out_path = args.assets_dir / f"{args.prefix}_smiles_values.txt"
        out_path.write_text("\n".join(smiles) + "\n")
        print(f"  wrote {out_path} ({len(smiles)} lines)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
