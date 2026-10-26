"""Generate the reference asset files from the reference database (``gpff_reference_assets``).

The evaluation reads its reference data from files in assets/ rather than from a
reference database (see ``gpff_calculate_metrics``). This script produces those
files from the reference QM9 database so they can be rebuilt from scratch:

    <prefix>_energy_U0_values.npy       reference energy_U0 distribution
    <prefix>_gap_values.npy             reference gap distribution
    <prefix>_elongation_values.npy      reference elongation distribution
    <prefix>_energy_U0_atomrefs_eV.npy  single-atom reference energies
    <prefix>_smiles_values.txt          one SMILES per molecule, in database order
    <prefix>_bond_lengths.npz           reference bond table: every bond's
                                        molecule, bond_type, environment, length
    <prefix>_bond_angles.npz            reference angle table: every angle's
                                        molecule, angle_type, environment, angle

The distribution and atom-reference arrays are read from the database's own
metadata; the SMILES and the two tables are computed in one walk of the database
with RDKit (slow; one SMILES line per molecule including the empty lines of
invalid ones, so positions still line up with a split file). --targets picks
which assets to build (default: all).

The run stops before doing any work if a requested array is not in the database.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.db import connect
from tqdm import tqdm

import gpff  # noqa: F401
from gpff.analysis import analyze_with_rdkit, collect_angles, collect_bonds

# assets read straight from the reference database's metadata, saved as .npy
METADATA_ARRAY_TARGETS = (
    "energy_U0_values",
    "gap_values",
    "elongation_values",
    "energy_U0_atomrefs_eV",
)
# the assets computed in one RDKit walk of the database: one SMILES per line,
# and the measurement tables as .npz
SMILES_TARGET = "smiles_values"
# target -> the analysis key it collects and its collector
TABLE_TARGETS = {
    "bond_lengths": ("bonds", collect_bonds),
    "bond_angles": ("angles", collect_angles),
}
WALK_TARGETS = (SMILES_TARGET, *TABLE_TARGETS)
TARGETS = (*METADATA_ARRAY_TARGETS, *WALK_TARGETS)


def _selected(requested: list[str], known) -> list[str]:
    """Resolve a --targets selection, expanding the "all" sentinel to `known` in order."""
    return [name for name in known if "all" in requested or name in requested]


def read_metadata_array(db_path: str | Path, key: str) -> np.ndarray:
    """One array stored in the reference database's metadata under `key`."""
    with connect(db_path) as db:
        metadata = db.metadata
    if key not in metadata:
        raise KeyError(f"No {key} array in the metadata of {db_path}.")
    return np.asarray(metadata[key], dtype=float)


def analyze_reference(
    db_path: str | Path, timeout: int = 5, verbose: bool = True
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """The isomeric SMILES and the measurement tables of every molecule, in database order.

    Walks the whole database once, so it is slow. Invalid molecules keep an
    empty SMILES, so positions still line up with a split file, and contribute
    no bonds and no angles. The tables come back keyed by their target name.
    """
    smiles = []
    measured: dict[str, list] = {target: [] for target in TABLE_TARGETS}
    with connect(db_path) as db:
        for row in tqdm(
            db.select(), total=db.count(), desc="Analyzing reference molecules", disable=not verbose
        ):
            analysis = analyze_with_rdkit(row.toatoms(), timeout=timeout)
            smiles.append(analysis["smiles"])
            for target, (key, _) in TABLE_TARGETS.items():
                measured[target].append(analysis[key])
    return smiles, {
        target: collect(measured[target]) for target, (_, collect) in TABLE_TARGETS.items()
    }


def missing_arrays(db_path: str | Path, targets) -> list[str]:
    """The requested metadata-array targets not present in the database's metadata."""
    wanted = [key for key in targets if key in METADATA_ARRAY_TARGETS]
    if not wanted:
        return []
    with connect(db_path) as db:
        metadata = db.metadata
    return [key for key in wanted if key not in metadata]


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
                        help="Per-molecule RDKit timeout for smiles_values and the tables.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.reference_db_path.exists():
        print(f"ERROR: --reference_db_path not found: {args.reference_db_path}", file=sys.stderr)
        return 1

    targets = _selected(args.targets, TARGETS)
    missing = missing_arrays(args.reference_db_path, targets)
    if missing:
        print("ERROR: arrays not in the reference database's metadata:", file=sys.stderr)
        for key in missing:
            print(f"  {key}", file=sys.stderr)
        return 1

    verbose = not args.quiet
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reference DB : {args.reference_db_path}")
    print(f"Assets       : {', '.join(targets)}")
    print(f"Assets dir   : {args.assets_dir}")

    for key in (t for t in targets if t in METADATA_ARRAY_TARGETS):
        out_path = args.assets_dir / f"{args.prefix}_{key}.npy"
        np.save(out_path, read_metadata_array(args.reference_db_path, key))
        print(f"  wrote {out_path}")

    if set(targets) & set(WALK_TARGETS):
        smiles, tables = analyze_reference(
            args.reference_db_path, timeout=args.timeout, verbose=verbose
        )
        if SMILES_TARGET in targets:
            out_path = args.assets_dir / f"{args.prefix}_smiles_values.txt"
            out_path.write_text("\n".join(smiles) + "\n")
            print(f"  wrote {out_path} ({len(smiles)} lines)")
        for target in (t for t in targets if t in TABLE_TARGETS):
            out_path = args.assets_dir / f"{args.prefix}_{target}.npz"
            np.savez(out_path, **tables[target])
            print(f"  wrote {out_path} ({len(tables[target]['molecule'])} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
