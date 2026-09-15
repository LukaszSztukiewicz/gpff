"""Generate the reference asset files from the reference database (``gpff_reference_assets``).

The evaluation reads its reference data from files in assets/ rather than from a
reference database (see ``gpff_calculate_metrics``). This script produces those
files from the reference QM9 database so they can be rebuilt from scratch:

    <prefix>_energy_U0_values.npy       reference energy_U0 distribution
    <prefix>_gap_values.npy             reference gap distribution
    <prefix>_elongation_values.npy      reference elongation distribution
    <prefix>_energy_U0_atomrefs_eV.npy  single-atom reference energies
    <prefix>_smiles_values.txt          one SMILES per molecule, in database order

Energy convention: ``energy_U0_values`` is the **atomization energy in eV**,
i.e. ``row_energy_U0 - sum(atomrefs_eV[z] for z in row atoms)``, not the raw
total U0 and not Hartree. ``gpff_prepare_metadata.predict_property`` computes
the same quantity for generated samples (model energy_U0 minus the atomref
sum, using the atomrefs exported here), and ``js_divergence_energy_U0``
compares the two distributions, so both sides must stay in this convention.

Each array target is either read straight from a precomputed array already
stored in the database metadata under its own key (used unchanged, in
whatever units it already carries), or, if absent, derived from the raw row
properties, geometry and atomrefs declared in the database. Raw property and
atomref values are read in the unit declared by the database's own
``_property_unit_dict`` (only ``"Ha"`` is supported for derivation) and
converted with ``ase.units.Hartree``. --targets picks which assets to build
(default: all); only the dependencies of the requested targets are resolved,
and all requested targets are validated before any file is written.

The SMILES asset is normally computed by walking the database with RDKit
(slow, one line per molecule including the empty lines of invalid ones, so
positions still line up with a split file). If the database was written with
the ``gpff_explicit_h_isomeric_v1`` ``smiles_format`` marker and every row
carries a string-valued ``smiles`` key-value pair (matching
``gpff_prepare_metadata``'s sample-side SMILES), those stored strings are
reused instead of recomputing them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase.db import connect
from ase.units import Hartree
from tqdm import tqdm

import gpff  # noqa: F401
from gpff.analysis import analyze_with_rdkit, compute_elongation

# assets read from the reference database's metadata (precomputed) or derived
# from its row properties/geometry, saved as .npy
METADATA_ARRAY_TARGETS = (
    "energy_U0_values",
    "gap_values",
    "elongation_values",
    "energy_U0_atomrefs_eV",
)
# the asset read from stored SMILES key-value pairs or computed by walking the
# database, saved as one SMILES per line
SMILES_TARGET = "smiles_values"
TARGETS = (*METADATA_ARRAY_TARGETS, SMILES_TARGET)

# marks a filtered database's `smiles` column as the isomeric, explicit-hydrogen
# string from analyze_with_rdkit(remove_hs=False) -- the same call and key
# gpff_prepare_metadata stores for generated samples.
SMILES_FORMAT_MARKER = "gpff_explicit_h_isomeric_v1"


class AssetResolutionError(ValueError):
    """An expected, actionable problem with the reference database's data."""


def _selected(requested: list[str], known) -> list[str]:
    """Resolve a --targets selection, expanding the "all" sentinel to `known` in order."""
    return [name for name in known if "all" in requested or name in requested]


def _validate_precomputed_array(
    raw: Any, key: str, db_path: str | Path, expected_length: int
) -> np.ndarray:
    """Validate a metadata array already meant as final output: finite, 1-D, the right length."""
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssetResolutionError(
            f"Precomputed '{key}' in {db_path} is not a numeric array: {exc}"
        ) from exc
    if arr.ndim != 1:
        raise AssetResolutionError(
            f"Precomputed '{key}' in {db_path} must be one-dimensional; got shape {arr.shape}."
        )
    if arr.shape[0] != expected_length:
        raise AssetResolutionError(
            f"Precomputed '{key}' in {db_path} has length {arr.shape[0]}, "
            f"expected {expected_length}."
        )
    if not np.all(np.isfinite(arr)):
        raise AssetResolutionError(f"Precomputed '{key}' in {db_path} contains non-finite values.")
    return arr


def _require_hartree_unit(metadata: dict, property_name: str, db_path: str | Path) -> None:
    """Require `property_name` to be declared in Hartree; raise a named, actionable error."""
    unit_dict = metadata.get("_property_unit_dict", {})
    if property_name not in unit_dict:
        raise AssetResolutionError(
            f"No unit declared for '{property_name}' in {db_path}'s _property_unit_dict; "
            "cannot derive from the raw value without it."
        )
    unit = unit_dict[property_name]
    if unit != "Ha":
        raise AssetResolutionError(
            f"Cannot derive '{property_name}' from {db_path}: declared unit is "
            f"{unit!r}, expected 'Ha'."
        )


def _read_row_property(row, property_name: str, db_path: str | Path) -> float:
    """One finite scalar from `row.data[property_name]`, or a named, actionable error."""
    if property_name not in row.data:
        raise AssetResolutionError(
            f"Row {row.id} in {db_path} has no '{property_name}' property."
        )
    raw_value = row.data[property_name]
    try:
        value = np.asarray(raw_value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        value = np.array([])
    if value.size != 1 or not np.isfinite(value[0]):
        raise AssetResolutionError(
            f"Row {row.id} in {db_path} has a malformed '{property_name}' value: "
            f"{raw_value!r}; expected exactly one finite number."
        )
    return float(value[0])


def _derive_atomrefs_eV(db_path: str | Path, metadata: dict) -> np.ndarray:
    """Convert the database's raw `atomrefs["energy_U0"]` (Hartree) to eV."""
    atomrefs = metadata.get("atomrefs") or {}
    energy_atomrefs = atomrefs.get("energy_U0")
    if not energy_atomrefs:
        raise AssetResolutionError(
            f"No 'atomrefs' -> 'energy_U0' entry in {db_path}'s metadata; needed to "
            "derive energy_U0_atomrefs_eV (no precomputed array was supplied either)."
        )
    try:
        raw = np.asarray(energy_atomrefs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise AssetResolutionError(
            f"'atomrefs' -> 'energy_U0' in {db_path} is not a numeric array: {exc}"
        ) from exc
    if raw.ndim != 1 or raw.shape[0] != 100 or not np.all(np.isfinite(raw)):
        raise AssetResolutionError(
            f"'atomrefs' -> 'energy_U0' in {db_path} must be 100 finite numbers "
            f"indexed by atomic number; got shape {raw.shape}."
        )
    _require_hartree_unit(metadata, "energy_U0", db_path)
    return raw * Hartree


def _build_gap_values(db_path: str | Path, metadata: dict) -> np.ndarray:
    _require_hartree_unit(metadata, "gap", db_path)
    values = []
    with connect(db_path) as db:
        for row in db.select():
            values.append(_read_row_property(row, "gap", db_path))
    return np.asarray(values, dtype=float) * Hartree


def _build_elongation_values(db_path: str | Path, metadata: dict) -> np.ndarray:
    distance_unit = metadata.get("_distance_unit")
    if distance_unit != "Ang":
        raise AssetResolutionError(
            f"Cannot derive elongation_values from {db_path}: declared distance "
            f"unit is {distance_unit!r}, expected 'Ang'."
        )
    values = []
    with connect(db_path) as db:
        for row in db.select():
            values.append(compute_elongation(row.toatoms()))
    return np.asarray(values, dtype=float)


def _build_energy_U0_values(
    db_path: str | Path, metadata: dict, atomrefs_eV: np.ndarray
) -> np.ndarray:
    _require_hartree_unit(metadata, "energy_U0", db_path)
    values = []
    with connect(db_path) as db:
        for row in db.select():
            raw = _read_row_property(row, "energy_U0", db_path)
            total_eV = raw * Hartree
            atomic_ref_sum = 0.0
            for z in row.numbers:
                if not (0 <= z < len(atomrefs_eV)):
                    raise AssetResolutionError(
                        f"Atomic number {int(z)} in row {row.id} of {db_path} is out "
                        f"of range for the {len(atomrefs_eV)}-entry atomrefs array."
                    )
                atomic_ref_sum += atomrefs_eV[z]
            values.append(total_eV - atomic_ref_sum)
    return np.asarray(values, dtype=float)


def compute_reference_smiles(
    db_path: str | Path, timeout: int = 5, verbose: bool = True
) -> list[str]:
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


def _read_stored_smiles_if_complete(db_path: str | Path, n_rows: int) -> Optional[list[str]]:
    """Stored `smiles` key-value pairs for every row, or None if any is missing/non-string."""
    smiles = []
    with connect(db_path) as db:
        for row in db.select():
            value = row.get("smiles")
            if not isinstance(value, str):
                return None
            smiles.append(value)
    if len(smiles) != n_rows:
        return None
    return smiles


class _AssetResolver:
    """Resolves requested targets from precomputed metadata arrays or derivation.

    Only computes what a requested target actually depends on, and caches
    intermediates (notably ``energy_U0_atomrefs_eV``, a dependency of
    ``energy_U0_values``) so they are resolved once even if used twice.
    """

    def __init__(self, db_path: str | Path, metadata: dict, n_rows: int):
        self.db_path = db_path
        self.metadata = metadata
        self.n_rows = n_rows
        self._cache: dict[str, Any] = {}

    def _precomputed(self, key: str, expected_length: int) -> Optional[np.ndarray]:
        if key not in self.metadata:
            return None
        return _validate_precomputed_array(self.metadata[key], key, self.db_path, expected_length)

    def energy_U0_atomrefs_eV(self) -> np.ndarray:
        if "energy_U0_atomrefs_eV" not in self._cache:
            precomputed = self._precomputed("energy_U0_atomrefs_eV", 100)
            self._cache["energy_U0_atomrefs_eV"] = (
                precomputed
                if precomputed is not None
                else _derive_atomrefs_eV(self.db_path, self.metadata)
            )
        return self._cache["energy_U0_atomrefs_eV"]

    def energy_U0_values(self) -> np.ndarray:
        if "energy_U0_values" not in self._cache:
            precomputed = self._precomputed("energy_U0_values", self.n_rows)
            if precomputed is not None:
                self._cache["energy_U0_values"] = precomputed
            else:
                atomrefs_eV = self.energy_U0_atomrefs_eV()
                self._cache["energy_U0_values"] = _build_energy_U0_values(
                    self.db_path, self.metadata, atomrefs_eV
                )
        return self._cache["energy_U0_values"]

    def gap_values(self) -> np.ndarray:
        if "gap_values" not in self._cache:
            precomputed = self._precomputed("gap_values", self.n_rows)
            self._cache["gap_values"] = (
                precomputed if precomputed is not None else _build_gap_values(self.db_path, self.metadata)
            )
        return self._cache["gap_values"]

    def elongation_values(self) -> np.ndarray:
        if "elongation_values" not in self._cache:
            precomputed = self._precomputed("elongation_values", self.n_rows)
            self._cache["elongation_values"] = (
                precomputed
                if precomputed is not None
                else _build_elongation_values(self.db_path, self.metadata)
            )
        return self._cache["elongation_values"]

    def smiles_values(self, timeout: int, verbose: bool) -> list[str]:
        if "smiles_values" not in self._cache:
            stored = None
            if self.metadata.get("smiles_format") == SMILES_FORMAT_MARKER:
                stored = _read_stored_smiles_if_complete(self.db_path, self.n_rows)
            self._cache["smiles_values"] = (
                stored
                if stored is not None
                else compute_reference_smiles(self.db_path, timeout=timeout, verbose=verbose)
            )
        return self._cache["smiles_values"]

    def resolve(self, key: str, timeout: int, verbose: bool):
        if key == SMILES_TARGET:
            return self.smiles_values(timeout=timeout, verbose=verbose)
        return getattr(self, key)()


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
    verbose = not args.quiet

    with connect(args.reference_db_path) as db:
        metadata = dict(db.metadata)
        n_rows = db.count()

    resolver = _AssetResolver(str(args.reference_db_path), metadata, n_rows)
    try:
        resolved = {key: resolver.resolve(key, timeout=args.timeout, verbose=verbose) for key in targets}
    except AssetResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reference DB : {args.reference_db_path}")
    print(f"Assets       : {', '.join(targets)}")
    print(f"Assets dir   : {args.assets_dir}")

    for key in (t for t in targets if t in METADATA_ARRAY_TARGETS):
        out_path = args.assets_dir / f"{args.prefix}_{key}.npy"
        np.save(out_path, resolved[key])
        print(f"  wrote {out_path}")

    if SMILES_TARGET in targets:
        smiles = resolved[SMILES_TARGET]
        out_path = args.assets_dir / f"{args.prefix}_smiles_values.txt"
        text = "\n".join(smiles) + "\n" if smiles else ""
        out_path.write_text(text)
        print(f"  wrote {out_path} ({len(smiles)} lines)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
