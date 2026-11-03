"""GEOM-Drugs, the MiDi split, cleaned with the GEOM-Drugs Revisited script.

One build writes one ASE database that holds every cleaned conformer once, and
one split file per *view* of it:

    geom_drugs.db     one conformer per row, hydrogens included, SMILES, the
                      molecule's id and the conformer's rank as key-value pairs
    split_all.npz     train_idx / val_idx / test_idx: every conformer
    split_top5.npz    the same, only the first 5 conformers of each molecule

A view is only a shorter list of rows, so the conformers are stored once. The
source is the MiDi split (train/val/test_data.pickle, https://doi.org/10.5281/zenodo.19150583):
for each molecule its SMILES and a list of RDKit molecules, one per conformer.
Each molecule is cleaned by the script of "GEOM-Drugs Revisited" (Nikitin et
al. 2025), vendored unchanged as ``geom_revisited_upstream``; its MIT licence is
kept next to it. A molecule is removed if its SMILES does not parse, no conformer
survives sanitization and kekulization, or any surviving conformer fails the
bond-length check; conformers that fail sanitization or are disconnected are
dropped one by one. Only atomic numbers and positions are stored: the bonds are
not.

The conformers of a molecule keep the order of the pickle, and ``conf_rank``
counts them from 1 after the cleaning. The pickle's order is the order of the
source's conformer lists; that it is the energy order is the claim of the MiDi
paper ("the 5 lowest energy conformations"), which this code does not check.
"""
from __future__ import annotations

import hashlib
import pickle
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.db import connect
from rdkit import Chem, rdBase
from tqdm import tqdm

from gpff.data.datasets import geom_revisited_upstream as upstream

__all__ = [
    "RawMol",
    "build_geom_drugs_revisited",
    "check_upstream",
    "clean_molecule",
    "file_digests",
    "load_records",
    "split_file_name",
    "DATABASE",
    "SOURCE_FILES",
    "SPLITS",
    "UPSTREAM",
    "ZENODO_DOI",
    "ZENODO_MD5",
]

DATABASE = "geom_drugs.db"
SPLITS = ("train", "val", "test")
SOURCE_FILES = {split: f"{split}_data.pickle" for split in SPLITS}

ZENODO_DOI = "10.5281/zenodo.19150583"
# the MD5 of the three files of that record
ZENODO_MD5 = {
    "train_data.pickle": "0e5a6fa70ff13e72d99ce5d081065a15",
    "val_data.pickle": "10fc3a99a74720b8b0cb7910779e23af",
    "test_data.pickle": "53f9c0c6195daf06f904619bd314a9c3",
}

# the cleaning script ``geom_revisited_upstream`` is a copy of
UPSTREAM = {
    "repository": "https://github.com/isayevlab/geom-drugs-3dgen-evaluation",
    "commit": "bfc5bed26c0ccfae0a2edba043735d50e3433c31",
    "file": "scripts/geom_drugs_processing.py",
    "sha256": "a957740a853deb373e977e6073325fcdabac5f2ec901bc1735b831285a496b92",
}

# what the build counts per split; molecules are removed for one of three reasons
# (see clean_molecule)
STATISTICS = (
    "molecules_read",
    "molecules_kept",
    "molecules_removed_smiles",
    "molecules_removed_no_conformer",
    "molecules_removed_topology",
    "conformers_read",
    "conformers_kept",
)

class RawMol:
    """An RDKit molecule as the pickle holds it: its binary string. It becomes a
    molecule only when it is cleaned, so that a split takes about the size of its
    file in memory instead of some 15 times that (RDKit molecules are large)."""

    __slots__ = ("binary",)

    def __init__(self, binary: bytes):
        self.binary = binary

    def __setstate__(self, state):
        pass  # the pickle's molecule properties (it has none) are not used

    def mol(self) -> Chem.Mol:
        return Chem.Mol(self.binary)


# What the pickles may contain: lists, tuples and strings, and RDKit molecules.
# Anything else is refused, because unpickling runs whatever the file names.
_MOLECULE = ("rdkit.Chem.rdchem", "Mol")
_ALLOWED_BUILTINS = {"list", "tuple", "dict", "set", "frozenset", "str", "bytes", "int", "float", "bool"}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == _MOLECULE:
            return RawMol
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def split_file_name(top_k: Optional[int]) -> str:
    """The split file of the view with the first ``top_k`` conformers of each
    molecule; None is the view with all of them."""
    return "split_all.npz" if top_k is None else f"split_top{top_k}.npz"


def check_upstream() -> None:
    """Stop if the vendored cleaning script is not the one UPSTREAM names."""
    digest = hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()
    if digest != UPSTREAM["sha256"]:
        raise RuntimeError(
            f"geom_revisited_upstream.py has sha256 {digest}, not the {UPSTREAM['sha256']} of "
            f"{UPSTREAM['file']} at {UPSTREAM['commit']}; it must stay a copy of that file."
        )


def file_digests(path) -> Dict:
    """md5 (what Zenodo lists), sha256 and the size of a file, read once."""
    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest(), "bytes": size}


def load_records(path) -> list:
    """The (SMILES, [RawMol, ...]) entries of a split pickle."""
    with open(path, "rb", buffering=1 << 24) as file:
        records = _RestrictedUnpickler(file).load()
    if not isinstance(records, list):
        raise ValueError(f"{path} holds a {type(records).__name__}, not a list of molecules.")
    return records


def clean_molecule(smiles: str, mols: Sequence) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]], Optional[str]]:
    """What the upstream ``process_geom_drugs`` does to one entry of a split.

    Returns the (atomic numbers, positions) of the conformers that stay, in
    order, and None as the reason, or None and why the molecule is removed:
    ``smiles`` (does not parse), ``no_conformer`` (none survives sanitization,
    kekulization and the single-fragment check) or ``topology`` (a surviving
    conformer has a bond that is too long or too short).
    """
    if Chem.MolFromSmiles(smiles) is None:
        return None, "smiles"
    mols = [mol.mol() if isinstance(mol, RawMol) else mol for mol in mols]
    sanitized = [upstream.process_molecule(mol) for mol in mols]
    sanitized = [mol for mol in sanitized if mol is not None]
    if not sanitized:
        return None, "no_conformer"
    if not all(upstream.validate_topology(mol) for mol in sanitized):
        return None, "topology"
    conformers = []
    for mol in sanitized:
        numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
        for conformer in mol.GetConformers():
            conformers.append((numbers, np.asarray(conformer.GetPositions(), dtype=np.float64)))
    return conformers, None


# the entries of the split being cleaned; set before the workers are forked
_RECORDS: list = []


def _clean_index(index: int):
    smiles, mols = _RECORDS[index]
    conformers, reason = clean_molecule(smiles, mols)
    return conformers, reason, len(mols)


def _cleaned(n: int, n_workers: int) -> Iterator:
    """The results for the entries 0..n-1 in order, so that a build does not
    depend on the number of workers."""
    if n_workers <= 1:
        for index in range(n):
            yield _clean_index(index)
            _RECORDS[index] = None  # the molecule is not needed again
    else:
        with get_context("fork").Pool(n_workers) as pool:
            yield from pool.imap(_clean_index, range(n), chunksize=32)


def build_geom_drugs_revisited(
    source_dir,
    target_dir,
    top_k: int = 5,
    max_molecules: Optional[int] = None,
    n_workers: int = 1,
    known_md5: Optional[Dict[str, str]] = None,
    progress: bool = True,
) -> Dict:
    """Write the cleaned MiDi split of GEOM-Drugs to one database and two splits.

    Args:
        source_dir: directory of train_data.pickle, val_data.pickle and
            test_data.pickle.
        target_dir: existing, empty directory to write geom_drugs.db,
            split_all.npz and split_top<top_k>.npz into.
        top_k: the second view has the first top_k conformers of each molecule,
            after the cleaning and in every split.
        max_molecules: read only the first entries of each split, for trial builds.
        n_workers: processes that clean molecules; the build is the same for any number.
        known_md5: MD5 the source files must have, by file name (ZENODO_MD5 for
            the files of the record); None accepts any files.
        progress: show a progress bar.

    Returns:
        Counts, the cleaning statistics and the digests of the source files.
    """
    global _RECORDS
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    check_upstream()
    source_dir, target_dir = Path(source_dir), Path(target_dir)

    sources = {}
    for filename in SOURCE_FILES.values():
        digests = file_digests(source_dir / filename)
        if known_md5 is not None and digests["md5"] != known_md5.get(filename):
            raise ValueError(
                f"{source_dir / filename} has md5 {digests['md5']}, not {known_md5.get(filename)}; "
                "refusing to unpickle it."
            )
        sources[filename] = digests

    db_path = target_dir / DATABASE
    if db_path.exists():
        raise FileExistsError(f"There already is a database at {db_path}.")

    rows_all, rows_top, molecules = {}, {}, {}
    cleaning, elements = {}, Counter()
    n_rows = n_molecules = 0
    rdBase.DisableLog("rdApp.*")  # RDKit warns about every molecule it sanitizes
    try:
        with connect(db_path, use_lock_file=False) as db:
            for split, filename in SOURCE_FILES.items():
                _RECORDS = load_records(source_dir / filename)
                if max_molecules is not None:
                    del _RECORDS[max_molecules:]
                smiles_of = [smiles for smiles, _ in _RECORDS]

                stats = Counter({key: 0 for key in STATISTICS})
                stats["molecules_read"] = len(_RECORDS)
                first_row, top_rows, kept_smiles = n_rows, [], set()
                results = _cleaned(len(_RECORDS), n_workers)
                with tqdm(total=len(_RECORDS), desc=f"Cleaning {split}", disable=not progress) as bar:
                    for index, (conformers, reason, n_read) in enumerate(results):
                        bar.update(1)
                        stats["conformers_read"] += n_read
                        if conformers is None:
                            stats[f"molecules_removed_{reason}"] += 1
                            continue
                        stats["molecules_kept"] += 1
                        stats["conformers_kept"] += len(conformers)
                        kept_smiles.add(smiles_of[index])
                        for rank, (numbers, positions) in enumerate(conformers, start=1):
                            db.write(
                                Atoms(numbers=numbers, positions=positions),
                                smiles=smiles_of[index],
                                mol_id=n_molecules,
                                conf_rank=rank,
                            )
                            if rank <= top_k:
                                top_rows.append(n_rows)
                            elements.update(numbers.tolist())
                            n_rows += 1
                        n_molecules += 1
                _RECORDS = []

                rows_all[split] = np.arange(first_row, n_rows, dtype=np.int64)
                rows_top[split] = np.array(top_rows, dtype=np.int64)
                molecules[split] = kept_smiles
                cleaning[split] = dict(sorted(stats.items()))

            db.metadata = {
                "protocol": "geom_drugs_revisited",
                "source_doi": ZENODO_DOI,
                "top_k": top_k,
                "max_molecules": max_molecules,
                "hydrogens": "kept",
                "_property_unit_dict": {},
                "_distance_unit": "Ang",
            }
    finally:
        rdBase.EnableLog("rdApp.*")

    for top, name in ((None, split_file_name(None)), (top_k, split_file_name(top_k))):
        rows = rows_all if top is None else rows_top
        np.savez(target_dir / name, **{f"{split}_idx": rows[split] for split in SPLITS})

    def view(rows):
        return {f"n_{split}": len(rows[split]) for split in SPLITS}

    return {
        "n_rows": n_rows,
        **view(rows_all),
        **{f"n_{split}_molecules": len(molecules[split]) for split in SPLITS},
        "molecules_shared_train_val": len(molecules["train"] & molecules["val"]),
        "molecules_shared_train_test": len(molecules["train"] & molecules["test"]),
        "molecules_shared_val_test": len(molecules["val"] & molecules["test"]),
        "views": {
            "all": {"split_file": split_file_name(None), **view(rows_all)},
            f"top{top_k}": {"split_file": split_file_name(top_k), "top_k": top_k, **view(rows_top)},
        },
        "elements": {int(z): n for z, n in sorted(elements.items())},
        "cleaning": cleaning,
        "sources": sources,
    }
