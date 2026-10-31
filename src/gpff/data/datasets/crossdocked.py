import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from tqdm import tqdm

__all__ = ["build_crossdocked", "N_LIGAND_ATOMS"]

LIGAND_DB_FILENAME = "crossdocked_ligand.db"
COMPLEX_DB_FILENAME = "crossdocked_complex.db"
# the pocket atoms of each test complex, in the format of
# condition_generator=precomputed_scaffold
POCKET_IDS_FILENAME = "crossdocked_complex_pocket_ids.json"

# per-row property of the complex database: its first n_ligand_atoms atoms are
# the ligand, the remaining ones the pocket
N_LIGAND_ATOMS = "n_ligand_atoms"


def _heavy_atoms(mol: Chem.Mol, path: str) -> Atoms:
    """The heavy atoms of a molecule read from ``path``, in file order."""
    if mol is None:
        raise ValueError(f"RDKit cannot read {path}.")
    if mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        raise ValueError(f"{path} has no atoms.")
    keep = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not keep:
        raise ValueError(f"{path} has no heavy atoms.")
    numbers = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in keep]
    positions = mol.GetConformer().GetPositions()[keep]
    return Atoms(numbers=numbers, positions=positions)


def read_ligand(path: str) -> Atoms:
    """The heavy atoms of a ligand SDF file."""
    with BlockLogs():
        mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    return _heavy_atoms(mol, path)


def read_pocket(path: str) -> Atoms:
    """The heavy atoms of a pocket PDB file."""
    with BlockLogs():
        mol = Chem.MolFromPDBFile(
            path, sanitize=False, removeHs=False, proximityBonding=False
        )
    return _heavy_atoms(mol, path)


def _validation_pockets(
    train_pairs: List[Tuple[str, str]], n_val: int, seed: int
) -> set:
    """Whole pocket directories of the train pairs, drawn in a seeded random order
    until they hold at least ``n_val`` pairs."""
    pairs_per_pocket = Counter(ligand.split("/")[0] for _, ligand in train_pairs)
    pockets = sorted(pairs_per_pocket)
    order = np.random.default_rng(seed).permutation(len(pockets))
    chosen, n_pairs = set(), 0
    for i in order:
        if n_pairs >= n_val:
            break
        chosen.add(pockets[i])
        n_pairs += pairs_per_pocket[pockets[i]]
    return chosen


def build_crossdocked(
    root: str,
    split_path: str,
    target_dir: str,
    n_val: int = 5000,
    seed: int = 20261020,
    progress: bool = True,
) -> Dict:
    """Write the CrossDocked pairs of a split file to a ligand and a complex database.

    The source is the pocket10 subset of CrossDocked2020 with its official split
    (``split_by_name.pt``: "train" and "test" lists of (pocket PDB, ligand SDF)
    paths relative to ``root``). Only the pairs in the split are read, so the
    out-of-split poses of the test pockets never enter training. Both databases
    have one row per pair, train pairs first, then test pairs, in split file
    order, so row i of one is row i of the other:

        crossdocked_ligand.db   the ligand's heavy atoms
        crossdocked_complex.db  the ligand's heavy atoms followed by the pocket's,
                                with the ligand atom count in "n_ligand_atoms"

    Hydrogens are dropped, as almost no ligand file has them. Every row carries
    its source files as the key-value pairs "pocket_file" and "ligand_file".
    The validation split is made of whole pocket directories of the train split,
    so no validation pocket is trained on; the test split is the official one.
    A train pair with a file that cannot be read (e.g. an empty pocket) is
    skipped and listed in the metadata of both databases; a test pair stops the
    build, so the test split is always complete.

    Args:
        root: directory the paths in the split file are relative to.
        split_path: the official split file, ``split_by_name.pt``.
        target_dir: existing, empty directory to write the databases, the split
            file (split.npz) and the test pocket atom ids into.
        n_val: least number of train pairs moved to the validation split.
        seed: seed of the order in which validation pockets are drawn.
        progress: show a progress bar.

    Returns:
        Row counts and element counts of the written databases, and the skipped
        train pairs.
    """
    split = torch.load(split_path, weights_only=True)
    pairs = [("train", *pair) for pair in split["train"]] + [
        ("test", *pair) for pair in split["test"]
    ]
    val_pockets = _validation_pockets(split["train"], n_val=n_val, seed=seed)

    ligand_path = os.path.join(target_dir, LIGAND_DB_FILENAME)
    complex_path = os.path.join(target_dir, COMPLEX_DB_FILENAME)
    for path in (ligand_path, complex_path):
        if os.path.exists(path):
            raise FileExistsError(f"There already is a database at {path}.")

    partitions = {"train": [], "val": [], "test": []}
    pocket_ids, skipped = [], []
    n_rows = 0
    ligand_elements, pocket_elements = Counter(), Counter()
    with connect(ligand_path, use_lock_file=False) as ligand_db, connect(
        complex_path, use_lock_file=False
    ) as complex_db:
        for population, pocket_file, ligand_file in tqdm(
            pairs, desc="Reading CrossDocked pairs", disable=not progress
        ):
            try:
                ligand = read_ligand(os.path.join(root, ligand_file))
                pocket = read_pocket(os.path.join(root, pocket_file))
            except ValueError as error:
                if population == "test":
                    raise
                skipped.append(
                    {"pocket_file": pocket_file, "ligand_file": ligand_file, "reason": str(error)}
                )
                continue
            index, n_rows = n_rows, n_rows + 1
            ligand_elements.update(ligand.numbers.tolist())
            pocket_elements.update(pocket.numbers.tolist())

            source = {"pocket_file": pocket_file, "ligand_file": ligand_file}
            ligand_db.write(ligand, **source)
            complex_db.write(
                ligand + pocket,
                data={N_LIGAND_ATOMS: np.array([len(ligand)])},
                **source,
            )

            if population == "test":
                partitions["test"].append(index)
                pocket_ids.append(
                    {
                        "dataset_idx": index,
                        "scaffold_ids": list(range(len(ligand), len(ligand) + len(pocket))),
                    }
                )
            elif ligand_file.split("/")[0] in val_pockets:
                partitions["val"].append(index)
            else:
                partitions["train"].append(index)

        metadata = {
            "source_root": str(root),
            "split_file": str(split_path),
            "hydrogens": "removed",
            "validation_pockets": sorted(val_pockets),
            "validation_seed": seed,
            "skipped_train_pairs": skipped,
            "_distance_unit": "Ang",
        }
        ligand_db.metadata = {**metadata, "_property_unit_dict": {}}
        complex_db.metadata = {**metadata, "_property_unit_dict": {N_LIGAND_ATOMS: ""}}

    np.savez(
        os.path.join(target_dir, "split.npz"),
        **{f"{name}_idx": np.array(rows, dtype=np.int64) for name, rows in partitions.items()},
    )
    with open(os.path.join(target_dir, POCKET_IDS_FILENAME), "w") as file:
        json.dump({"molecules": pocket_ids}, file)

    return {
        "n_rows": n_rows,
        "n_skipped_train_pairs": len(skipped),
        **{f"n_{name}": len(rows) for name, rows in partitions.items()},
        "n_val_pockets": len(val_pockets),
        "ligand_elements": {int(z): n for z, n in sorted(ligand_elements.items())},
        "pocket_elements": {int(z): n for z, n in sorted(pocket_elements.items())},
        "skipped_train_pairs": skipped,
    }
