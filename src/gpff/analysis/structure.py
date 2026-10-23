"""Analysis of single structures (one ase.Atoms in, one result out)."""
from functools import lru_cache
from io import StringIO
import itertools
import multiprocessing
import signal
from typing import Any, Dict

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.rdBase import BlockLogs

__all__ = [
    "analyze_with_rdkit",
    "describe_angles",
    "describe_bonds",
    "get_rdkit_validity",
    "compute_elongation",
    "get_posebusters_validity",
]

# SMILES bond symbols, so a bond type reads like C-C, C=O or C:C
_BOND_SYMBOLS = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
    Chem.BondType.AROMATIC: ":",
}
_HYBRIDIZATIONS = {
    Chem.HybridizationType.SP: "sp",
    Chem.HybridizationType.SP2: "sp2",
    Chem.HybridizationType.SP3: "sp3",
}

# fork, so that a timed PoseBusters check costs no interpreter start-up and no
# import of posebusters; it works whatever start method the caller's pool uses.
# Python warns that forking a process with threads (torch is imported) can
# deadlock the child, but a deadlocked child runs out of time like a slow one
# and its molecule counts as invalid, so no job hangs on it.
_FORK = multiprocessing.get_context("fork")


def _no_bonds() -> Dict[str, list]:
    return {"bond_type": [], "environment": [], "length": []}


def _no_angles() -> Dict[str, list]:
    return {"angle_type": [], "environment": [], "angle": []}


def _bond_order(atom: Chem.Atom) -> tuple:
    """Sort key putting H last, then by element and hybridization."""
    return (
        atom.GetAtomicNum() == 1,
        atom.GetAtomicNum(),
        _HYBRIDIZATIONS.get(atom.GetHybridization(), ""),
    )


class _RDKitTimeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _RDKitTimeout()


def _invalid_analysis() -> Dict[str, Any]:
    return {
        "rdkit_valid": False,
        "ring_counts": np.array([np.nan, np.nan, np.nan, np.nan]),
        "smiles": "",
        "canonical_smiles": "",
        "elongation": np.nan,
        "bonds": _no_bonds(),
        "angles": _no_angles(),
    }


def describe_bonds(mol: Chem.Mol) -> Dict[str, list]:
    """The type, chemical environment and length of every bond of a sanitized mol.

    The type is the two elements and the bond order, written like SMILES (C-C,
    C=O, C:C, C-H). The environment adds the hybridization of both heavy atoms
    and whether the bond sits in a 3- or 4-membered ring, e.g. C(sp2)-C(sp3) or
    C(sp3)-C(sp3) r3. One bond type mixes environments whose lengths differ by
    several hundredths of an Angstrom (a C-C single bond is ~1.53 A between two
    sp3 carbons, ~1.46 A between two conjugated sp2 ones), while the lengths of
    one environment gather around a single value. Both labels come from the
    bond graph, never from the length. H is listed last, the other atoms by
    atomic number, then hybridization.

    Args:
        mol: RDKit molecule with bonds, a conformer and perceived hybridization
            (e.g. after DetermineBonds)

    Returns:
        Dictionary of three equally long lists: bond_type, environment and
        length (in Angstroms), one entry per bond
    """
    positions = mol.GetConformer().GetPositions()
    bonds = _no_bonds()
    for bond in mol.GetBonds():
        atoms = sorted((bond.GetBeginAtom(), bond.GetEndAtom()), key=_bond_order)
        symbol = _BOND_SYMBOLS.get(bond.GetBondType(), "~")
        labels = []
        for atom in atoms:
            if atom.GetAtomicNum() == 1:
                labels.append("H")
            else:
                hybridization = atom.GetHybridization()
                name = _HYBRIDIZATIONS.get(hybridization, str(hybridization).lower())
                labels.append(f"{atom.GetSymbol()}({name})")
        environment = symbol.join(labels)
        # small rings strain their bonds; the smallest ring decides
        for ring_size in (3, 4):
            if bond.IsInRingSize(ring_size):
                environment += f" r{ring_size}"
                break

        i, j = (atom.GetIdx() for atom in atoms)
        bonds["bond_type"].append(symbol.join(atom.GetSymbol() for atom in atoms))
        bonds["environment"].append(environment)
        bonds["length"].append(float(np.linalg.norm(positions[i] - positions[j])))
    return bonds


def describe_angles(mol: Chem.Mol) -> Dict[str, list]:
    """The type, chemical environment and size of every bond angle of a sanitized mol.

    One angle per pair of bonds meeting at an atom. The type is the three
    elements with the bond symbols and the central atom in the middle (C-C=O,
    H-C-H). The environment adds the central atom's hybridization, which is what
    sets the angle it prefers (109.5, 120 or 180 degrees), and whether both
    bonds lie in the same 3- or 4-membered ring, which forces it far below that.
    The two arms are ordered as describe_bonds orders a bond's atoms.

    Args:
        mol: RDKit molecule with bonds, a conformer and perceived hybridization
            (e.g. after DetermineBonds)

    Returns:
        Dictionary of three equally long lists: angle_type, environment and
        angle (in degrees), one entry per angle
    """
    positions = mol.GetConformer().GetPositions()
    angles = _no_angles()
    for center in mol.GetAtoms():
        arms = sorted(
            ((bond, bond.GetOtherAtom(center)) for bond in center.GetBonds()),
            key=lambda arm: _bond_order(arm[1]),
        )
        hybridization = center.GetHybridization()
        name = _HYBRIDIZATIONS.get(hybridization, str(hybridization).lower())
        for (first, left), (second, right) in itertools.combinations(arms, 2):
            symbols = [_BOND_SYMBOLS.get(bond.GetBondType(), "~") for bond in (first, second)]
            # the arms read outwards from the central atom, left one first
            angle_type = (
                f"{left.GetSymbol()}{symbols[0]}{center.GetSymbol()}"
                f"{symbols[1]}{right.GetSymbol()}"
            )
            environment = (
                f"{left.GetSymbol()}{symbols[0]}{center.GetSymbol()}({name})"
                f"{symbols[1]}{right.GetSymbol()}"
            )
            # an angle inside a small ring is the ring's, not the hybridization's
            for ring_size in (3, 4):
                if first.IsInRingSize(ring_size) and second.IsInRingSize(ring_size):
                    environment += f" r{ring_size}"
                    break

            left_arm = positions[left.GetIdx()] - positions[center.GetIdx()]
            right_arm = positions[right.GetIdx()] - positions[center.GetIdx()]
            cosine = np.dot(left_arm, right_arm) / (
                np.linalg.norm(left_arm) * np.linalg.norm(right_arm)
            )
            angles["angle_type"].append(angle_type)
            angles["environment"].append(environment)
            angles["angle"].append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


def analyze_with_rdkit(
    atoms: Atoms,
    timeout: int = 5,
    remove_hs: bool = False,
) -> Dict[str, Any]:
    """
    Check if an ASE Atoms object is chemically valid and compute ring statistics.

    A molecule whose analysis takes longer than ``timeout`` seconds is invalid.
    See ``_analyze_with_rdkit`` for the returned dictionary.
    """
    try:
        return _analyze_with_rdkit(atoms, timeout, remove_hs)
    except _RDKitTimeout:
        # the alarm can also go off just after the analysis, e.g. while an RDKit
        # error is being handled, where _analyze_with_rdkit cannot catch it
        signal.alarm(0)
        return _invalid_analysis()


def _analyze_with_rdkit(
    atoms: Atoms,
    timeout: int,
    remove_hs: bool,
) -> Dict[str, Any]:
    """
    Check if an ASE Atoms object is chemically valid and compute ring statistics.

    This function replicates the logic from the GetSmiles transform used in the
    check_validity.py script.

    Args:
        atoms: ASE Atoms object to check
        timeout: Timeout in seconds (0 to disable)

    Returns:
        Dictionary with keys:
        - rdkit_valid: Boolean indicating if the molecule is chemically valid
        - ring_counts: Array with counts of 3-, 4-, 5-, and 6-membered rings
        - smiles: isomeric SMILES of the molecule, i.e. with stereochemistry
          (empty if invalid)
        - elongation: Maximum pairwise distance in Angstroms (np.nan if invalid)
        - bonds: describe_bonds of the molecule with its hydrogens, also with
          remove_hs (empty lists if invalid)
        - angles: describe_angles of the same molecule (empty lists if invalid)
    """

    # Set up timeout if specified
    if timeout > 0:
        old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(timeout)

    # analyze molecule
    try:
        # Block RDKit logs
        with BlockLogs():
            # Convert Atoms to XYZ format
            tmp = StringIO()
            write(tmp, atoms, format="xyz")
            tmp.seek(0)
            xyz = "".join(tmp)
            mol = Chem.MolFromXYZBlock(xyz)

            # Try to find valid structure
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=0,
                allowChargedFragments=False,
                embedChiral=True,
            )
            # before the hydrogens go, so the X-H bonds and angles are described too
            bonds = describe_bonds(mol)
            angles = describe_angles(mol)
            if remove_hs:
                mol = Chem.RemoveHs(mol)

            # Check for radical electrons (not allowed by default)
            radical_found = False
            for at in mol.GetAtoms():
                if at.GetNumRadicalElectrons() > 0:
                    radical_found = True
                    break

            # the isomeric SMILES keeps stereochemistry; it stays empty for
            # radicals, which novelty and uniqueness then never count.
            smiles = ""
            canonical_smiles = ""
            # if no radical electrons are found, get smiles
            if not radical_found:
                smiles = Chem.MolToSmiles(mol)
                canonical_smiles = Chem.CanonSmiles(smiles)
            # Determine validity
            # invalid if smiles could be found and no fragments exist
            if canonical_smiles != "" and "." not in canonical_smiles:
                valid = True
            else:
                valid = False

            # get other statistics
            if valid:
                # get ring counts
                sssr = Chem.GetSymmSSSR(mol)
                rings = [len(sssr[i]) for i in range(len(sssr))]
                n_rings = torch.bincount(
                    torch.tensor(rings, dtype=torch.int), minlength=7
                )
                ring_counts = np.array([n_rings[3], n_rings[4], n_rings[5], n_rings[6]])

                # get elongation
                elongation = compute_elongation(atoms)
            else:
                ring_counts = np.array([np.nan, np.nan, np.nan, np.nan])
                elongation = np.nan
                bonds = _no_bonds()
                angles = _no_angles()

        return {
            "rdkit_valid": valid,
            "ring_counts": ring_counts,
            "smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "elongation": elongation,
            "bonds": bonds,
            "angles": angles,
        }

    except Exception:
        # Any other exception (the timeout among them), return invalid
        return _invalid_analysis()

    finally:
        # Clean up timeout
        if timeout > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def get_rdkit_validity(
    atoms: Atoms,
    timeout: int = 5,
    remove_hs: bool = False,
) -> bool:
    """
    Check if an ASE Atoms object is chemically valid.
    """
    return analyze_with_rdkit(atoms, timeout, remove_hs)["rdkit_valid"]


@lru_cache(maxsize=None)
def _posebusters_config() -> Dict[str, Any]:
    """The PoseBusters "mol" config, read from its YAML file once."""
    return PoseBusters(config="mol").config


def get_posebusters_validity(atoms: Atoms, timeout: float = 0) -> bool:
    """Run the PoseBusters "mol" checks on a single ASE Atoms object.

    Returns True only if the geometry converts to an RDKit mol and all
    PoseBusters checks pass.

    Args:
        atoms: ASE Atoms object to check
        timeout: seconds one check may take, 0 for no limit. A check over the
            limit counts as invalid, as an RDKit timeout does. PoseBusters has
            no limit of its own and a few generated structures keep it busy for
            hours inside RDKit's C++ code, where the SIGALRM of
            analyze_with_rdkit never arrives, so the check runs in a forked
            child that is killed when its time is up.
    """
    if timeout <= 0:
        return _posebusters_validity(atoms)

    receiver, sender = _FORK.Pipe(duplex=False)
    child = _FORK.Process(target=_posebusters_child, args=(atoms, sender), daemon=True)
    child.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            child.kill()
            return False
        return bool(receiver.recv())
    except EOFError:  # the child died without an answer
        return False
    finally:
        receiver.close()
        child.join()


def _posebusters_child(atoms: Atoms, sender) -> None:
    try:
        sender.send(_posebusters_validity(atoms))
    except Exception:
        sender.send(False)
    finally:
        sender.close()


def _posebusters_validity(atoms: Atoms) -> bool:
    try:
        with BlockLogs():
            tmp = StringIO()
            write(tmp, atoms, format="xyz")
            mol = Chem.MolFromXYZBlock(tmp.getvalue())
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=0,
                allowChargedFragments=False,
                embedChiral=True,
            )
    except Exception:
        return False

    # a fresh instance per molecule: posebusters 0.2.12 never clears the results
    # it keeps on the instance, keyed by str(mol), i.e. the Mol's reused address,
    # so a shared one piles every molecule's entries onto a few keys. Only the
    # config loading is shared.
    results = PoseBusters(config=dict(_posebusters_config())).bust(mol)
    return bool(results.all(axis=1).iloc[0])


def compute_elongation(atoms: Atoms) -> float:
    """
    Calculate the maximum pairwise distance between any two atoms in a molecule.

    Args:
        atoms: ASE Atoms object

    Returns:
        float: Maximum pairwise distance in Angstroms
    """
    if len(atoms) < 2:
        return 0.0

    # Get all pairwise distances
    distances = atoms.get_all_distances()

    # Get the maximum distance (excluding diagonal elements which are 0)
    # Use upper triangle to avoid counting each pair twice
    upper_triangle = np.triu(distances, k=1)
    max_distance = np.max(upper_triangle)

    return max_distance


