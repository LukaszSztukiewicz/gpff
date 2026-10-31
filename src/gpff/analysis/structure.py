"""Analysis of single structures (one ase.Atoms in, one result out)."""
from functools import lru_cache
from io import StringIO
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
    "get_rdkit_validity",
    "compute_elongation",
    "get_posebusters_validity",
]


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
        "elongation": np.nan
    }


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
        # the alarm can also go off just after the analysis, e.g. while an
        # RDKit error is being handled, where _analyze_with_rdkit cannot catch it
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

        return {
            "rdkit_valid": valid,
            "ring_counts": ring_counts,
            "smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "elongation": elongation
        }

    except Exception:
        # Any other exception, return invalid
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


def get_posebusters_validity(atoms: Atoms) -> bool:
    """Run the PoseBusters "mol" checks on a single ASE Atoms object.

    Returns True only if the geometry converts to an RDKit mol and all
    PoseBusters checks pass.
    """
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


