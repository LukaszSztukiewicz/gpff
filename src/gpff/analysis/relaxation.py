"""UFF relaxation and bond lengths of single structures (one ase.Atoms in, one result out)."""
from io import StringIO
from typing import Any, Dict

import numpy as np
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdForceFieldHelpers
from rdkit.rdBase import BlockLogs

__all__ = [
    "mol_from_atoms",
    "relax_with_uff",
    "get_bond_lengths",
]

BOND_SYMBOLS = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
    Chem.BondType.AROMATIC: ":",
}


def mol_from_atoms(atoms: Atoms, bond_orders: bool = True) -> Chem.Mol:
    """
    Read an ASE Atoms object into an RDKit Mol, with bonds perceived from the geometry.

    Args:
        atoms: ASE Atoms object to read.
        bond_orders: if True, perceive bonds and bond orders as the validity
            check does (neutral, no charged fragments); this raises if no
            valid assignment exists. If False, perceive the connectivity only
            (all bonds single) and sanitize without valence checks, so that
            structures with over-coordinated atoms can be read as well.

    Returns:
        The Mol, with atoms in the order of ``atoms``.
    """
    tmp = StringIO()
    write(tmp, atoms, format="xyz")
    mol = Chem.MolFromXYZBlock(tmp.getvalue())
    if bond_orders:
        rdDetermineBonds.DetermineBonds(mol, charge=0, allowChargedFragments=False)
    else:
        rdDetermineBonds.DetermineConnectivity(mol)
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    return mol


def relax_with_uff(
    atoms: Atoms,
    bond_orders: bool = True,
    max_iters: int = 2000,
) -> Dict[str, Any]:
    """
    Minimize a structure with RDKit's UFF, on the bonds perceived from it.

    UFF never forms or breaks bonds, so the relaxed structure keeps the
    topology it was set up with; it can fix bond lengths and angles, not a
    wrong connectivity.

    Args:
        atoms: ASE Atoms object to relax.
        bond_orders: perceive bond orders (needs a valid molecule) or only the
            connectivity, see mol_from_atoms. Without bond orders RDKit types
            the atoms from single bonds only.
        max_iters: maximum number of minimizer iterations.

    Returns:
        Dictionary with keys:
        - atoms: the relaxed structure (a copy of the input if UFF could not
          be set up)
        - relaxed: whether UFF was set up and run
        - converged: whether the minimizer converged within max_iters
        - energy_before, energy_after: UFF energies in kcal/mol (np.nan if
          not relaxed)
    """
    result = {
        "atoms": atoms.copy(),
        "relaxed": False,
        "converged": False,
        "energy_before": np.nan,
        "energy_after": np.nan,
    }
    try:
        with BlockLogs():
            mol = mol_from_atoms(atoms, bond_orders=bond_orders)
            if not rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
                return result
            force_field = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
            energy_before = force_field.CalcEnergy()
            # 0 if converged, 1 if more iterations are needed
            status = force_field.Minimize(maxIts=max_iters)
            energy_after = force_field.CalcEnergy()
            positions = np.array(force_field.Positions()).reshape(-1, 3)
    except Exception:
        return result

    result["atoms"].positions = positions
    result.update(
        relaxed=True,
        converged=status == 0,
        energy_before=energy_before,
        energy_after=energy_after,
    )
    return result


def get_bond_lengths(mol: Chem.Mol, positions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Group the bond lengths of a structure by bond type.

    Args:
        mol: Mol with the bonds, e.g. from mol_from_atoms.
        positions: (n_atoms, 3) positions in Angstrom, in the atom order of mol.
            They may differ from the Mol's own conformer, e.g. to measure the
            bonds of a structure after relaxation.

    Returns:
        Bond lengths in Angstrom per bond type. The type is the two elements
        in Hill order joined by the bond symbol, e.g. "C-H", "C=O" or "C:C"
        (aromatic).
    """
    hill = lambda symbol: (symbol != "C", symbol != "H", symbol)
    lengths: Dict[str, list] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        symbols = sorted(
            [mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()],
            key=hill,
        )
        label = BOND_SYMBOLS.get(bond.GetBondType(), "~").join(symbols)
        lengths.setdefault(label, []).append(np.linalg.norm(positions[i] - positions[j]))
    return {label: np.array(values) for label, values in lengths.items()}
