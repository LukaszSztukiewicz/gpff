"""UFF and GFN2-xTB relaxation and bond lengths of single structures (one ase.Atoms in, one result out)."""
import os
import subprocess
import tempfile
from io import StringIO
from typing import Any, Dict

import numpy as np
from ase import Atoms, units
from ase.io import read, write
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdForceFieldHelpers
from rdkit.rdBase import BlockLogs

__all__ = [
    "mol_from_atoms",
    "relax_with_uff",
    "relax_with_xtb",
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


def relax_with_xtb(
    atoms: Atoms,
    xtb_command: str = "xtb",
    charge: int = 0,
    uhf: int = 0,
    max_iters: int = 500,
    timeout: float = 300,
) -> Dict[str, Any]:
    """
    Minimize a structure with GFN2-xTB, by running the xtb program.

    Unlike UFF, xTB takes no bonds: it relaxes any geometry, and bonds can
    form or break on the way.

    Args:
        atoms: ASE Atoms object to relax.
        xtb_command: the xtb executable. It finds its parameter files through
            XTBPATH, and OMP_NUM_THREADS sets its threads.
        charge: total charge.
        uhf: number of unpaired electrons.
        max_iters: maximum number of optimization cycles.
        timeout: seconds after which the xtb run is stopped.

    Returns:
        Dictionary with the keys of relax_with_uff. The energies are xTB
        total energies in kcal/mol. A run that stops at max_iters counts as
        relaxed but not converged, and returns its last geometry.
    """
    result = {
        "atoms": atoms.copy(),
        "relaxed": False,
        "converged": False,
        "energy_before": np.nan,
        "energy_after": np.nan,
    }
    with tempfile.TemporaryDirectory() as workdir:
        write(os.path.join(workdir, "input.xyz"), atoms, format="xyz")
        command = [
            xtb_command, "input.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf),
            "--opt", "normal", "--cycles", str(max_iters),
        ]
        try:
            run = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return result

        # xtbopt.log holds every optimization step, the input geometry first;
        # without it the first SCF failed
        log_path = os.path.join(workdir, "xtbopt.log")
        if not os.path.exists(log_path):
            return result
        with open(log_path) as f:
            energies = [float(line.split()[1]) for line in f if line.lstrip().startswith("energy:")]
        final_path = os.path.join(workdir, "xtbopt.xyz")
        if not energies or not os.path.exists(final_path):
            return result
        positions = read(final_path).positions
        # xtb also writes .xtboptok when it runs out of cycles, so read its verdict instead
        converged = "GEOMETRY OPTIMIZATION CONVERGED" in run.stdout

    hartree = units.Hartree / (units.kcal / units.mol)
    result["atoms"].positions = positions
    result.update(
        relaxed=True,
        converged=converged,
        energy_before=energies[0] * hartree,
        energy_after=energies[-1] * hartree,
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
