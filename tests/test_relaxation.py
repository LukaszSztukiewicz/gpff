import shutil

import numpy as np
import pytest
from ase.build import molecule

from gpff.analysis import (
    analyze_with_rdkit,
    get_bond_lengths,
    mol_from_atoms,
    relax_with_uff,
    relax_with_xtb,
)

needs_xtb = pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb is not on PATH")


def _stretched_ethanol():
    atoms = molecule("CH3CH2OH")
    # move the O and its H 0.15 A away from the C they are bound to
    c, o = 1, 2
    assert atoms.get_chemical_symbols()[o] == "O"
    h = int(np.argsort(atoms.get_distances(o, range(len(atoms))))[1])
    direction = atoms.positions[o] - atoms.positions[c]
    direction /= np.linalg.norm(direction)
    atoms.positions[[o, h]] += 0.15 * direction
    return atoms, c, o


def test_get_bond_lengths_groups_by_type():
    atoms = molecule("CH3CH2OH")
    lengths = get_bond_lengths(mol_from_atoms(atoms), atoms.positions)

    assert {label: len(values) for label, values in lengths.items()} == {
        "C-C": 1, "C-O": 1, "C-H": 5, "H-O": 1,
    }
    np.testing.assert_allclose(lengths["C-C"], [atoms.get_distance(0, 1)])


def test_relax_with_uff_restores_a_stretched_bond():
    atoms, c, o = _stretched_ethanol()
    result = relax_with_uff(atoms)

    assert result["relaxed"] and result["converged"]
    assert result["energy_after"] < result["energy_before"]
    assert abs(result["atoms"].get_distance(c, o) - 1.42) < 0.05
    # the input is left untouched
    assert abs(atoms.get_distance(c, o) - result["atoms"].get_distance(c, o)) > 0.1
    assert analyze_with_rdkit(result["atoms"])["canonical_smiles"] == "CCO"


def test_relax_with_uff_without_bond_orders_reads_an_over_coordinated_carbon():
    atoms = molecule("CH4")
    # a fifth H right next to the carbon: no valid bond orders exist
    atoms += molecule("H2")[:1]
    atoms.positions[-1] = atoms.positions[0] + [0.0, 0.0, -1.05]

    assert not relax_with_uff(atoms)["relaxed"]
    assert len(mol_from_atoms(atoms, bond_orders=False).GetAtomWithIdx(0).GetBonds()) == 5


@needs_xtb
def test_relax_with_xtb_restores_a_stretched_bond():
    atoms, c, o = _stretched_ethanol()
    result = relax_with_xtb(atoms)

    assert result["relaxed"] and result["converged"]
    assert result["energy_after"] < result["energy_before"]
    assert abs(result["atoms"].get_distance(c, o) - 1.42) < 0.05
    assert analyze_with_rdkit(result["atoms"])["canonical_smiles"] == "CCO"


@needs_xtb
def test_relax_with_xtb_reports_a_run_out_of_cycles():
    atoms, _, _ = _stretched_ethanol()
    result = relax_with_xtb(atoms, max_iters=1)

    assert result["relaxed"] and not result["converged"]
