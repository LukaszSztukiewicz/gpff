import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.db import connect

from gpff.analysis import (
    FAILURES,
    analyze_with_rdkit,
    collect_element_counts,
    numbers_from_element_counts,
)
from gpff.analysis import structure as structure_module
from gpff.cli.calculate_metrics import load_metadata
from gpff.cli.prepare_metadata import compute_structure_arrays, write_array


def two_methanes(distance: float = 10.0) -> Atoms:
    first, second = molecule("CH4"), molecule("CH4")
    second.translate([distance, 0, 0])
    return first + second


def test_a_valid_molecule_has_no_failure():
    analysis = analyze_with_rdkit(molecule("CH3CH2OH"))

    assert analysis["rdkit_valid"] and analysis["failure"] == "valid"
    assert analysis["n_fragments"] == 1


def test_molecules_far_apart_are_fragments():
    analysis = analyze_with_rdkit(two_methanes())

    assert not analysis["rdkit_valid"]
    assert analysis["failure"] == "fragments"
    assert analysis["n_fragments"] == 2


def test_a_methyl_radical_is_a_radical():
    analysis = analyze_with_rdkit(molecule("CH3"))

    assert not analysis["rdkit_valid"]
    assert analysis["failure"] == "radical"


def test_a_nonfinite_coordinate_is_reported_as_such():
    atoms = molecule("CH3CH2OH")
    atoms.positions[0, 1] = np.nan

    analysis = analyze_with_rdkit(atoms)

    assert not analysis["rdkit_valid"]
    assert analysis["failure"] == "nonfinite"
    assert analysis["n_fragments"] == -1


def test_no_fitting_bond_orders_is_bond_orders(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Final molecular charge does not match input")

    monkeypatch.setattr(structure_module.rdDetermineBonds, "DetermineBonds", fail)

    assert analyze_with_rdkit(molecule("CH3CH2OH"))["failure"] == "bond_orders"


def test_a_timeout_inside_the_analysis_is_a_timeout(monkeypatch):
    def time_out(*args, **kwargs):
        raise structure_module._RDKitTimeout()

    monkeypatch.setattr(structure_module.rdDetermineBonds, "DetermineBonds", time_out)

    assert analyze_with_rdkit(molecule("CH3CH2OH"))["failure"] == "timeout"


def test_failure_is_valid_exactly_when_rdkit_accepts_the_molecule():
    squeezed = molecule("CH3CH2OH")
    squeezed.positions *= 0.5
    for atoms in (molecule("CH3CH2OH"), molecule("C6H6"), two_methanes(), molecule("CH3"),
                  squeezed, Atoms(numbers=[6, 6], positions=[[0, 0, 0], [5, 0, 0]])):
        analysis = analyze_with_rdkit(atoms)

        assert analysis["failure"] in FAILURES
        assert analysis["rdkit_valid"] == (analysis["failure"] == "valid")


def test_element_counts_have_a_row_per_molecule_and_a_column_per_element():
    table = collect_element_counts([[1, 1, 6], [6, 8], [1]])

    assert sorted(table) == ["C", "H", "O"]
    assert table["H"].tolist() == [2, 0, 1]
    assert table["C"].tolist() == [1, 1, 0]
    assert table["O"].tolist() == [0, 1, 0]
    assert [sorted(m.tolist()) for m in numbers_from_element_counts(table)] == [
        [1, 1, 6], [6, 8], [1]
    ]


def test_the_rdkit_pass_writes_failure_fragments_and_element_counts(tmp_path):
    db_path = tmp_path / "samples.db"
    with connect(db_path) as db:
        db.write(molecule("CH3CH2OH"))
        db.write(two_methanes())

    arrays = compute_structure_arrays(str(db_path), run_rdkit=True, run_posebusters=False)
    for key in ("failure", "n_fragments", "element_counts"):
        write_array(tmp_path / "metadata", key, arrays[key])
    loaded = load_metadata(tmp_path / "metadata")

    assert loaded["failure"].tolist() == ["valid", "fragments"]
    assert loaded["n_fragments"].tolist() == [1, 2]
    assert loaded["element_counts"]["H"].tolist() == [6, 8]
    assert loaded["element_counts"]["C"].tolist() == [2, 2]
    assert loaded["element_counts"]["O"].tolist() == [1, 0]
