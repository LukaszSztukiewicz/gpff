import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.db import connect

from gpff.cli.calculate_metrics import compute_metrics
from gpff.cli.prepare_metadata import compute_structure_arrays
from gpff.cli import relax as relax_module
from gpff.cli.relax import relax_database


def stretched_ethanol():
    """Ethanol with its C-O bond pulled 0.15 A too long."""
    atoms = molecule("CH3CH2OH")
    c, o = 1, 2
    direction = atoms.positions[o] - atoms.positions[c]
    atoms.positions[o] += 0.15 * direction / np.linalg.norm(direction)
    return atoms


def test_relax_database_keeps_the_rows_and_adds_the_energies(tmp_path, monkeypatch):
    far_apart = Atoms(numbers=[6, 6], positions=[[0, 0, 0], [8, 0, 0]])
    samples = tmp_path / "samples.db"
    with connect(samples) as db:
        db.write(stretched_ethanol(), data={"nfe": 16}, seed=7)
        db.write(far_apart, data={"nfe": 16}, seed=7)
    relaxed = tmp_path / "relaxed.db"

    # a method that cannot be set up hands the structure back untouched
    relax_one = relax_module._relax_one
    monkeypatch.setattr(relax_module, "_relax_one", lambda atoms, *args: (
        relax_one(atoms, *args) if len(atoms) > 2 else {
            "atoms": atoms.copy(), "relaxed": False, "converged": False,
            "energy_before": np.nan, "energy_after": np.nan,
        }
    ))
    counts = relax_database(samples, relaxed, method="uff", n_workers=1, verbose=False)

    assert counts == {"n_structures": 2, "relaxed": 1, "converged": 1}
    with connect(relaxed) as db:
        rows = list(db.select())
    assert len(rows) == 2
    # the row data and key-value pairs of the input survive
    assert [row.data["nfe"] for row in rows] == [16, 16]
    assert [row.seed for row in rows] == [7, 7]
    assert rows[0].data["relaxed"] and rows[0].data["strain"] > 0
    assert not rows[1].data["relaxed"] and np.isnan(rows[1].data["strain"])
    # the structure it could not relax is written unchanged
    assert rows[1].toatoms().get_distance(0, 1) == pytest.approx(8.0)


def test_relaxing_shortens_the_stretched_bond(tmp_path):
    samples = tmp_path / "samples.db"
    with connect(samples) as db:
        db.write(stretched_ethanol())
    relaxed = tmp_path / "relaxed.db"
    relax_database(samples, relaxed, method="uff", verbose=False)

    with connect(samples) as db:
        before = db.get(1).toatoms().get_distance(1, 2)
    with connect(relaxed) as db:
        after = db.get(1).toatoms().get_distance(1, 2)

    assert before == pytest.approx(1.57, abs=0.03)
    assert after < before


def test_mean_strain_over_the_valid_molecules(tmp_path):
    samples = tmp_path / "samples.db"
    with connect(samples) as db:
        db.write(stretched_ethanol())
        db.write(stretched_ethanol())
    relaxed = tmp_path / "relaxed.db"
    relax_database(samples, relaxed, method="uff", verbose=False)

    arrays = compute_structure_arrays(str(relaxed), run_rdkit=True, run_posebusters=False)
    results = compute_metrics(
        arrays, ["mean_strain"], reference_distributions={}, reference_smiles=None
    )

    assert arrays["strain"].shape == (2,)
    assert results["mean_strain"] == pytest.approx(arrays["strain"].mean())
    assert results["mean_strain"] > 0


def test_strain_is_nan_for_a_database_nobody_relaxed(tmp_path):
    samples = tmp_path / "samples.db"
    with connect(samples) as db:
        db.write(molecule("CH3CH2OH"))

    arrays = compute_structure_arrays(str(samples), run_rdkit=True, run_posebusters=False)

    assert np.isnan(arrays["strain"]).all()
