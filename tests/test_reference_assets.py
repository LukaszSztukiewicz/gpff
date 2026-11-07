import sys

import numpy as np
import pytest
from ase.build import molecule
from ase.db import connect
from ase.units import Hartree

from gpff.analysis import analyze_with_rdkit, compute_elongation
from gpff.cli import reference_assets
from gpff.data.datasets.qm9 import build_filtered_qm9

# QM9's energy_U0 single-atom reference energies, in Hartree
U0_ATOMREFS = {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730}
MOLECULES = ("CH4", "H2O", "CH3OH")


def write_raw_qm9(path, **metadata_changes):
    """A stand-in for the raw QM9 database, with schnetpack's units and atomrefs."""
    atomrefs = [U0_ATOMREFS.get(z, 0.0) for z in range(100)]
    with connect(path) as db:
        for i, name in enumerate(MOLECULES):
            db.write(molecule(name), data={"energy_U0": np.array([-40.0 - i]), "gap": np.array([0.2 + 0.1 * i])})
        db.metadata = {
            "_property_unit_dict": {"energy_U0": "Ha", "gap": "Ha"},
            "_distance_unit": "Ang",
            "atomrefs": {"energy_U0": atomrefs},
            **metadata_changes,
        }


def run_reference_assets(monkeypatch, db_path, assets_dir, *targets):
    argv = ["gpff_reference_assets", "--reference_db_path", str(db_path), "--assets_dir", str(assets_dir)]
    monkeypatch.setattr(sys, "argv", [*argv, "--quiet", *(["--targets", *targets] if targets else [])])
    return reference_assets.main()


def test_assets_are_derived_from_a_freshly_filtered_database(tmp_path, monkeypatch):
    write_raw_qm9(tmp_path / "qm9.db")
    filtered = tmp_path / "qm9_filtered.db"
    build_filtered_qm9(str(tmp_path / "qm9.db"), str(filtered), progress=False)

    assert run_reference_assets(monkeypatch, filtered, tmp_path / "assets") == 0

    assets = {
        key: np.load(tmp_path / "assets" / f"qm9_{key}.npy") for key in reference_assets.METADATA_ARRAY_TARGETS
    }
    with connect(filtered) as db:
        rows = list(db.select())
    assert len(rows) == len(MOLECULES)
    # the atomization energy in eV, as gpff_prepare_metadata computes it for the samples
    np.testing.assert_allclose(
        assets["energy_U0_values"],
        [(row.data["energy_U0"][0] - sum(U0_ATOMREFS[z] for z in row.numbers)) * Hartree for row in rows],
    )
    np.testing.assert_allclose(assets["gap_values"], [row.data["gap"][0] * Hartree for row in rows])
    np.testing.assert_allclose(assets["elongation_values"], [compute_elongation(row.toatoms()) for row in rows])
    assert assets["energy_U0_atomrefs_eV"][6] == pytest.approx(U0_ATOMREFS[6] * Hartree)
    smiles = (tmp_path / "assets" / "qm9_smiles_values.txt").read_text().splitlines()
    assert len(smiles) == len(MOLECULES) and all(smiles)


def test_filtered_qm9_stores_the_smiles_the_samples_and_assets_use(tmp_path):
    # the layout of the paper's database: explicit-hydrogen "smiles", plus the
    # canonical form, under a named convention
    write_raw_qm9(tmp_path / "qm9.db")
    build_filtered_qm9(str(tmp_path / "qm9.db"), str(tmp_path / "qm9_filtered.db"), progress=False)

    with connect(tmp_path / "qm9_filtered.db") as db:
        assert db.metadata["smiles_format"] == "gpff_explicit_h_isomeric_v1"
        for row in db.select():
            analysis = analyze_with_rdkit(row.toatoms())
            assert row.smiles == analysis["smiles"] and "[H]" in row.smiles
            assert row.canonical_smiles == analysis["canonical_smiles"]


def test_a_stored_array_is_used_as_it_is(tmp_path, monkeypatch):
    # older reference databases carry the arrays in their metadata, so neither
    # the rows nor their units are looked at
    with connect(tmp_path / "reference.db") as db:
        for name in MOLECULES:
            db.write(molecule(name))
        db.metadata = {"gap_values": [1.0, 2.0, 3.0]}

    assert run_reference_assets(monkeypatch, tmp_path / "reference.db", tmp_path / "assets", "gap_values") == 0
    np.testing.assert_array_equal(np.load(tmp_path / "assets" / "qm9_gap_values.npy"), [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "metadata_changes, reason",
    [
        ({"gap_values": [1.0, 2.0]}, "2 values for 3 molecules"),
        ({"_property_unit_dict": {"energy_U0": "Ha", "gap": "eV"}}, "gap is in 'eV'"),
        ({"atomrefs": {}}, "no energy_U0 atomrefs"),
        ({"_distance_unit": "Bohr"}, "distances are in 'Bohr'"),
    ],
)
def test_one_array_that_cannot_be_built_stops_the_run_before_any_work(
    tmp_path, monkeypatch, capsys, metadata_changes, reason
):
    write_raw_qm9(tmp_path / "reference.db", **metadata_changes)

    assert run_reference_assets(monkeypatch, tmp_path / "reference.db", tmp_path / "assets") == 1
    assert reason in capsys.readouterr().err
    assert not (tmp_path / "assets").exists()


def test_a_failed_smiles_walk_writes_no_assets(tmp_path, monkeypatch):
    # the arrays are built before the walk, and none may land next to the
    # SMILES file of an earlier run
    write_raw_qm9(tmp_path / "reference.db")

    def crash(*args, **kwargs):
        raise RuntimeError("RDKit crashed")

    monkeypatch.setattr(reference_assets, "compute_reference_smiles", crash)
    with pytest.raises(RuntimeError):
        run_reference_assets(monkeypatch, tmp_path / "reference.db", tmp_path / "assets")
    assert not (tmp_path / "assets").exists()
