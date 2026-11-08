import json

import numpy as np
import pytest
from ase.build import molecule
from ase.db import connect

from gpff.analysis.relaxation_diagnostics import (
    compare_pair, internal_displacement, kabsch_rmsd, summarize_pairs,
)
from gpff.cli.relax import relax_database
from gpff.cli.relax_report import compare_databases, main


def test_rigid_motion_is_not_internal_geometry_change():
    before = molecule("CH3CHO").positions
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    after = before @ rotation + [4, -2, 1]
    assert kabsch_rmsd(before, after) < 1e-12
    # Reflections cannot be removed by a proper rotation.
    assert kabsch_rmsd(before, before * [-1, 1, 1]) > 0.1
    r = before - before.mean(0)
    rigid_step = np.cross([0.01, 0.02, -0.01], r) + [1, 2, 3]
    np.testing.assert_allclose(internal_displacement(rigid_step, before), 0, atol=1e-12)


def test_summary_separates_rescued_broken_and_identity_changed_cohorts():
    def row(before, after, strain, *, changed=False, relaxed=True, converged=True):
        return {"valid_before": before, "valid_after": after,
                "smiles_before": "CC" if before else "",
                "smiles_after": ("CO" if changed else "CC") if after else "",
                "relaxed": relaxed, "converged": converged, "strain": strain,
                "rmsd": 1., "aligned_rmsd": 0.1, "bond_orders": True}
    records = [row(True, True, 2), row(True, False, 6), row(False, True, 100),
               row(True, True, 4, changed=True, converged=False),
               row(True, True, 0, relaxed=False, converged=False)]
    result = summarize_pairs(records)
    assert (result["gained"], result["lost"], result["smiles_changed"], result["smiles_compared"]) == (1, 1, 1, 3)
    cohorts = result["cohorts"]
    assert cohorts["initially_valid"]["strain"] == {"n": 3, "mean": 4., "median": 4., "p90": 5.6}
    assert cohorts["finally_valid"]["strain"]["mean"] == pytest.approx(106 / 3)
    assert cohorts["initially_valid"]["converged_strain"]["n"] == 2
    assert cohorts["same_identity"]["strain"]["median"] == 2
    json.dumps(result, allow_nan=False)
    assert summarize_pairs([])["cohorts"]["all"]["strain"]["median"] is None


def test_report_cli_writes_pairs_summary_and_provenance(tmp_path, monkeypatch):
    source, relaxed, output = tmp_path / "source.db", tmp_path / "relaxed.db", tmp_path / "report"
    atoms = molecule("CH3CHO")
    with connect(source) as db:
        db.write(atoms)
    relax_database(source, relaxed, verbose=False)
    monkeypatch.setattr("sys.argv", ["gpff_relax_report", "--samples_db_path", str(source),
                                    "--relaxed_db_path", str(relaxed), "--output_dir", str(output), "--quiet"])
    assert main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["pairing"] == "source_md5_and_atom_order"
    assert summary["valid_before"] == summary["valid_after"] == 1
    with np.load(output / "pairs.npz", allow_pickle=False) as pairs:
        assert pairs["source_id"].tolist() == [1]
        assert pairs["smiles_before"].tolist() == pairs["smiles_after"].tolist()
    assert "initially_valid" in (output / "summary.md").read_text()
    assert json.loads((output / "provenance.json").read_text())["results"]["paired_relaxation"]["inputs"]
    with connect(source) as db:
        db.write(atoms)
    with pytest.raises(ValueError, match="row counts"):
        compare_databases(source, relaxed, verbose=False)


def test_report_rejects_wrong_source_even_when_its_formula_matches(tmp_path):
    source, other, relaxed = (tmp_path / name for name in ("source.db", "other.db", "relaxed.db"))
    with connect(source) as db:
        db.write(molecule("CH4"))
    relax_database(source, relaxed, verbose=False)
    with connect(other) as db:
        atoms = molecule("CH4")
        atoms.positions *= 1.1
        db.write(atoms)
    with pytest.raises(ValueError, match="different source"):
        compare_databases(other, relaxed, verbose=False)
    with pytest.raises(ValueError, match="different files"):
        relax_database(source, source, verbose=False)
    with connect(source) as db:
        assert db.count() == 1


def test_pair_rejects_atom_order_mismatch():
    atoms = molecule("H2O")
    with pytest.raises(ValueError, match="same atoms"):
        compare_pair(atoms, atoms[::-1], {})
