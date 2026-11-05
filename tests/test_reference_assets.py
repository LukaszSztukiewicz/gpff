import sys

import numpy as np
from ase.build import molecule
from ase.db import connect

from gpff.cli import reference_assets


def test_the_walk_targets_are_written_one_row_per_molecule(tmp_path, monkeypatch):
    db_path = tmp_path / "reference.db"
    with connect(db_path) as db:
        for name in ("CH3CH2OH", "CH3", "C6H6"):
            db.write(molecule(name))
    assets = tmp_path / "assets"

    monkeypatch.setattr(sys, "argv", [
        "gpff_reference_assets", "--reference_db_path", str(db_path), "--assets_dir", str(assets),
        "--quiet", "--targets", "smiles_values", "bond_lengths", "bond_angles", "element_counts",
    ])
    assert reference_assets.main() == 0

    # the radical keeps an empty line, so the lines still line up with a split file
    assert (assets / "qm9_smiles_values.txt").read_text().splitlines()[1] == ""
    with np.load(assets / "qm9_element_counts.npz") as counts:
        assert counts["C"].tolist() == [2, 1, 6]
        assert counts["H"].tolist() == [6, 3, 6]
    with np.load(assets / "qm9_bond_lengths.npz") as bonds:
        # ethanol's 8 bonds and benzene's 12; the radical is invalid and has none
        assert sorted(set(bonds["molecule"].tolist())) == [0, 2]
        assert len(bonds["length"]) == 20
