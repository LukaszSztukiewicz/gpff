from ase import Atoms
from ase.db import connect

from gpff.cli.prepare_metadata import _SamplesData


def test_samples_db_loads_without_being_written(tmp_path):
    # gpff_generate stores only the run config, none of schnetpack's units
    db_path = str(tmp_path / "samples.db")
    with connect(db_path) as db:
        db.write(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]))
        db.metadata = {"config": {"seed": 42}}

    dataset = _SamplesData(datapath=db_path, load_properties=[], distance_unit="Ang")

    assert len(dataset) == 1
    with connect(db_path) as db:
        assert db.metadata == {"config": {"seed": 42}}
