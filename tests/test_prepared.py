import numpy as np
import pytest
from ase import Atoms
from ase.db import connect
from schnetpack import properties
from schnetpack.data import AtomsDataModuleError

from gpff.data.datasets import PreparedAtomsData, database_n_atoms

# GEOM-Drugs rows are conformers of molecules of very different sizes
SIZES = [3, 5, 8, 12, 4, 9, 6, 15]


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "geom_drugs.db"
    with connect(path) as db:
        db.metadata = {
            "_property_unit_dict": {},
            "_distance_unit": "Ang",
            "atomref": {},
        }
        for n in SIZES:
            positions = np.random.default_rng(n).normal(size=(n, 3))
            db.write(Atoms(numbers=[6] * n, positions=positions))
    split = tmp_path / "split.npz"
    np.savez(split, train_idx=[0, 1, 2, 3], val_idx=[4, 5], test_idx=[6, 7])
    return path, split


def make(database, **kwargs):
    path, split = database
    module = PreparedAtomsData(
        datapath=str(path),
        batch_size=2,
        split_file=str(split),
        load_properties=[],
        num_workers=0,
        **kwargs,
    )
    module.prepare_data()
    module.setup()
    return module


def sizes(dataset):
    return [int(dataset[i][properties.n_atoms]) for i in range(len(dataset))]


def test_the_split_file_fixes_the_partitions(database):
    module = make(database)
    assert sizes(module.train_dataset) == [3, 5, 8, 12]
    assert sizes(module.val_dataset) == [4, 9]
    assert sizes(module.test_dataset) == [6, 15]


@pytest.mark.parametrize("cache_in_memory", [False, True])
def test_max_n_atoms_filters_every_split(database, cache_in_memory):
    module = make(database, max_n_atoms=9, cache_in_memory=cache_in_memory)
    assert sizes(module.train_dataset) == [3, 5, 8]
    assert sizes(module.val_dataset) == [4, 9]
    assert sizes(module.test_dataset) == [6]


def test_the_sizes_are_read_from_the_database_by_dataset_index(database):
    path, _ = database
    module = make(database)
    assert database_n_atoms(path).tolist() == sizes(module.dataset) == SIZES


def test_row_ids_with_a_gap_are_an_error(database):
    path, _ = database
    with connect(path) as db:
        db.delete([2])  # dataset index 1 would read row 2, which is gone
    with pytest.raises(AtomsDataModuleError, match="row ids"):
        database_n_atoms(path)


def test_a_missing_database_is_an_error_and_is_not_built(tmp_path):
    module = PreparedAtomsData(
        datapath=str(tmp_path / "missing.db"), batch_size=2, num_workers=0
    )
    with pytest.raises(AtomsDataModuleError):
        module.prepare_data()
    assert not (tmp_path / "missing.db").exists()
