"""Verify the shared fixtures in conftest.py work standalone before relying on them."""
import numpy as np
from ase.db import connect
from schnetpack.utils import load_model

from gpff.data.datasets.qm9 import QM9Valid
from tests.conftest import N_FIXTURE_MOLECULES, TEST_SIZE, TRAIN_SIZE, VAL_SIZE


def test_tiny_raw_db_has_20_rows(tiny_raw_db):
    with connect(tiny_raw_db) as db:
        assert db.count() == N_FIXTURE_MOLECULES


def test_tiny_filtered_db_has_20_rows_and_valid_flag(tiny_filtered_db):
    with connect(tiny_filtered_db) as db:
        assert db.count() == N_FIXTURE_MOLECULES
        assert "valid" in db.metadata.get("filter_flags", [])


def test_tiny_split_indices_disjoint_and_cover_all_rows(tiny_split, tiny_filtered_db):
    with connect(tiny_filtered_db) as db:
        n_rows = db.count()

    split = np.load(tiny_split)
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    assert len(train_idx) == TRAIN_SIZE
    assert len(val_idx) == VAL_SIZE
    assert len(test_idx) == TEST_SIZE

    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx.tolist())), "split indices overlap"
    assert set(all_idx.tolist()) == set(range(n_rows)), "split does not cover every row"


def test_qm9_valid_prepares_and_sets_up_on_fixture(tiny_filtered_db, tiny_split):
    qm9_valid = QM9Valid(
        datapath=tiny_filtered_db,
        batch_size=2,
        split_file=tiny_split,
        load_properties=[],
        num_workers=0,
    )
    qm9_valid.prepare_data()
    qm9_valid.setup()

    assert len(qm9_valid._train_dataset) == TRAIN_SIZE
    assert len(qm9_valid._val_dataset) == VAL_SIZE
    assert len(qm9_valid._test_dataset) == TEST_SIZE

    n_structures = sum(batch["_n_atoms"].numel() for batch in qm9_valid.test_dataloader())
    assert n_structures == TEST_SIZE == 4


def test_tiny_checkpoint_loads(tiny_checkpoint):
    model = load_model(tiny_checkpoint, device="cpu")
    assert model is not None
