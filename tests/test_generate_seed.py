"""C1 regression tests: `gpff_generate` must actually apply `config.seed`.

These run `gpff_generate` as a real subprocess for every case. C1 is about
*separate processes* -- seeding inside this pytest process would not prove
the CLI seeds itself.
"""
import numpy as np
import pytest
from ase.db import connect

from tests.conftest import run_cli, venv_console_script

N_SAMPLES = 4  # <= the fixture's 4-molecule test split


def _run_generate(run_dir, seed, tiny_checkpoint, tiny_split, tiny_filtered_db):
    run_dir.mkdir()
    output_db = run_dir / "samples.db"
    argv = [
        venv_console_script("gpff_generate"),
        f"model_path={tiny_checkpoint}",
        f"split_path={tiny_split}",
        f"datapath={tiny_filtered_db}",
        f"output_db={output_db}",
        f"seed={seed}",
        f"n_samples={N_SAMPLES}",
        "batch_size=2",
        "num_workers=0",
        "prior=gaussian",
        "prior.sigma=2",
        "sampler=direct_denoising",
        "sampler.calculator.device=cpu",
        "sampler.n_steps=2",
    ]
    run_cli(argv, cwd=run_dir, timeout=60)
    return str(output_db)


def _read_samples(db_path):
    with connect(db_path) as db:
        metadata = db.metadata
        rows = [
            (tuple(int(z) for z in row.numbers), row.positions.copy())
            for row in db.select()
        ]
    assert len(rows) == N_SAMPLES
    for _, positions in rows:
        assert np.all(np.isfinite(positions))
    return metadata, rows


@pytest.fixture
def generate_run(tmp_path, tiny_checkpoint, tiny_split, tiny_filtered_db):
    """Run gpff_generate with a given seed and name, in its own subprocess/cwd."""

    def _run(run_name, seed):
        db_path = _run_generate(
            tmp_path / run_name, seed, tiny_checkpoint, tiny_split, tiny_filtered_db
        )
        return _read_samples(db_path)

    return _run


def test_same_seed_repeats(generate_run):
    _, rows_a = generate_run("run_a", seed=11)
    _, rows_b = generate_run("run_b", seed=11)

    assert [numbers for numbers, _ in rows_a] == [numbers for numbers, _ in rows_b]
    for (_, pos_a), (_, pos_b) in zip(rows_a, rows_b):
        np.testing.assert_allclose(pos_a, pos_b, rtol=0, atol=0)


def test_different_seed_differs(generate_run):
    _, rows_11 = generate_run("run_11", seed=11)
    _, rows_22 = generate_run("run_22", seed=22)

    # sanity: same test-split molecules are drawn regardless of seed (the
    # dataloader order does not depend on it), only the sampled geometry does
    assert [numbers for numbers, _ in rows_11] == [numbers for numbers, _ in rows_22]

    differs = any(
        not np.allclose(pos_a, pos_b, rtol=0, atol=0)
        for (_, pos_a), (_, pos_b) in zip(rows_11, rows_22)
    )
    assert differs, "seed=11 and seed=22 produced identical positions"


def test_seed_recorded(generate_run):
    metadata, _ = generate_run("run_recorded", seed=11)
    assert metadata["seed"] == 11
