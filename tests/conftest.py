"""Shared fixtures for the gpff regression tests.

Fixtures build a self-contained, CPU-only "tiny QM9" universe: a fake raw
database (`tiny_raw_db`), the filtered dataset generation reads
(`tiny_filtered_db`), a matching split (`tiny_split`) and a minimal untrained
checkpoint schnetpack can load (`tiny_checkpoint`). They are session-scoped
for speed; tests that mutate a database or its metadata must copy the file
first (see `copy_db`) rather than writing into a shared fixture.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.db import connect

import gpff.cli  # noqa: F401 - registers OmegaConf resolvers + torch.load shim
from gpff.analysis import get_rdkit_validity
from gpff.data.datasets.qm9 import build_filtered_qm9

# Appendix A of docs/impl-01-gpff-fixes.md: QM9 energy_U0 single-atom reference
# energies, in Hartree, indexed by atomic number.
ATOMREF_ENERGY_U0_HARTREE = {
    1: -0.500273,
    6: -37.846772,
    7: -54.583861,
    8: -75.064579,
    9: -99.718730,
}

# ase.build.molecule (G2) names used to build the fixture; small, common,
# RDKit-valid organic molecules restricted to H, C, N, O.
FIXTURE_MOLECULE_NAMES = ["CH4", "C2H6", "CH3OH", "CH3CHO", "H2O", "CH3CH2OH"]
N_FIXTURE_MOLECULES = 20
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 12, 4, 4


def atomrefs_energy_U0_array() -> list:
    """The 100-entry energy_U0 atomrefs list (Hartree) from Appendix A."""
    values = [0.0] * 100
    for z, value in ATOMREF_ENERGY_U0_HARTREE.items():
        values[z] = value
    return values


def _build_tiny_molecules():
    """`N_FIXTURE_MOLECULES` small, RDKit-valid molecules with reproducible jitter."""
    rng = np.random.default_rng(0)
    molecules = []
    attempts = 0
    max_attempts = N_FIXTURE_MOLECULES * 50
    name_idx = 0
    while len(molecules) < N_FIXTURE_MOLECULES and attempts < max_attempts:
        attempts += 1
        name = FIXTURE_MOLECULE_NAMES[name_idx % len(FIXTURE_MOLECULE_NAMES)]
        name_idx += 1
        atoms = molecule(name)
        atoms.positions = atoms.positions + rng.normal(scale=0.01, size=atoms.positions.shape)
        if get_rdkit_validity(atoms):
            molecules.append(atoms)
    if len(molecules) < N_FIXTURE_MOLECULES:
        raise RuntimeError(
            f"Could only build {len(molecules)}/{N_FIXTURE_MOLECULES} RDKit-valid "
            f"fixture molecules after {attempts} attempts."
        )
    return molecules


def copy_db(src_path: str, dst_path: str) -> str:
    """Copy an ASE database file so a test can mutate its rows or metadata."""
    shutil.copy2(src_path, dst_path)
    return dst_path


def venv_console_script(name: str) -> str:
    """Absolute path to a console-script entry point in the active virtual environment."""
    path = os.path.join(os.path.dirname(sys.executable), name)
    if not os.path.exists(path):
        raise RuntimeError(
            f"No {name!r} console script next to the active Python at {sys.executable}."
        )
    return path


def run_cli(argv, cwd, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess:
    """Run a console-script entry point the way the regression tests need it to run.

    Uses a private working directory (Hydra writes logs and split lock files
    there) and sets PWD explicitly, since `cwd=` alone does not update it and
    both generate.yaml and train.yaml put `file://${oc.env:PWD}` on the Hydra
    config searchpath. Pins thread counts so CPU runs stay cheap and repeatable.
    """
    cwd = str(cwd)
    env = {**os.environ, "PWD": cwd, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [str(a) for a in argv],
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(str(a) for a in argv)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


@pytest.fixture(scope="session")
def tiny_raw_db(tmp_path_factory) -> str:
    """A 20-molecule ASE database that looks like raw QM9 to schnetpack."""
    path = str(tmp_path_factory.mktemp("tiny_raw") / "qm9_raw.db")
    molecules = _build_tiny_molecules()
    with connect(path) as db:
        for i, atoms in enumerate(molecules):
            data = {
                "energy_U0": np.array([-40.0 - 0.1 * i]),
                "gap": np.array([0.1 + 0.01 * i]),
            }
            db.write(atoms, data=data)
        db.metadata = {
            "_property_unit_dict": {"energy_U0": "Ha", "gap": "Ha"},
            "_distance_unit": "Ang",
            "atomrefs": {"energy_U0": atomrefs_energy_U0_array()},
        }
    return path


@pytest.fixture(scope="session")
def tiny_filtered_db(tmp_path_factory, tiny_raw_db) -> str:
    """The generation dataset: `tiny_raw_db` run through `build_filtered_qm9`."""
    path = str(tmp_path_factory.mktemp("tiny_filtered") / "qm9_filtered.db")
    n_written = build_filtered_qm9(tiny_raw_db, path, progress=False)
    assert n_written == N_FIXTURE_MOLECULES, (
        f"Expected all {N_FIXTURE_MOLECULES} fixture molecules to be valid, "
        f"got {n_written}."
    )
    return path


@pytest.fixture(scope="session")
def tiny_split(tmp_path_factory, tiny_filtered_db) -> str:
    """A 12/4/4 split over `tiny_filtered_db`'s zero-based filtered row order."""
    with connect(tiny_filtered_db) as db:
        n_rows = db.count()
    assert n_rows == N_FIXTURE_MOLECULES

    path = str(tmp_path_factory.mktemp("tiny_split") / "split.npz")
    train_idx = np.arange(0, TRAIN_SIZE)
    val_idx = np.arange(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)
    test_idx = np.arange(TRAIN_SIZE + VAL_SIZE, TRAIN_SIZE + VAL_SIZE + TEST_SIZE)
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(range(n_rows))
    np.savez(path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return path


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> str:
    """A minimal, untrained GPFF checkpoint loadable by `schnetpack.utils.load_model`.

    Built by composing the real training config (shrunk for speed) and saving
    the model through the exact code path `gpff_train` uses, without training
    or touching the trainer/logger/datamodule. Untrained weights are fine: C1
    tests generation *determinism*, not sample quality.
    """
    import hydra
    from hydra import compose, initialize_config_module
    from schnetpack.utils import str2class

    overrides = [
        "experiment=gpff",
        "globals.n_atom_basis=8",
        "globals.n_interactions=1",
        "globals.sigma_max=2",
        "run.id=fixture",
    ]
    with initialize_config_module(config_module="gpff.configs", version_base="1.2"):
        config = compose(config_name="train", overrides=overrides)

    torch.manual_seed(0)
    model = hydra.utils.instantiate(config.model)

    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )
    task = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    path = str(tmp_path_factory.mktemp("tiny_checkpoint") / "best_model")
    task.save_model(path, do_postprocessing=True)
    return path
