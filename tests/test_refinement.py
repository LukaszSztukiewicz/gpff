import numpy as np
import pytest
import torch
from ase import Atoms
from ase.db import connect
from schnetpack import properties

from gpff import keys
from gpff.analysis.damping import probe_damping
from gpff.cli.refine import refine_database
from gpff.sample.base import Calculator
from gpff.sample.refinement import refine_structures


class Harmonic(torch.nn.Module):
    def __init__(self, factor=-2.):
        super().__init__()
        self.factor = factor

    def forward(self, batch):
        return {keys.forces: self.factor * batch[properties.R]}


def calculator(factor=-2.):
    return Calculator(Harmonic(factor), device="cpu", dtype="float64")


def structures():
    return [Atoms("H2", positions=[[0.1, 0, 0], [-0.1, 0, 0]]),
            Atoms("H2", positions=[[0, 0, 0], [0, 0, 0]])]


def test_dd_refinement_counts_terminal_check_and_preserves_inputs():
    atoms = structures()
    results, trace = refine_structures(atoms, calculator(), f_max=0.01, max_steps=4, trace=True)
    assert [r["dd_nfe"] for r in results] == [2, 1]
    assert all(r["dd_converged"] and not r["dd_failed"] for r in results)
    np.testing.assert_allclose(results[0]["atoms"].positions, 0)
    assert atoms[0].positions[0, 0] == 0.1
    np.testing.assert_allclose(trace[:2, 0], [0.1, 0])
    assert np.isnan(trace[2:, 0]).all()
    assert np.isnan(trace[1:, 1]).all()
    # Rebatching preserves ordering, convergence and NFE counts.
    single, _ = refine_structures(atoms, calculator(), f_max=0.01, max_steps=4, batch_size=1)
    assert [r["dd_nfe"] for r in single] == [2, 1]


def test_step_limit_is_not_force_convergence():
    results, _ = refine_structures(structures()[:1], calculator(), max_steps=1)
    assert results[0]["dd_nfe"] == 1
    assert not results[0]["dd_converged"]  # no terminal force check at the new position yet
    assert not results[0]["dd_failed"]
    with pytest.raises(ValueError, match="positive"):
        refine_structures(structures(), calculator(), max_steps=0)


def test_refinement_database_keeps_metadata_and_total_nfe(tmp_path):
    source, output, trace = (tmp_path / name for name in ("source.db", "refined.db", "trace.npz"))
    with connect(source) as db:
        db.write(structures()[0], seed=7, data={"nfe": 16, "custom": "keep", "strain": 4, "relaxed": True})
        db.metadata = {"dataset": "test", "relaxed_with": "uff"}
    counts = refine_database(source, output, calculator(), max_steps=4, trace_path=trace)
    assert counts == {"n_structures": 1, "converged": 1, "failed": 0, "additional_nfe": 2}
    with connect(output) as db:
        row = db.get(1)
        assert row.seed == 7 and row.data["custom"] == "keep"
        assert row.data["nfe"] == 18 and row.data["dd_nfe"] == 2
        assert "strain" not in row.data and "relaxed" not in row.data
        assert db.metadata["dataset"] == "test" and "relaxed_with" not in db.metadata
        assert db.metadata["refinement"]["source_md5"]
    with np.load(trace) as saved:
        assert saved["max_shift"].shape == (4, 1)
    with pytest.raises(FileExistsError):
        refine_database(source, output, calculator())


def test_damping_probe_detects_period_two_and_damping_removes_it():
    atoms = structures()[:1]
    # F=-4x gives x_next=-x for eta=1, and x_next=0 for eta=1/2.
    oscillating, damped = probe_damping(atoms, calculator(-4), etas=(1., 0.5), steps=3)
    np.testing.assert_allclose(oscillating["step_cosine"][1:], -1)
    np.testing.assert_allclose(oscillating["max_shift"], 0.2)
    np.testing.assert_allclose(oscillating["translation"], 0)
    np.testing.assert_allclose(damped["max_shift"][:, 0], [0.1, 0, 0])
    assert np.isnan(damped["step_cosine"]).all()  # zero-length steps have no cosine
    assert atoms[0].positions[0, 0] == 0.1


def test_damping_rejects_nonfinite_predictions():
    with pytest.raises(ValueError, match="Nonfinite"):
        probe_damping(structures()[:1], calculator(float("nan")), steps=2)
