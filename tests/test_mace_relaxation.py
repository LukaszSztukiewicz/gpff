"""Numerical validation of the custom batched MACE path; no model download."""
import os

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.optimize import LBFGS

from gpff.analysis.mace_relaxation import _BatchedMACE, _lbfgs, relax_with_mace, KCAL_PER_EV


def test_batched_lbfgs_freezes_finished_molecules_and_reports_step_limits():
    batch = torch.tensor([0, 1, 1])
    calls = []

    def harmonic(positions, active=None):
        if active is None:
            active = torch.ones(2, dtype=torch.bool)
        calls.append(active.clone())
        energy = torch.zeros(2, dtype=positions.dtype).index_add_(0, batch, positions.square().sum(1) / 2)
        return energy, -positions

    positions = torch.tensor([[0., 0, 0], [1., 0, 0], [-1., 0, 0]], dtype=torch.float64)
    final, _, _, converged, failed, steps = _lbfgs(harmonic, positions, batch, 2, 1e-8, 20)
    assert converged.tolist() == [True, True] and not failed.any()
    assert steps[0] == 0 and steps[1] > 0
    assert all(call.tolist() == [False, True] for call in calls[1:])
    np.testing.assert_allclose(final, 0, atol=1e-8)
    _, _, _, converged, _, steps = _lbfgs(harmonic, positions, batch, 2, 1e-8, 1)
    assert not converged[1] and steps[1] == 1


@pytest.fixture
def mace_model():
    pytest.importorskip("mace")
    path = os.environ.get("GPFF_MACE_MODEL")
    if not path:
        pytest.skip("set GPFF_MACE_MODEL to an existing model")
    return path


def test_batched_mace_energies_forces_and_active_compaction_match_calculator(mace_model):
    from mace.calculators import MACECalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MACECalculator(model_paths=mace_model, device=device, default_dtype="float64")
    atoms_list = [molecule(name) for name in ("CH4", "H2O", "CH3CHO")]
    model = _BatchedMACE(calculator, atoms_list, "float64", device)
    positions = torch.tensor(np.concatenate([a.positions for a in atoms_list]), dtype=torch.float64, device=device)
    # Changing a position exercises the dynamic graph, not only its initial state.
    positions[6, 0] += 0.03
    reference_e, reference_f = [], []
    offset = 0
    for original in atoms_list:
        atoms = original.copy()
        atoms.positions = positions[offset:offset + len(atoms)].cpu().numpy()
        offset += len(atoms)
        atoms.calc = calculator
        reference_e.append(atoms.get_potential_energy())
        reference_f.append(atoms.get_forces())
    energy, forces = model(positions)
    np.testing.assert_allclose(energy.cpu(), reference_e, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(forces.cpu(), np.concatenate(reference_f), rtol=1e-7, atol=1e-7)
    active = torch.tensor([False, True, True], device=device)
    e_active, f_active = model(positions, active)
    assert torch.isnan(e_active[0])
    torch.testing.assert_close(e_active[active], energy[active], rtol=1e-8, atol=1e-7)
    torch.testing.assert_close(f_active[model.batch > 0], forces[model.batch > 0], rtol=1e-7, atol=1e-7)
    assert torch.count_nonzero(f_active[model.batch == 0]) == 0


def test_batched_mace_optimization_matches_ase_lbfgs(mace_model):
    from mace.calculators import MACECalculator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    structures = [molecule(name) for name in ("CH4", "H2O")]
    for atoms in structures:
        atoms.positions *= 1.08
    results = relax_with_mace(structures, mace_model, device=device, progress=False)
    calculator = MACECalculator(model_paths=mace_model, device=device, default_dtype="float64")
    for original, result in zip(structures, results):
        atoms = original.copy()
        atoms.calc = calculator
        optimizer = LBFGS(atoms, memory=20, alpha=70, maxstep=0.2, logfile=None)
        assert optimizer.run(fmax=0.02, steps=500)
        assert result["relaxed"] and result["converged"]
        assert result["energy_after"] / KCAL_PER_EV == pytest.approx(atoms.get_potential_energy(), abs=1e-3)
        rmsd = np.sqrt(np.square(atoms.positions - result["atoms"].positions).sum(1).mean())
        assert rmsd < 0.02
