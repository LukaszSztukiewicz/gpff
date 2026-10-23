import math

import pytest
import torch
import schnetpack.properties as properties
from schnetpack.atomistic import PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF

from gpff.model import GatedEquivariant, PaiNN

X0_RADIUS = 10.0


def _model(prenorm, bounded):
    torch.manual_seed(0)
    representation = PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=10.0),
        cutoff_fn=CosineCutoff(10.0),
        return_vector_representation=True,
        prenorm=prenorm,
    )
    head = GatedEquivariant(n_in=16, n_layers=2, bounded=bounded, x0_radius=X0_RADIUS)
    model = NeuralNetworkPotential(
        representation=representation, input_modules=[PairwiseDistances()], output_modules=[head]
    )
    return model.double().eval()


def _batch(positions, numbers):
    """all-to-all neighbors within each molecule"""
    n = torch.tensor([len(z) for z in numbers])
    start = torch.cumsum(n, 0) - n
    idx_i, idx_j = [], []
    for k, s in zip(n.tolist(), start.tolist()):
        i, j = torch.meshgrid(torch.arange(k), torch.arange(k), indexing="ij")
        mask = i != j
        idx_i.append(i[mask] + s)
        idx_j.append(j[mask] + s)
    idx_i, idx_j = torch.cat(idx_i), torch.cat(idx_j)
    return {
        properties.Z: torch.cat(numbers),
        properties.R: torch.cat(positions),
        properties.n_atoms: n,
        properties.idx_m: torch.repeat_interleave(torch.arange(len(n)), n),
        properties.idx_i: idx_i,
        properties.idx_j: idx_j,
        properties.offsets: torch.zeros(len(idx_i), 3, dtype=torch.float64),
    }


def _molecules(scale=1.0):
    torch.manual_seed(1)
    numbers = [torch.tensor([6, 6, 8, 1, 1, 1]), torch.tensor([7, 6, 1, 1, 1])]
    positions = [scale * 1.5 * torch.randn(len(z), 3, dtype=torch.float64) for z in numbers]
    return positions, numbers


def _forces(model, positions, numbers):
    with torch.no_grad():
        return model(_batch(positions, numbers))["forces"]


@pytest.mark.parametrize("prenorm", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_forces_are_e3_and_permutation_equivariant(prenorm, bounded):
    model = _model(prenorm, bounded)
    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    shift = torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    moved = _forces(model, [p @ rotation.T + shift for p in positions], numbers)
    torch.testing.assert_close(moved, forces @ rotation.T)

    perm = torch.randperm(len(numbers[0]))
    permuted = _forces(model, [positions[0][perm], positions[1]], [numbers[0][perm], numbers[1]])
    torch.testing.assert_close(permuted[: len(perm)], forces[: len(perm)][perm])
    torch.testing.assert_close(permuted[len(perm):], forces[len(perm):])


@pytest.mark.parametrize("scale", [1e-3, 1e-1, 1.0, 30.0])
def test_bounded_forces_obey_the_bound_of_their_target(scale):
    # |2 (E[x_0 | x] - x)_i| <= 2 (r_i + R): the head keeps |F_i| <= 2 (r_i + x0_radius)
    model = _model(prenorm=True, bounded=True)
    positions, numbers = _molecules(scale)
    forces = _forces(model, positions, numbers)
    r = torch.cat([(p - p.mean(0)).norm(dim=-1) for p in positions])

    assert torch.isfinite(forces).all()
    assert (forces.norm(dim=-1) <= 2 * (r + X0_RADIUS)).all()


def test_prenorm_keeps_a_feature_excursion_from_blowing_up_the_output():
    positions, numbers = _molecules()
    gains = {}
    for prenorm in (False, True):
        model = _model(prenorm, bounded=False)
        base = _forces(model, positions, numbers).norm(dim=-1).max()
        # scale the features leaving the first block, as a dense atom cluster does
        hook = model.representation.mixing[0].register_forward_hook(lambda m, i, out: (out[0] * 1e6, out[1] * 1e6))
        excursion = _forces(model, positions, numbers).norm(dim=-1).max()
        hook.remove()
        gains[prenorm] = (excursion / base).item()

    assert math.isfinite(gains[True]) and gains[True] < 1e3
    assert not math.isfinite(gains[False]) or gains[False] > 1e6


def test_options_default_off_and_old_pickles_still_load():
    model = _model(prenorm=False, bounded=False)
    assert not any("norms" in k or "skip_logit" in k for k in model.state_dict())

    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    # a model pickled before the options existed has no such instance attributes
    del model.representation.prenorm
    del model.output_modules[0].bounded
    torch.testing.assert_close(_forces(model, positions, numbers), forces)
