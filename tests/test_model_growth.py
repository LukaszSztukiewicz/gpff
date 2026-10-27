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


def _model(prenorm, bounded, scale_norm=False, scale_epsilon=1e-12):
    torch.manual_seed(0)
    representation = PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=10.0),
        cutoff_fn=CosineCutoff(10.0),
        return_vector_representation=True,
        prenorm=prenorm,
        scale_norm=scale_norm,
        scale_epsilon=scale_epsilon,
    )
    head = GatedEquivariant(n_in=16, n_layers=2, bounded=bounded, x0_radius=X0_RADIUS, scaled=scale_norm)
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


def _randomize_scale_parts(model):
    """the scale embedding and the skip weights start at zero; give them values"""
    torch.manual_seed(2)
    with torch.no_grad():
        model.representation.scale_embedding.weight.normal_()
        model.representation.scale_embedding.bias.normal_()
        model.output_modules[0].skip_logit_radius_weight.fill_(0.7)


def _assert_equivariant(model):
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


@pytest.mark.parametrize("prenorm", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_forces_are_e3_and_permutation_equivariant(prenorm, bounded):
    _assert_equivariant(_model(prenorm, bounded))


@pytest.mark.parametrize("prenorm", [False, True])
def test_scaled_forces_are_e3_and_permutation_equivariant(prenorm):
    model = _model(prenorm, bounded=False, scale_norm=True)
    _randomize_scale_parts(model)
    _assert_equivariant(model)


@pytest.mark.parametrize("scale", [1e-2, 1.0, 30.0, 1e3])
def test_scaled_model_starts_dilation_equivariant(scale):
    # F(c x) = c F(x): the network only sees the molecule in units of its own radius
    model = _model(prenorm=True, bounded=False, scale_norm=True)
    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    dilated = _forces(model, [scale * p for p in positions], numbers)
    torch.testing.assert_close(dilated, scale * forces)

    plain = _model(prenorm=True, bounded=False)
    assert not torch.allclose(_forces(plain, [30 * p for p in positions], numbers), 30 * _forces(plain, positions, numbers))


@pytest.mark.parametrize("scale", [1e-3, 1e-1, 1.0, 30.0])
def test_scale_norm_hands_the_radial_basis_a_scale_free_geometry(scale):
    model = _model(prenorm=True, bounded=False, scale_norm=True)
    seen = []
    model.representation.radial_basis.register_forward_hook(lambda m, args, out: seen.append(args[0]))
    positions, numbers = _molecules(scale)
    _forces(model, positions, numbers)

    # molecules of 5-6 atoms in units of their RMS radius: pair distances of order 1-5
    assert 0.5 < seen[0].max() < 10


def test_scale_embedding_lets_the_model_use_the_absolute_scale():
    model = _model(prenorm=True, bounded=False, scale_norm=True)
    _randomize_scale_parts(model)
    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    dilated = _forces(model, [3 * p for p in positions], numbers)
    assert not torch.allclose(dilated, 3 * forces)


def test_zero_initialized_scale_parts_learn():
    model = _model(prenorm=True, bounded=False, scale_norm=True)
    positions, numbers = _molecules()
    model(_batch(positions, numbers))["forces"].pow(2).sum().backward()
    assert model.representation.scale_embedding.weight.grad.abs().sum() > 0
    assert model.output_modules[0].skip_logit_radius_weight.grad.abs() > 0


def test_bounded_and_scaled_forces_are_alternatives():
    with pytest.raises(ValueError):
        GatedEquivariant(n_in=16, bounded=True, scaled=True)


def test_scaled_forces_need_a_scale_norm_representation():
    model = _model(prenorm=False, bounded=False, scale_norm=True)
    model.representation.scale_norm = False
    positions, numbers = _molecules()
    with pytest.raises(KeyError, match="scale_norm"):
        _forces(model, positions, numbers)


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
    assert not any("norms" in k or "skip_logit" in k or "scale_embedding" in k for k in model.state_dict())

    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    # a model pickled before the options existed has no such instance attributes
    del model.representation.prenorm
    del model.output_modules[0].bounded
    del model.representation.scale_norm
    del model.output_modules[0].scaled
    torch.testing.assert_close(_forces(model, positions, numbers), forces)


def test_bounded_models_pickled_before_the_radius_term_still_load():
    model = _model(prenorm=True, bounded=True)
    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    # the radius weight starts at 0, so a model without it predicts the same at init
    del model.output_modules[0].skip_logit_radius_weight
    torch.testing.assert_close(_forces(model, positions, numbers), forces)
