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


def _model(prenorm, bounded, degree_norm=False, preconditioned=False, sigma_data=None):
    torch.manual_seed(0)
    representation = PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=10.0),
        cutoff_fn=CosineCutoff(10.0),
        return_vector_representation=True,
        prenorm=prenorm,
        degree_norm=degree_norm,
        sigma_data=sigma_data,
    )
    head = GatedEquivariant(
        n_in=16, n_layers=2, bounded=bounded, x0_radius=X0_RADIUS, preconditioned=preconditioned
    )
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


def _chain_batch(positions, numbers):
    """each atom neighbors the next one, so the degrees are 1, 2, ..., 2, 1"""
    batch = _batch(positions, numbers)
    n = batch[properties.n_atoms]
    start = torch.cumsum(n, 0) - n
    idx_i, idx_j = [], []
    for k, s in zip(n.tolist(), start.tolist()):
        a = torch.arange(k - 1) + s
        idx_i += [a, a + 1]
        idx_j += [a + 1, a]
    batch[properties.idx_i], batch[properties.idx_j] = torch.cat(idx_i), torch.cat(idx_j)
    batch[properties.offsets] = torch.zeros(len(batch[properties.idx_i]), 3, dtype=torch.float64)
    return batch


def _molecules(scale=1.0):
    torch.manual_seed(1)
    numbers = [torch.tensor([6, 6, 8, 1, 1, 1]), torch.tensor([7, 6, 1, 1, 1])]
    positions = [scale * 1.5 * torch.randn(len(z), 3, dtype=torch.float64) for z in numbers]
    return positions, numbers


def _forces(model, positions, numbers):
    with torch.no_grad():
        return model(_batch(positions, numbers))["forces"]


def _rho(p):
    """per-axis RMS radius"""
    return (p - p.mean(0)).pow(2).sum(-1).mean().div(3).sqrt()


def _randomize_zero_initialized_parts(model):
    """the noise embedding and the radius weight of the skip logit start at zero; give them values"""
    torch.manual_seed(2)
    with torch.no_grad():
        if model.representation.sigma_data is not None:
            model.representation.noise_embedding.weight.normal_()
        head = model.output_modules[0]
        if head.bounded or head.preconditioned:
            head.skip_logit_radius_weight.fill_(0.7)


@pytest.mark.parametrize("prenorm", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("degree_norm", [False, True])
def test_forces_are_e3_and_permutation_equivariant(prenorm, bounded, degree_norm):
    _assert_equivariant(_model(prenorm, bounded, degree_norm))


@pytest.mark.parametrize("prenorm", [False, True])
@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("sigma_data", [None, 1.0, 10.0])
def test_edm_options_keep_the_forces_e3_and_permutation_equivariant(prenorm, preconditioned, sigma_data):
    model = _model(prenorm, bounded=False, preconditioned=preconditioned, sigma_data=sigma_data)
    _randomize_zero_initialized_parts(model)
    _assert_equivariant(model)


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


def _stream_after_first_interaction(model, batch):
    seen = []
    hook = model.representation.mixing[0].register_forward_pre_hook(lambda m, args: seen.append(args))
    with torch.no_grad():
        model(dict(batch))
    hook.remove()
    return seen[0]


@pytest.mark.parametrize("graph", [_batch, _chain_batch])
def test_degree_norm_divides_the_summed_messages_by_the_root_of_the_degree(graph):
    # without prenorm the mixing block reads the stream the interaction left: embedding + summed messages
    plain, normed = _model(False, False), _model(False, False, degree_norm=True)
    batch = graph(*_molecules())
    q0 = plain.representation.embedding(batch[properties.Z]).unsqueeze(1)
    q_plain, mu_plain = _stream_after_first_interaction(plain, batch)
    q_normed, mu_normed = _stream_after_first_interaction(normed, batch)

    degree = torch.bincount(batch[properties.idx_i], minlength=len(batch[properties.Z]))
    scale = degree.clamp(min=1).double().rsqrt()[:, None, None]
    torch.testing.assert_close(q_normed - q0, (q_plain - q0) * scale)
    torch.testing.assert_close(mu_normed, mu_plain * scale)
    assert not torch.allclose(q_normed, q_plain)


def test_degree_norm_leaves_atoms_without_neighbors_alone():
    model = _model(False, False, degree_norm=True)
    batch = _chain_batch(*_molecules())
    # the last atom of the first molecule loses its only neighbor
    keep = ~((batch[properties.idx_i] == 5) | (batch[properties.idx_j] == 5))
    for key in (properties.idx_i, properties.idx_j, properties.offsets):
        batch[key] = batch[key][keep]
    with torch.no_grad():
        assert torch.isfinite(model(batch)["forces"]).all()


def test_options_default_off_and_old_pickles_still_load():
    model = _model(prenorm=False, bounded=False)
    assert not any(
        "norms" in k or "skip_logit" in k or "noise_embedding" in k for k in model.state_dict()
    )

    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    # a model pickled before the options existed has no such instance attributes
    del model.representation.prenorm
    del model.representation.degree_norm
    del model.representation.sigma_data
    del model.output_modules[0].bounded
    del model.output_modules[0].preconditioned
    torch.testing.assert_close(_forces(model, positions, numbers), forces)


def test_bounded_models_pickled_before_the_radius_term_still_load():
    model = _model(prenorm=True, bounded=True)
    positions, numbers = _molecules()
    forces = _forces(model, positions, numbers)
    # the radius weight starts at 0, so a model without it predicts the same at init
    del model.output_modules[0].skip_logit_radius_weight
    torch.testing.assert_close(_forces(model, positions, numbers), forces)


@pytest.mark.parametrize("sigma", [0.01, 0.1, 1.4, 5.0, 30.0])
def test_preconditioned_forces_give_the_network_a_unit_variance_target(sigma):
    # at the least-squares skip weight alpha = sigma^2 / (sigma^2 + sigma_data^2) (EDM's 1 - c_skip),
    # the head scales v by EDM's 2 c_out, and the share of the target left to v has unit variance
    sigma_data, n = 1.4, 4000
    g = torch.Generator().manual_seed(0)
    x_0 = sigma_data * torch.randn(n, 3, generator=g, dtype=torch.float64)
    x_0 = x_0 - x_0.mean(0)
    noise = torch.randn(n, 3, generator=g, dtype=torch.float64)
    x_t = x_0 + sigma * (noise - noise.mean(0))
    target = 2 * (x_0 - x_t)

    head = GatedEquivariant(n_in=4, preconditioned=True).double()
    with torch.no_grad():
        head.skip_logit_bias.zero_()
    inputs = {
        properties.R: x_t,
        properties.idx_m: torch.zeros(n, dtype=torch.long),
        properties.n_atoms: torch.tensor([n]),
    }
    logit = torch.full((n, 1), 2 * math.log(sigma / sigma_data), dtype=torch.float64)
    skip = head._preconditioned_forces(inputs, logit, torch.zeros(n, 3, dtype=torch.float64))
    c_out = head._preconditioned_forces(inputs, logit, torch.ones(n, 3, dtype=torch.float64)) - skip

    edm_c_out = sigma * sigma_data / math.sqrt(sigma**2 + sigma_data**2)
    torch.testing.assert_close(c_out, torch.full_like(c_out, 2 * edm_c_out), rtol=0.02, atol=0)
    assert abs(((target - skip) / c_out).std().item() - 1) < 0.03


def test_preconditioned_c_out_leaves_the_skip_weight_to_the_skip_term():
    head = GatedEquivariant(n_in=4, preconditioned=True).double()
    batch = _batch(*_molecules())
    n = len(batch[properties.Z])
    g = torch.Generator().manual_seed(3)
    s = torch.randn(n, 1, generator=g, dtype=torch.float64)
    v = torch.randn(n, 3, generator=g, dtype=torch.float64)
    weights = torch.randn(n, 3, generator=g, dtype=torch.float64)
    (head._preconditioned_forces(batch, s, v) * weights).sum().backward()

    # d/dz of -2 alpha x only, as if c_out were a constant
    with torch.no_grad():
        x, _, logit = head._skip_logit(batch, s)
        alpha = torch.sigmoid(logit)[batch[properties.idx_m]]
        expected = (-2 * x * weights * alpha * (1 - alpha)).sum()
    torch.testing.assert_close(head.skip_logit_bias.grad, expected)


@pytest.mark.parametrize("prenorm", [False, True])
def test_sigma_data_measures_a_molecule_in_units_of_the_larger_of_its_radius_and_sigma_data(prenorm):
    sigma_data = 2.0
    model = _model(prenorm, bounded=False, sigma_data=sigma_data)
    plain = _model(prenorm, bounded=False)
    plain.load_state_dict(model.state_dict(), strict=False)
    positions, numbers = _molecules()

    # below sigma_data: the fixed units sigma_data, whatever the noise embedding learned
    small = [0.5 * p for p in positions]
    assert all(_rho(p) < sigma_data for p in small)
    _randomize_zero_initialized_parts(model)
    torch.testing.assert_close(
        _forces(model, small, numbers), _forces(plain, [p / sigma_data for p in small], numbers)
    )

    # above: the molecule's own radius, with the noise embedding at its start
    torch.nn.init.zeros_(model.representation.noise_embedding.weight)
    large = [30 * p for p in positions]
    assert all(_rho(p) > sigma_data for p in large)
    torch.testing.assert_close(
        _forces(model, large, numbers), _forces(plain, [p / _rho(p) for p in large], numbers)
    )


def test_preconditioned_model_starts_dilation_equivariant_above_sigma_data_only():
    # F(c x) = c F(x) for noise clouds larger than sigma_data, so all large noise levels look alike;
    # a molecule below it keeps its absolute size, so a dilated clean molecule is not a fixed point
    sigma_data = 2.0
    model = _model(prenorm=True, bounded=False, preconditioned=True, sigma_data=sigma_data)
    positions, numbers = _molecules()

    large = [20 * p for p in positions]
    forces = _forces(model, large, numbers)
    for c in (0.5, 3.0):
        assert all(_rho(c * p) > sigma_data for p in large)
        torch.testing.assert_close(_forces(model, [c * p for p in large], numbers), c * forces)

    small = [0.5 * p for p in positions]
    assert all(_rho(1.5 * p) < sigma_data for p in small)
    assert not torch.allclose(_forces(model, [1.5 * p for p in small], numbers), 1.5 * _forces(model, small, numbers))


def test_noise_embedding_learns_from_molecules_above_sigma_data_only():
    model = _model(prenorm=True, bounded=False, preconditioned=True, sigma_data=2.0)
    positions, numbers = _molecules()
    for scale, learns in ((0.5, False), (20.0, True)):
        model.zero_grad()
        model(_batch([scale * p for p in positions], numbers))["forces"].pow(2).sum().backward()
        assert (model.representation.noise_embedding.weight.grad.abs().sum() > 0).item() == learns
    assert model.output_modules[0].skip_logit_radius_weight.grad.abs() > 0


def test_compiled_forward_with_sigma_data_matches_eager():
    model = _model(prenorm=True, bounded=False, preconditioned=True, sigma_data=2.0)
    _randomize_zero_initialized_parts(model)
    batch = _batch([20 * p for p in _molecules()[0]], _molecules()[1])
    with torch.no_grad():
        expected = model(dict(batch))["forces"]
        compiled = torch.compile(model, backend="eager", dynamic=True)(dict(batch))["forces"]
    torch.testing.assert_close(compiled, expected)


def test_bounded_and_preconditioned_forces_are_alternatives():
    with pytest.raises(ValueError, match="alternatives"):
        GatedEquivariant(n_in=16, bounded=True, preconditioned=True)


@pytest.mark.parametrize("sigma_data", [0.0, -1.0])
def test_sigma_data_must_be_positive(sigma_data):
    with pytest.raises(ValueError, match="sigma_data"):
        _model(prenorm=False, bounded=False, sigma_data=sigma_data)


@pytest.mark.parametrize("preconditioned", [False, True])
def test_sigma_data_draws_no_random_numbers(preconditioned):
    # the head is built after the representation, so a draw would change its weights
    plain = _model(prenorm=False, bounded=False, preconditioned=preconditioned)
    plain_rng = torch.random.get_rng_state()
    model = _model(prenorm=False, bounded=False, preconditioned=preconditioned, sigma_data=2.5)

    assert torch.equal(torch.random.get_rng_state(), plain_rng)
    assert model.representation.noise_embedding.weight.abs().max() == 0
    params = dict(model.named_parameters())
    assert set(params) - set(dict(plain.named_parameters())) == {"representation.noise_embedding.weight"}
    for name, value in plain.named_parameters():
        torch.testing.assert_close(params[name], value, rtol=0, atol=0, msg=name)
