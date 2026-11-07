import math

import pytest
import torch
from torch import nn

from schnetpack import properties

from gpff import keys
from gpff.sample import Calculator, DirectDenoisingSampler
from gpff.sample.direct_denoising import _H_PREV, _LEVEL, _X0_PREV
from gpff.sample.jump_optimizers import (
    FIRE,
    LBFGS,
    MDMin,
    Adam,
    Anderson,
    ConjugateGradient,
    Momentum,
    Nesterov,
)


class _PullToOrigin(nn.Module):
    """Forces -R: the jump to x_0 halves the positions, sigma_hat = std(R) / 2."""

    def forward(self, batch):
        return {keys.forces: -batch[properties.R]}


class _Contraction(nn.Module):
    """x_0 = M x with M = diag(0.95, 0.6, 0): plain jumps converge at rate 0.95."""

    def forward(self, batch):
        positions = batch[properties.R]
        return {keys.forces: 2 * (positions * torch.tensor([0.95, 0.6, 0.0]) - positions)}


class _IsotropicContraction(nn.Module):
    """x_0 = x / (1 + |x|^2 / 10) per atom: rotation-equivariant, contracting."""

    def forward(self, batch):
        positions = batch[properties.R]
        x_0 = positions / (1 + positions.square().sum(-1, keepdim=True) / 10)
        return {keys.forces: 2 * (x_0 - positions)}


class _GaussianDenoiser(nn.Module):
    """
    Posterior mean for data N(0, s^2) per coordinate, with sigma read off the
    input: x_0 = s^2 / mean(x^2) x. Exact along the probability-flow ODE, whose
    solution is x(sigma) = x(sigma_0) sqrt((s^2 + sigma^2) / (s^2 + sigma_0^2)).
    """

    s = 1.0

    def forward(self, batch):
        positions = batch[properties.R]
        x_0 = self.s**2 / positions.square().mean() * positions
        return {keys.forces: 2 * (x_0 - positions)}


class _Restoring(nn.Module):
    """Forces pulling each atom to its target, with a stiffness per atom."""

    def forward(self, batch):
        return {keys.forces: 2 * batch["stiffness"] * (batch["target"] - batch[properties.R])}


def _batch(n_atoms=(3, 4), scale=1.0):
    torch.manual_seed(0)
    n_atoms = torch.tensor(n_atoms)
    return {
        properties.R: scale * torch.randn(int(n_atoms.sum()), 3, dtype=torch.float64),
        properties.Z: torch.ones(int(n_atoms.sum()), dtype=torch.long),
        properties.idx_m: torch.repeat_interleave(torch.arange(len(n_atoms)), n_atoms),
        properties.n_atoms: n_atoms,
        properties.idx: torch.arange(len(n_atoms)),
        properties.cell: torch.zeros(len(n_atoms), 3, 3),
        properties.pbc: torch.zeros(len(n_atoms), 3, dtype=torch.bool),
    }


def _mixed_batch():
    """sigma_hat of _PullToOrigin about 0.05 for the small molecule, about 5 for the large one."""
    torch.manual_seed(0)
    batch = _batch((3, 4))
    batch[properties.R] = torch.cat([0.1 * torch.randn(3, 3), 10.0 * torch.randn(4, 3)])
    return batch


def _denoise(model, batch, dtype=torch.float64, sampler_cls=DirectDenoisingSampler, **kwargs):
    calculator = Calculator(model=model, device="cpu", dtype=dtype)
    torch.manual_seed(1)
    return sampler_cls(calculator=calculator, **kwargs).denoise(batch)[properties.R]


def _step_inputs(sigma, forces=None, n_atoms=(3, 4)):
    """What next_positions reads: the model input and the calculator's predictions."""
    batch = _batch(n_atoms)
    x = batch[properties.R]
    if forces is None:
        forces = torch.randn_like(x)
    predictions = {keys.x_0: x + forces / 2, keys.forces: forces, keys.sigma_curr: sigma}
    return batch, x, predictions


# transport


def test_without_transport_the_sampler_is_unchanged():
    batch = _mixed_batch()
    x = batch[properties.R].clone()
    out = _denoise(_PullToOrigin(), batch, dtype=torch.float32, n_steps=2, stochastic_lambda=1.0)

    # the update before transport existed: noise at ratio 1/2, jump, no noise, jump
    torch.manual_seed(1)
    x = (x + 0.5 * torch.randn_like(x)) / 2
    x = (x + 0.0 * torch.randn_like(x)) / 2
    torch.testing.assert_close(out, x)


def test_transport_takes_ddim_steps_without_noise_above_transport_sigma():
    batch = _mixed_batch()
    x = batch[properties.R].clone()
    kwargs = dict(n_steps=3, stochastic_lambda=1.0, transport_sigma=1.0, transport_ratio=0.5)
    out = _denoise(_PullToOrigin(), batch, dtype=torch.float32, **kwargs)

    # the large molecule stays above transport_sigma (5, 3.75, 2.8): three damped
    # steps x + 0.5 * F / 2 = 0.75 x and never any noise
    torch.testing.assert_close(out[3:], 0.75**3 * x[3:])
    # the small one jumps from the first step on and is perturbed from the second
    # on, as before: no estimate yet, so no noise before the first call (the noise
    # is still drawn, so the random stream matches the sampler without transport)
    torch.manual_seed(1)
    expected = x[:3]
    for ratio in [0.0, 1 / 3, 0.0]:
        expected = (expected + ratio * torch.randn_like(x)[:3]) / 2
    torch.testing.assert_close(out[:3], expected)


def test_transport_grid_runs_from_sigma_max_to_transport_sigma():
    sampler = DirectDenoisingSampler(transport_sigma=0.3, transport_steps=10, transport_sigma_max=30.0)
    levels = sampler.transport_levels

    assert len(levels) == 11
    torch.testing.assert_close(levels[[0, -1]], torch.tensor([30.0, 0.3], dtype=levels.dtype))
    assert (levels[1:] < levels[:-1]).all()


def test_transport_ratios_step_to_the_first_level_below_sigma_and_at_least_one_level():
    sampler = DirectDenoisingSampler(transport_sigma=0.3, transport_steps=10, transport_sigma_max=30.0)
    levels = sampler.transport_levels
    idx_m = torch.tensor([0, 0, 1, 1, 1])
    batch = {
        properties.idx_m: idx_m,
        properties.n_atoms: torch.tensor([2, 3]),
        _LEVEL: torch.tensor([0, 4]),
    }
    # molecule 0 reads as far below level 1, molecule 1 as above its current level
    sigma_mol = torch.tensor([float(levels[5]) * 1.01, float(levels[3])], dtype=torch.float64)
    sigma = sigma_mol[idx_m].unsqueeze(-1)
    in_transport = torch.ones_like(sigma, dtype=torch.bool)

    ratio = sampler.transport_ratios(batch, sigma, in_transport)

    assert batch[_LEVEL].tolist() == [5, 5]
    torch.testing.assert_close(ratio[[0, 2], 0], levels[[5, 5]] / sigma_mol)


def test_only_the_molecules_in_transport_move_to_the_next_level():
    sampler = DirectDenoisingSampler(transport_sigma=0.3, transport_steps=10)
    idx_m = torch.tensor([0, 0, 1, 1, 1])
    batch = {properties.idx_m: idx_m, properties.n_atoms: torch.tensor([2, 3]), _LEVEL: torch.tensor([2, 3])}
    sigma = torch.full((5, 1), 3.0, dtype=torch.float64)
    in_transport = (idx_m == 0).unsqueeze(-1)

    sampler.transport_ratios(batch, sigma, in_transport)

    assert batch[_LEVEL][1] == 3
    assert batch[_LEVEL][0] > 2


def test_the_transport_ends_after_the_last_level():
    sampler = DirectDenoisingSampler(transport_sigma=0.3, transport_steps=10)
    sigma = torch.full((2, 1), 5.0)
    batch = {properties.idx_m: torch.tensor([0, 1]), _LEVEL: torch.tensor([9, 10])}

    # the sigma estimate is above transport_sigma, but molecule 1 has no levels left
    assert sampler.in_transport(batch, sigma).squeeze(-1).tolist() == [True, False]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(transport_steps=10),
        dict(transport_gamma=0.5),
        dict(transport_order=2),
        dict(transport_sigma=0.3, transport_steps=16),
        dict(transport_sigma=0.3, transport_steps=0),
        dict(transport_sigma=0.3, transport_gamma=1.5),
        dict(transport_sigma=0.3, transport_order=3),
        dict(transport_sigma=0.3, transport_order=2, transport_gamma=0.5),
        dict(relax_steps=0),
        dict(relax_steps=17),
    ],
)
def test_options_that_make_no_sense_are_refused(kwargs):
    with pytest.raises(ValueError):
        DirectDenoisingSampler(n_steps=16, **kwargs)


def test_stochastic_transport_steps_add_the_ddpm_noise():
    sampler = DirectDenoisingSampler(n_steps=16, transport_sigma=1.0, transport_ratio=0.5, transport_gamma=1.0)
    batch, x, predictions = _step_inputs(torch.full((7, 1), 5.0, dtype=torch.float64))
    forces = predictions[keys.forces]

    torch.manual_seed(7)
    x_next, _, _ = sampler.next_positions(batch, 3, x, predictions)

    # r = 1/2, g = r sqrt(1 - r^2): the step is 1 - sqrt(r^2 - g^2) = 1 - r^2 of F/2
    torch.manual_seed(7)
    g = 0.5 * math.sqrt(0.75)
    expected = x + (1 - 0.25) * forces / 2 + g * 5.0 * torch.randn_like(x)
    torch.testing.assert_close(x_next, expected)


def test_without_gamma_the_transport_step_is_deterministic():
    sampler = DirectDenoisingSampler(n_steps=16, transport_sigma=1.0, transport_ratio=0.25)
    batch, x, predictions = _step_inputs(torch.full((7, 1), 5.0, dtype=torch.float64))

    first, _, _ = sampler.next_positions(batch, 3, x, predictions)
    second, _, _ = sampler.next_positions(batch, 3, x, predictions)

    torch.testing.assert_close(first, x + 0.75 * predictions[keys.forces] / 2)
    assert torch.equal(first, second)


# second-order transport


# transport ends near the data's own spread s = 1: sigma 30 -> 0.47 / 0.94 / 0.47
@pytest.mark.parametrize("ratio, n_steps", [(0.25, 3), (0.5, 5), (0.5, 6)])
def test_second_order_transport_follows_the_ode_closer(ratio, n_steps):
    start = _batch(n_atoms=(200,), scale=30.0)[properties.R]
    s2 = _GaussianDenoiser.s**2
    sigma_start2 = start.square().mean() - s2
    sigma_end2 = sigma_start2 * ratio ** (2 * n_steps)
    exact = start * ((s2 + sigma_end2) / (s2 + sigma_start2)).sqrt()

    # transport_sigma 0 keeps every step a transport step
    kwargs = dict(n_steps=n_steps, stochastic_lambda=0.0, transport_sigma=0.0, transport_ratio=ratio)
    first = _denoise(_GaussianDenoiser(), _batch(n_atoms=(200,), scale=30.0), **kwargs)
    second = _denoise(_GaussianDenoiser(), _batch(n_atoms=(200,), scale=30.0), transport_order=2, **kwargs)

    first_error = (first - exact).norm() / exact.norm()
    second_error = (second - exact).norm() / exact.norm()
    assert second_error < first_error / 2


def test_second_order_step_weights_the_change_of_the_clean_estimate_by_the_ratio_of_the_steps():
    sampler = DirectDenoisingSampler(n_steps=16, transport_sigma=1.0, transport_ratio=0.5, transport_order=2)
    batch, x, predictions = _step_inputs(torch.full((7, 1), 5.0, dtype=torch.float64))
    x_0 = predictions[keys.x_0]
    x_0_prev = torch.randn_like(x)
    # the last step of molecule 0 had ratio 1/4, twice the log ratio of this one's 1/2;
    # molecule 1 took none
    idx_m = batch[properties.idx_m]
    batch[_X0_PREV] = x_0_prev
    batch[_H_PREV] = torch.zeros(7, 1, dtype=x.dtype)
    batch[_H_PREV][idx_m == 0] = math.log(4)

    x_next, _, history = sampler.next_positions(batch, 3, x, predictions)

    # x_0 + (x_0 - x_0_prev) h / (2 h_prev), then the DDIM step to half the noise level
    first_order = x + 0.5 * (x_0 - x)
    second_order = x + 0.5 * ((x_0 - x) + 0.25 * (x_0 - x_0_prev))
    first_molecule = (idx_m == 0).unsqueeze(-1)
    torch.testing.assert_close(x_next, torch.where(first_molecule, second_order, first_order))
    # and both took a transport step of ratio 1/2 now
    torch.testing.assert_close(history[_H_PREV], torch.full((7, 1), math.log(2), dtype=x.dtype))
    torch.testing.assert_close(history[_X0_PREV], x_0)


# refinement


def test_refine_step_over_relaxes_the_jump():
    batch = _batch()
    x = batch[properties.R].clone()
    out = _denoise(_PullToOrigin(), batch, n_steps=1, refine_step=1.5)

    # x + 1.5 * F / 2 with F = -x
    torch.testing.assert_close(out, 0.25 * x)


def test_rigid_free_forces_are_for_the_molecules_past_transport_only():
    sampler = DirectDenoisingSampler(n_steps=16, transport_sigma=1.0, transport_ratio=0.5, refine_rigid_free=True)
    idx_m = _batch()[properties.idx_m]
    sigma = torch.where(idx_m == 0, 5.0, 0.1).unsqueeze(-1).to(torch.float64)
    batch, x, predictions = _step_inputs(sigma)
    forces = predictions[keys.forces]

    x_next, free_forces, _ = sampler.next_positions(batch, 3, x, predictions)

    in_transport = (idx_m == 0).unsqueeze(-1)
    assert torch.equal(free_forces[in_transport.squeeze(-1)], forces[in_transport.squeeze(-1)])
    past = ~in_transport.squeeze(-1)
    assert not torch.allclose(free_forces[past], forces[past])
    # no net force on the molecule past transport, and its jump follows the rigid-free force
    torch.testing.assert_close(free_forces[past].sum(0), torch.zeros(3, dtype=torch.float64))
    expected = torch.where(in_transport, x + 0.5 * forces / 2, x + free_forces / 2)
    torch.testing.assert_close(x_next, expected)


def test_relax_steps_take_the_noise_off_the_last_steps():
    sampler = DirectDenoisingSampler(n_steps=6, relax_steps=3)
    # the 4-step schedule, then two more steps without noise
    assert [sampler.get_noise_ratio(step) for step in range(6, 0, -1)] == [0.75, 0.5, 0.25, 0.0, 0.0, 0.0]

    plain = DirectDenoisingSampler(n_steps=6)
    assert [plain.get_noise_ratio(step) for step in range(6, 0, -1)] == [
        1 - plain.get_denoising_ratio(step) for step in range(6, 0, -1)
    ]


# jump optimizers

# all start with the plain jump; Adam, which does not, is tested on its own
OPTIMIZERS = {
    "anderson": lambda: Anderson(depth=2),
    "momentum": Momentum,
    "nesterov": Nesterov,
    "mdmin": MDMin,
    "fire": FIRE,
    "cg": ConjugateGradient,
    "lbfgs": LBFGS,
}


def test_anderson_converges_where_plain_jumps_crawl():
    plain = _denoise(_Contraction(), _batch(), n_steps=6, stochastic_lambda=0.0)
    accelerated = _denoise(
        _Contraction(), _batch(), n_steps=6, stochastic_lambda=0.0, jump_optimizer=Anderson(depth=2, max_step=1e6)
    )

    # the fixed point is 0; the slow coordinate keeps 0.95^6 = 0.74 of its start
    start = _batch()[properties.R]
    assert plain[:, 0].abs().max() > 0.7 * start[:, 0].abs().max()
    # two contracting modes: depth 2 solves the linear problem after three calls
    assert accelerated.abs().max() < 1e-6 * start.abs().max()


def test_anderson_max_zero_takes_plain_jumps():
    plain = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=0.0)
    capped = _denoise(
        _Contraction(), _batch(), n_steps=5, stochastic_lambda=0.0, jump_optimizer=Anderson(depth=2, max_step=0.0)
    )
    torch.testing.assert_close(capped, plain)


def test_anderson_skips_perturbed_jumps():
    # with noise on every step but the last, no two clean jumps follow each other
    plain = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=1.0)
    accelerated = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=1.0, jump_optimizer=Anderson(depth=2))
    torch.testing.assert_close(accelerated, plain)


def test_anderson_converges_the_noise_free_finish_of_stochastic_sampling():
    # noise on steps 8..6, then five jumps without noise; the noise draws match
    kwargs = dict(n_steps=8, stochastic_lambda=1.0, relax_steps=5)
    plain = _denoise(_Contraction(), _batch(), **kwargs)
    accelerated = _denoise(_Contraction(), _batch(), jump_optimizer=Anderson(depth=2, max_step=1e6), **kwargs)

    assert accelerated.abs().max() < 1e-6 * plain.abs().max()


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_first_jump_and_stochastic_sampling_are_unchanged(name):
    # one jump, and noise on every step but the last: each jump is a fresh start
    for kwargs in [dict(n_steps=1, stochastic_lambda=0.0), dict(n_steps=4, stochastic_lambda=1.0)]:
        plain = _denoise(_Contraction(), _batch(), **kwargs)
        optimized = _denoise(_Contraction(), _batch(), jump_optimizer=OPTIMIZERS[name](), **kwargs)
        torch.testing.assert_close(optimized, plain)


@pytest.mark.parametrize("name", ["anderson", "momentum", "nesterov", "mdmin", "cg", "lbfgs"])
def test_history_speeds_up_the_slow_mode(name):
    kwargs = dict(n_steps=12, stochastic_lambda=0.0)
    plain = _denoise(_Contraction(), _batch(), **kwargs)
    optimized = _denoise(_Contraction(), _batch(), jump_optimizer=OPTIMIZERS[name](), **kwargs)
    # the fixed point is 0; the slow coordinate keeps 0.95^12 = 0.54 under plain jumps
    assert optimized[:, 0].abs().max() < 0.6 * plain[:, 0].abs().max()


def test_cg_and_lbfgs_solve_the_linear_problem():
    # uncapped: the slow mode (curvature 0.05) needs a step of 20 plain jumps
    start = _batch()[properties.R].abs().max()
    kwargs = dict(stochastic_lambda=0.0)
    # three distinct curvatures: CG ends after the first jump and three line searches
    cg = _denoise(_Contraction(), _batch(), n_steps=4, jump_optimizer=ConjugateGradient(max_step=1e6), **kwargs)
    assert cg.abs().max() < 1e-10 * start
    lbfgs = _denoise(_Contraction(), _batch(), n_steps=12, jump_optimizer=LBFGS(max_step=1e6), **kwargs)
    assert lbfgs.abs().max() < 1e-6 * start


def _rotation():
    return torch.linalg.matrix_exp(
        torch.tensor([[0.0, -0.3, 0.2], [0.3, 0.0, -0.5], [-0.2, 0.5, 0.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizers_are_rotation_equivariant(name):
    rotation = _rotation()
    batch, rotated = _batch(scale=2.0), _batch(scale=2.0)
    rotated[properties.R] = rotated[properties.R] @ rotation.T
    kwargs = dict(n_steps=6, stochastic_lambda=0.0)
    out = _denoise(_IsotropicContraction(), batch, jump_optimizer=OPTIMIZERS[name](), **kwargs)
    out_rotated = _denoise(_IsotropicContraction(), rotated, jump_optimizer=OPTIMIZERS[name](), **kwargs)
    torch.testing.assert_close(out_rotated, out @ rotation.T)


def test_adam_depends_on_the_orientation_and_moves_lr_per_coordinate():
    rotation = _rotation()
    batch, rotated = _batch(scale=2.0), _batch(scale=2.0)
    rotated[properties.R] = rotated[properties.R] @ rotation.T
    kwargs = dict(n_steps=6, stochastic_lambda=0.0, jump_optimizer=Adam(lr=0.05))
    out = _denoise(_IsotropicContraction(), batch, **kwargs)
    out_rotated = _denoise(_IsotropicContraction(), rotated, **kwargs)
    assert not torch.allclose(out_rotated, out @ rotation.T)

    # bias-corrected, the first step is lr times the sign of the force
    first = _denoise(
        _IsotropicContraction(), _batch(scale=2.0), n_steps=1, stochastic_lambda=0.0, jump_optimizer=Adam(lr=0.05)
    )
    start = _batch(scale=2.0)[properties.R]
    torch.testing.assert_close((first - start).abs(), torch.full_like(start, 0.05), rtol=1e-5, atol=1e-6)


def test_mdmin_and_fire_default_steps_match_the_plain_jump():
    assert MDMin().dt ** 2 / 2 == pytest.approx(1.0)
    assert FIRE().dt ** 2 == 1.0
    assert math.isclose(MDMin().dt, math.sqrt(2))


class _KickFirstMolecule(DirectDenoisingSampler):
    """Moves the first molecule before every model call, as noise or a shape constraint would."""

    def preprocess_step(self, batch, step, conditions=None, **kwargs):
        first = (batch[properties.idx_m] == 0).unsqueeze(-1)
        batch[properties.R] = torch.where(first, batch[properties.R] + 1e-3, batch[properties.R])
        return batch


def test_the_optimizer_restarts_only_the_molecules_whose_input_changed():
    kwargs = dict(n_steps=8, stochastic_lambda=0.0, sampler_cls=_KickFirstMolecule)
    plain = _denoise(_Contraction(), _batch(), **kwargs)
    accelerated = _denoise(_Contraction(), _batch(), jump_optimizer=Anderson(depth=2, max_step=1e6), **kwargs)

    # the kick leaves the first molecule (3 atoms) no clean jump; the second one is accelerated
    torch.testing.assert_close(accelerated[:3], plain[:3])
    assert accelerated[3:, 0].abs().max() < 0.1 * plain[3:, 0].abs().max()


def test_the_optimizer_leaves_the_molecules_in_transport_alone():
    # sigma_hat 0.05 for the small molecule: never in transport, so its second jump already
    # differs; the large one (5, 3.75, 2.8) is in transport throughout and takes damped steps only
    kwargs = dict(n_steps=3, stochastic_lambda=0.0, transport_sigma=1.0, transport_ratio=0.5)
    plain = _denoise(_PullToOrigin(), _mixed_batch(), **kwargs)
    optimized = _denoise(_PullToOrigin(), _mixed_batch(), jump_optimizer=Momentum(momentum=0.5), **kwargs)

    torch.testing.assert_close(optimized[3:], plain[3:])
    assert not torch.allclose(optimized[:3], plain[:3])


# the adaptive samplers together


def test_adaptive_sampling_keeps_the_samples_when_converged_molecules_skip_the_model():
    sizes = (5, 9, 3, 12, 7, 4)
    g = torch.Generator().manual_seed(1)
    inputs = _batch(sizes)
    stiffness = torch.rand(len(sizes), generator=g, dtype=torch.float64) * 0.9 + 0.05
    inputs["stiffness"] = stiffness[inputs[properties.idx_m]].unsqueeze(-1)
    inputs["target"] = inputs[properties.R].clone()
    inputs[properties.R] = inputs[properties.R] + 20 * torch.randn(
        inputs[properties.R].shape, generator=g, dtype=torch.float64
    )

    results = {}
    for skip in (False, True):
        sampler = DirectDenoisingSampler(
            calculator=Calculator(model=_Restoring(), device="cpu", dtype=torch.float64, skip_converged=skip),
            n_steps=24,
            stochastic_lambda=0.0,
            transport_sigma=0.5,
            transport_steps=6,
            refine_step=1.5,
            refine_rigid_free=True,
            jump_optimizer=Anderson(depth=2),
            f_max=0.05,
        )
        results[skip] = sampler.denoise({key: value.clone() for key, value in inputs.items()})

    reference, out = results[False], results[True]
    torch.testing.assert_close(out[properties.R], reference[properties.R])
    assert torch.equal(out[keys.nfe], reference[keys.nfe])
    # some molecules converge before the last step, so the model skips them
    assert (out[keys.nfe] < 24).any()
