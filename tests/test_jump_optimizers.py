import math

import pytest
import torch
from torch import nn

from schnetpack import properties

from gpff import keys
from gpff.sample import Calculator, DirectDenoisingSampler
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

from test_dd_acceleration import _Contraction, _batch


class _IsotropicContraction(nn.Module):
    """x_0 = x / (1 + |x|^2 / 10) per atom: rotation-equivariant, contracting."""

    def forward(self, batch):
        positions = batch[properties.R]
        x_0 = positions / (1 + positions.square().sum(-1, keepdim=True) / 10)
        return {keys.forces: 2 * (x_0 - positions)}


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


def _denoise(model, batch, **kwargs):
    calculator = Calculator(model=model, device="cpu", dtype=torch.float64)
    torch.manual_seed(1)
    return DirectDenoisingSampler(calculator=calculator, **kwargs).denoise(batch)[properties.R]


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


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizers_are_rotation_equivariant(name):
    rotation = torch.linalg.matrix_exp(torch.tensor([[0.0, -0.3, 0.2], [0.3, 0.0, -0.5], [-0.2, 0.5, 0.0]],
                                                    dtype=torch.float64))
    batch, rotated = _batch(scale=2.0), _batch(scale=2.0)
    rotated[properties.R] = rotated[properties.R] @ rotation.T
    kwargs = dict(n_steps=6, stochastic_lambda=0.0)
    out = _denoise(_IsotropicContraction(), batch, jump_optimizer=OPTIMIZERS[name](), **kwargs)
    out_rotated = _denoise(_IsotropicContraction(), rotated, jump_optimizer=OPTIMIZERS[name](), **kwargs)
    torch.testing.assert_close(out_rotated, out @ rotation.T)


def test_adam_depends_on_the_orientation_and_moves_lr_per_coordinate():
    rotation = torch.linalg.matrix_exp(torch.tensor([[0.0, -0.3, 0.2], [0.3, 0.0, -0.5], [-0.2, 0.5, 0.0]],
                                                    dtype=torch.float64))
    batch, rotated = _batch(scale=2.0), _batch(scale=2.0)
    rotated[properties.R] = rotated[properties.R] @ rotation.T
    kwargs = dict(n_steps=6, stochastic_lambda=0.0, jump_optimizer=Adam(lr=0.05))
    out = _denoise(_IsotropicContraction(), batch, **kwargs)
    out_rotated = _denoise(_IsotropicContraction(), rotated, **kwargs)
    assert not torch.allclose(out_rotated, out @ rotation.T)

    # bias-corrected, the first step is lr times the sign of the force
    first = _denoise(_IsotropicContraction(), _batch(scale=2.0), n_steps=1, stochastic_lambda=0.0,
                     jump_optimizer=Adam(lr=0.05))
    start = _batch(scale=2.0)[properties.R]
    torch.testing.assert_close((first - start).abs(), torch.full_like(start, 0.05), rtol=1e-5, atol=1e-6)


def test_mdmin_and_fire_default_steps_match_the_plain_jump():
    assert MDMin().dt ** 2 / 2 == pytest.approx(1.0)
    assert FIRE().dt ** 2 == 1.0
    assert math.isclose(MDMin().dt, math.sqrt(2))
