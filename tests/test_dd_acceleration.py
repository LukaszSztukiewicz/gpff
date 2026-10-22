import pytest
import torch
from torch import nn

from schnetpack import properties

from gpff import keys
from gpff.sample import Calculator, DirectDenoisingSampler


class _Contraction(nn.Module):
    """x_0 = M x with M = diag(0.95, 0.6, 0): plain jumps converge at rate 0.95."""

    def forward(self, batch):
        positions = batch[properties.R]
        return {keys.forces: 2 * (positions * torch.tensor([0.95, 0.6, 0.0]) - positions)}


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


def _denoise(model, batch, **kwargs):
    calculator = Calculator(model=model, device="cpu", dtype=torch.float64)
    torch.manual_seed(1)
    return DirectDenoisingSampler(calculator=calculator, **kwargs).denoise(batch)[properties.R]


def test_anderson_converges_where_plain_jumps_crawl():
    plain = _denoise(_Contraction(), _batch(), n_steps=6, stochastic_lambda=0.0)
    accelerated = _denoise(
        _Contraction(), _batch(), n_steps=6, stochastic_lambda=0.0, anderson_depth=2, anderson_max=1e6
    )

    # the fixed point is 0; the slow coordinate keeps 0.95^6 = 0.74 of its start
    start = _batch()[properties.R]
    assert plain[:, 0].abs().max() > 0.7 * start[:, 0].abs().max()
    # two contracting modes: depth 2 solves the linear problem after three calls
    assert accelerated.abs().max() < 1e-6 * start.abs().max()


def test_anderson_max_zero_takes_plain_jumps():
    plain = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=0.0)
    capped = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=0.0, anderson_depth=2, anderson_max=0.0)
    torch.testing.assert_close(capped, plain)


def test_anderson_skips_perturbed_jumps():
    # with noise on every step but the last, no two clean jumps follow each other
    plain = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=1.0)
    accelerated = _denoise(_Contraction(), _batch(), n_steps=5, stochastic_lambda=1.0, anderson_depth=2)
    torch.testing.assert_close(accelerated, plain)


def test_anderson_converges_the_noise_free_finish_of_stochastic_sampling():
    # noise on steps 8..6, then five jumps without noise; the noise draws match
    kwargs = dict(n_steps=8, stochastic_lambda=1.0, relax_steps=5)
    plain = _denoise(_Contraction(), _batch(), **kwargs)
    accelerated = _denoise(_Contraction(), _batch(), anderson_depth=2, anderson_max=1e6, **kwargs)

    assert accelerated.abs().max() < 1e-6 * plain.abs().max()


def test_relax_steps_take_the_noise_off_the_last_steps():
    sampler = DirectDenoisingSampler(n_steps=6, relax_steps=3)
    # the 4-step schedule, then two more steps without noise
    assert [sampler.get_noise_ratio(step) for step in range(6, 0, -1)] == [0.75, 0.5, 0.25, 0.0, 0.0, 0.0]

    plain = DirectDenoisingSampler(n_steps=6)
    assert [plain.get_noise_ratio(step) for step in range(6, 0, -1)] == [
        1 - plain.get_denoising_ratio(step) for step in range(6, 0, -1)
    ]


# transport ends near the data's own spread s = 1: sigma 30 -> 0.47 / 0.94 / 0.47
@pytest.mark.parametrize("ratio, n_steps", [(0.25, 3), (0.5, 5), (0.5, 6)])
def test_second_order_transport_follows_the_ode_closer(ratio, n_steps):
    batch = _batch(n_atoms=(200,), scale=30.0)
    start = batch[properties.R].clone()
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
