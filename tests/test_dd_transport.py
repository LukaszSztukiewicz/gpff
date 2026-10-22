import torch
from torch import nn

from schnetpack import properties

from gpff import keys
from gpff.sample import Calculator, DirectDenoisingSampler


class _PullToOrigin(nn.Module):
    """Forces -R: the jump to x_0 halves the positions, sigma_hat = std(R) / 2."""

    def forward(self, batch):
        return {keys.forces: -batch[properties.R]}


def _batch():
    torch.manual_seed(0)
    # sigma_hat about 0.05 for the small molecule, about 5 for the large one
    small = 0.1 * torch.randn(3, 3)
    large = 10.0 * torch.randn(4, 3)
    return {
        properties.R: torch.cat([small, large]),
        properties.Z: torch.ones(7, dtype=torch.long),
        properties.idx_m: torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        properties.n_atoms: torch.tensor([3, 4]),
        properties.idx: torch.tensor([0, 1]),
        properties.cell: torch.zeros(2, 3, 3),
        properties.pbc: torch.zeros(2, 3, dtype=torch.bool),
    }


def _sampler(**kwargs):
    return DirectDenoisingSampler(calculator=Calculator(model=_PullToOrigin(), device="cpu"), **kwargs)


def test_without_transport_the_sampler_is_unchanged():
    batch = _batch()
    x = batch[properties.R].clone()
    torch.manual_seed(1)
    out = _sampler(n_steps=2, stochastic_lambda=1.0).denoise(batch)

    # the update before transport existed: noise at ratio 1/2, jump, no noise, jump
    torch.manual_seed(1)
    x = (x + 0.5 * torch.randn_like(x)) / 2
    x = (x + 0.0 * torch.randn_like(x)) / 2
    torch.testing.assert_close(out[properties.R], x)


def test_transport_takes_ddim_steps_without_noise_above_transport_sigma():
    batch = _batch()
    x = batch[properties.R].clone()
    torch.manual_seed(1)
    out = _sampler(n_steps=3, stochastic_lambda=1.0, transport_sigma=1.0, transport_ratio=0.5).denoise(batch)

    # the large molecule stays above transport_sigma (5, 3.75, 2.8): three damped
    # steps x + 0.5 * F / 2 = 0.75 x and never any noise
    torch.testing.assert_close(out[properties.R][3:], 0.75**3 * x[3:])
    # the small one jumps from the first step on and is perturbed from the second
    # on, as before: no estimate yet, so no noise before the first call (the noise
    # is still drawn, so the random stream matches the sampler without transport)
    torch.manual_seed(1)
    expected = x[:3]
    for ratio in [0.0, 1 / 3, 0.0]:
        expected = (expected + ratio * torch.randn_like(x)[:3]) / 2
    torch.testing.assert_close(out[properties.R][:3], expected)
