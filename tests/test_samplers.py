import torch
from schnetpack import properties

from gpff import keys
from gpff.sample import AdaptiveStochasticHeunSampler, Calculator


class TimeRecorder(torch.nn.Module):
    """Force model that records the time and sigma_next it is called with."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs):
        self.calls.append((inputs[keys.t].clone(), inputs.get(keys.sigma_next)))
        return {keys.forces: -inputs[properties.R]}


def test_adaptive_stochastic_heun_passes_sigma_next_to_the_second_stage():
    torch.manual_seed(0)
    n_mols, n_atoms = 2, 5
    batch = {
        properties.R: torch.randn(n_mols * n_atoms, 3) * 5,
        properties.idx_m: torch.arange(n_mols).repeat_interleave(n_atoms),
        properties.idx_i: torch.zeros(1, dtype=torch.long),
        properties.n_atoms: torch.full((n_mols,), n_atoms),
    }
    model = TimeRecorder()
    sampler = AdaptiveStochasticHeunSampler(
        calculator=Calculator(model=model, device="cpu"), n_steps=4, time_aware_model=True
    )
    sampler.denoise(batch)

    # calls: the initial sigma estimate, then (stage 1, stage 2) per step
    sigma_max = sampler.noise_schedule.sigma_max
    second_stages = model.calls[2::2]
    assert second_stages
    for t, sigma_next in second_stages:
        torch.testing.assert_close(t, sigma_next / sigma_max)
