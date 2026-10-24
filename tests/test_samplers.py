import pytest
import torch
from schnetpack import properties

from gpff import keys
from gpff.sample import (
    AdaptiveStochasticHeunSampler,
    Calculator,
    DirectDenoisingSampler,
    ScaffoldDirectDenoisingSampler,
)
from gpff.sample.observers import SamplerObserver


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


class Denoiser(torch.nn.Module):
    """Force model whose x_0 prediction is a fixed structure."""

    def __init__(self, x_0):
        super().__init__()
        self.x_0 = x_0

    def forward(self, inputs):
        return {keys.forces: 2 * (self.x_0 - inputs[properties.R])}


class PositionRecorder(SamplerObserver):
    def on_start(self, sampler, batch):
        self.steps = []

    def on_step_end(self, batch, step):
        self.steps.append((batch[keys.converged].clone(), batch[properties.R].clone()))


@pytest.mark.parametrize("sampler_cls", [DirectDenoisingSampler, ScaffoldDirectDenoisingSampler])
def test_direct_denoising_keeps_converged_molecules_in_place(sampler_cls):
    torch.manual_seed(0)
    n_mols, n_atoms = 8, 6
    idx_m = torch.arange(n_mols).repeat_interleave(n_atoms)
    x_0 = torch.randn(n_mols * n_atoms, 3)
    scaffold_mask = torch.arange(n_mols * n_atoms) % n_atoms < 2
    # a prior that already keeps the scaffold atoms at their clean positions
    positions = torch.where(scaffold_mask[:, None], x_0, torch.randn_like(x_0) * 5)
    batch = {
        properties.R: positions,
        properties.idx_m: idx_m,
        properties.idx_i: torch.zeros(1, dtype=torch.long),
        properties.n_atoms: torch.full((n_mols,), n_atoms),
    }
    recorder = PositionRecorder()
    sampler = sampler_cls(calculator=Calculator(model=Denoiser(x_0), device="cpu"), n_steps=20, f_max=2.5)
    sampler.denoise(batch, conditions={keys.scaffold_mask: scaffold_mask}, observers=[recorder])

    converged_early = False
    for (converged, before), (_, after) in zip(recorder.steps, recorder.steps[1:]):
        frozen = converged[idx_m]
        converged_early |= bool(converged.any())
        torch.testing.assert_close(after[frozen], before[frozen])
    assert converged_early
