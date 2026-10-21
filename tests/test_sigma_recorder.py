import numpy as np
import torch
from torch import nn

from schnetpack import properties

from gpff import keys
from gpff.cli.generate import _get_sigmas_path, _save_sigmas
from gpff.sample import Calculator, DirectDenoisingSampler, SigmaRecorder


class _PullToOrigin(nn.Module):
    """Forces -R, so direct denoising halves the positions at every step."""

    def forward(self, batch):
        return {keys.forces: -batch[properties.R]}


def _batch():
    torch.manual_seed(0)
    small = 0.1 * torch.randn(3, 3)
    large = 5.0 * torch.randn(4, 3)
    return {
        properties.R: torch.cat([small, large]),
        properties.Z: torch.ones(7, dtype=torch.long),
        properties.idx_m: torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        properties.n_atoms: torch.tensor([3, 4]),
        properties.idx: torch.tensor([0, 1]),
        properties.cell: torch.zeros(2, 3, 3),
        properties.pbc: torch.zeros(2, 3, dtype=torch.bool),
    }


def test_sigma_recorder_records_sigma_hat_at_every_evaluation():
    batch = _batch()
    positions = batch[properties.R].clone()
    calculator = Calculator(model=_PullToOrigin(), device="cpu")
    # the small molecule's largest force is below f_max, so it converges in the first step
    sampler = DirectDenoisingSampler(calculator=calculator, n_steps=3, stochastic_lambda=0.0, f_max=1.0)
    recorder = SigmaRecorder()

    sampler.denoise(batch, observers=[recorder])

    sigma_small = (positions[:3] / 2).flatten().std()
    sigma_large = (positions[3:] / 2).flatten().std()
    nan = float("nan")
    expected = torch.tensor([
        [sigma_small, sigma_large],
        [nan, sigma_large / 2],
        [nan, sigma_large / 4],
    ])
    torch.testing.assert_close(recorder.sigmas, expected, equal_nan=True)
    # the calculator only notifies observers during a denoise call
    assert calculator.observers == []


def test_sigma_records_are_saved_in_database_order(tmp_path):
    nan = float("nan")
    # two batches that stopped after a different number of evaluations
    records = [
        torch.tensor([[0.5, 2.0], [0.25, 1.0]]),
        torch.tensor([[3.0], [1.5], [nan]]),
    ]
    path = _get_sigmas_path(str(tmp_path / "samples_run.db"))
    assert path == str(tmp_path / "sigmas_run.npz")

    _save_sigmas(path, records)

    np.testing.assert_array_equal(
        np.load(path)["sigma"],
        np.array([[0.5, 0.25, nan], [2.0, 1.0, nan], [3.0, 1.5, nan]], dtype=np.float32),
    )


def test_sigmas_path_without_samples_in_the_name():
    assert _get_sigmas_path("out/run.db") == "out/run_sigmas.npz"
