import torch
from schnetpack import properties

from gpff import keys
from gpff.sample.base import Calculator


class NoiseModel(torch.nn.Module):
    """Predicts the noise under eps_pred, like the diffusion baselines."""

    def forward(self, inputs):
        return {keys.eps_pred: torch.ones_like(inputs[properties.R])}


def test_noise_model_predictions_are_read_by_default():
    batch = {
        properties.R: torch.zeros(4, 3),
        properties.idx_m: torch.tensor([0, 0, 1, 1]),
        properties.idx_i: torch.tensor([0, 1, 2, 3]),
        keys.converged: torch.zeros(2, dtype=torch.bool),
        keys.nfe: torch.zeros(2, dtype=torch.int32),
    }
    calculator = Calculator(model=NoiseModel(), noise_model=True, device="cpu")
    results = calculator.calculate(batch, x0=True, noise=True, sigma_t=2.0)

    torch.testing.assert_close(results[keys.x_0], torch.full((4, 3), -2.0))
    torch.testing.assert_close(results[keys.noise], torch.ones(4, 3))
