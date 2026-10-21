import pytest
import torch

from gpff.pseudo_pes.weights import FlooredSigmaWeights


def _weights(sigma: float, n_atoms: int = 5, sigma_floor: float = 0.03):
    x_0 = torch.zeros(n_atoms, 3, dtype=torch.float64)
    sigma_t = torch.full((n_atoms, 1), sigma, dtype=torch.float64)
    return FlooredSigmaWeights(sigma_floor=sigma_floor)(sigma=sigma_t, x_0=x_0)


def test_floored_sigma_weights_match_sigma_weights_far_above_the_floor():
    _, force_weight = _weights(3.0)
    torch.testing.assert_close(force_weight, torch.full((5, 1), 1 / 3.0**2, dtype=torch.float64), rtol=1e-3, atol=0)


def test_floored_sigma_weights_level_off_at_the_floor():
    _, force_weight = _weights(0.0)
    torch.testing.assert_close(force_weight, torch.full((5, 1), 1 / 0.03**2, dtype=torch.float64))

    # halfway in sigma^2 at sigma = sigma_floor
    _, force_weight = _weights(0.03)
    torch.testing.assert_close(force_weight, torch.full((5, 1), 0.5 / 0.03**2, dtype=torch.float64))


def test_floored_sigma_weights_decrease_with_sigma():
    sigmas = [0.0, 0.01, 0.03, 0.1, 1.0, 30.0]
    force_weights = [_weights(sigma)[1][0, 0].item() for sigma in sigmas]
    assert all(a > b for a, b in zip(force_weights, force_weights[1:]))


def test_floored_sigma_weights_energy_weight_is_the_squared_force_weight():
    energy_weight, force_weight = _weights(0.2)
    torch.testing.assert_close(energy_weight, force_weight**2)
