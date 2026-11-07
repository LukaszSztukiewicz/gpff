import pytest
import torch

from gpff.pseudo_pes.weights import FlooredSigmaWeights, SigmaWeights, SoftSigmaWeights

N_ATOMS = 5
FLOORED_AND_SOFT = [FlooredSigmaWeights, SoftSigmaWeights]


def force_weight(generator, sigma: float):
    x_0 = torch.zeros(N_ATOMS, 3, dtype=torch.float64)
    sigma_t = torch.full((N_ATOMS, 1), sigma, dtype=torch.float64)
    _, weight = generator(sigma=sigma_t, x_0=x_0)
    assert weight.shape == (N_ATOMS, 1)
    return weight[0, 0].item()


@pytest.mark.parametrize("factory", FLOORED_AND_SOFT)
def test_weights_are_finite_and_non_increasing_in_sigma_down_to_zero(factory):
    sigmas = [0.0, 0.01, 0.03, 0.1, 0.5, 1.0, 3.0, 30.0]
    weights = [force_weight(factory(), sigma) for sigma in sigmas]

    assert torch.isfinite(torch.tensor(weights)).all()
    assert all(a >= b for a, b in zip(weights, weights[1:]))


@pytest.mark.parametrize("factory", FLOORED_AND_SOFT)
def test_energy_weight_is_the_squared_force_weight(factory):
    x_0 = torch.zeros(N_ATOMS, 3, dtype=torch.float64)
    sigma_t = torch.full((N_ATOMS, 1), 0.2, dtype=torch.float64)
    energy_weight, weight = factory()(sigma=sigma_t, x_0=x_0)

    torch.testing.assert_close(energy_weight, weight**2)


@pytest.mark.parametrize("factory", FLOORED_AND_SOFT)
def test_clipping_still_applies(factory):
    assert force_weight(factory(max_clipping=5.0), 0.0) == 5.0


def test_floored_sigma_weights_match_sigma_weights_far_above_the_floor():
    assert force_weight(FlooredSigmaWeights(), 3.0) == pytest.approx(1 / 3.0**2, rel=1e-3)
    assert force_weight(FlooredSigmaWeights(), 3.0) == pytest.approx(force_weight(SigmaWeights(), 3.0), rel=1e-3)


def test_floored_sigma_weights_level_off_at_the_floor():
    assert force_weight(FlooredSigmaWeights(), 0.0) == pytest.approx(1 / 0.03**2)
    # halfway at sigma = sigma_floor
    assert force_weight(FlooredSigmaWeights(), 0.03) == pytest.approx(0.5 / 0.03**2)


def test_soft_sigma_weights_fall_like_sigma_squared_above_the_knee():
    generator = SoftSigmaWeights(sigma_knee=1.0)

    assert force_weight(generator, 3.0) == pytest.approx(1 / 3.0**2)
    assert force_weight(generator, 1.0) == pytest.approx(1.0)


def test_soft_sigma_weights_fall_like_sigma_below_the_knee_and_level_off_at_the_floor():
    generator = SoftSigmaWeights(sigma_knee=2.0, sigma_floor=0.1)

    assert force_weight(generator, 0.5) == pytest.approx(1 / (0.5 * 2.0))
    assert force_weight(generator, 0.1) == pytest.approx(1 / (0.1 * 2.0))
    assert force_weight(generator, 0.0) == pytest.approx(1 / (0.1 * 2.0))
    # no jump at the knee
    assert force_weight(generator, 2.0 - 1e-9) == pytest.approx(force_weight(generator, 2.0), rel=1e-6)


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (FlooredSigmaWeights, {"sigma_floor": 0.0}),
        (SoftSigmaWeights, {"sigma_floor": 0.0}),
        (SoftSigmaWeights, {"sigma_knee": 0.01, "sigma_floor": 0.03}),
    ],
)
def test_weights_refuse_a_floor_that_would_not_bound_them(factory, kwargs):
    with pytest.raises(ValueError, match="sigma_floor"):
        factory(**kwargs)
