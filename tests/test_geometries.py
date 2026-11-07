import math

import pytest
import torch

from gpff.pseudo_pes.geometries import ShapeGeometries, VarianceExplodingGeometries, VariancePreservingGeometries
from gpff.utils import batch_compute_principal_variances


@pytest.mark.parametrize("pass_noise", [False, True])
def test_ve_mean_free_noise_keeps_centroid(pass_noise):
    torch.manual_seed(0)
    x_0 = torch.randn(18, 3, dtype=torch.float64)
    x_0 = x_0 - x_0.mean(0)
    noise = torch.randn_like(x_0) if pass_noise else None

    sampler = VarianceExplodingGeometries(sigma_max=1.0, mean_free_noise=True)
    x_t, _, _, noise_out = sampler(x_0=x_0, t=0.5, noise=noise)

    torch.testing.assert_close(noise_out.mean(0), torch.zeros(3, dtype=x_0.dtype))
    torch.testing.assert_close(x_t.mean(0), torch.zeros(3, dtype=x_0.dtype))


def test_ve_without_mean_free_noise_leaves_noise_untouched():
    torch.manual_seed(0)
    x_0 = torch.randn(18, 3, dtype=torch.float64)
    noise = torch.randn_like(x_0)

    sampler = VarianceExplodingGeometries(sigma_max=1.0, mean_free_noise=False)
    _, _, _, noise_out = sampler(x_0=x_0, t=0.5, noise=noise)

    torch.testing.assert_close(noise_out, noise)


def test_vp_noise_is_returned_centered_and_t_one_is_allowed():
    torch.manual_seed(0)
    x_0 = torch.randn(18, 3, dtype=torch.float64)
    x_0 = x_0 - x_0.mean(0)

    sampler = VariancePreservingGeometries(beta_min=1e-4, beta_max=1e-2, timesteps=1000)
    x_t, sigma_t, t, noise = sampler(x_0=x_0, t=1.0)

    alpha = sampler.alphas[-1]
    torch.testing.assert_close(x_t, alpha.sqrt() * x_0 + (1 - alpha).sqrt() * noise)
    torch.testing.assert_close(noise.mean(0), torch.zeros(3, dtype=x_0.dtype))
    assert t.shape == sigma_t.shape == (18, 1)


def anisotropic_molecule(n_atoms=18):
    torch.manual_seed(0)
    x_0 = torch.randn(n_atoms, 3, dtype=torch.float64) * torch.tensor([3.0, 1.5, 0.5], dtype=torch.float64)
    return x_0 - x_0.mean(0)


def test_shape_noise_has_the_size_and_shape_of_the_molecule():
    x_0 = anisotropic_molecule()
    idx_m = torch.zeros(len(x_0), dtype=torch.long)

    x_t, _, _, noise = ShapeGeometries()(x_0=x_0, t=1.0)

    # at t = 1 the geometry is the noise, as drawn by prior=shape
    torch.testing.assert_close(x_t, noise)
    torch.testing.assert_close(x_t.mean(0), torch.zeros(3, dtype=x_0.dtype))
    torch.testing.assert_close(
        batch_compute_principal_variances(x_t, idx_m), batch_compute_principal_variances(x_0, idx_m)
    )


def test_shape_sigma_is_the_scale_of_the_displacement():
    x_0 = anisotropic_molecule()
    sampler = ShapeGeometries()
    squared_displacements = []
    for _ in range(500):
        x_t, sigma_t, _, _ = sampler(x_0=x_0, t=0.5)
        squared_displacements.append(((x_t - x_0) ** 2).mean())

    assert abs(torch.stack(squared_displacements).mean() / sigma_t[0, 0] ** 2 - 1) < 0.05


def test_shape_per_atom_times_keep_the_centroid():
    x_0 = anisotropic_molecule()
    x_t, sigma_t, t, _ = ShapeGeometries(t_sigma=0.3)(x_0=x_0, t=0.5)

    torch.testing.assert_close(x_t.mean(0), x_0.mean(0))
    assert t.shape == sigma_t.shape == (len(x_0), 1)


@pytest.mark.parametrize("flattened_axes", [1, 2])
def test_shape_noise_is_finite_for_planar_and_linear_molecules(flattened_axes):
    x_0 = anisotropic_molecule()
    x_0[:, 3 - flattened_axes:] = 0.0  # planar (one axis) or linear (two axes)
    x_0 = x_0 @ torch.linalg.qr(torch.randn(3, 3, dtype=x_0.dtype))[0]  # in a tilted frame

    x_t, sigma_t, _, noise = ShapeGeometries()(x_0=x_0, t=0.7)

    assert torch.isfinite(x_t).all() and torch.isfinite(noise).all() and torch.isfinite(sigma_t).all()


def draw(sampler, n_draws, seed=0):
    """The noise levels of consecutive calls and the first atom's x_t - x_0 of each."""
    torch.manual_seed(seed)
    x_0 = torch.zeros(4, 3, dtype=torch.float64)
    outputs = [sampler(x_0=x_0) for _ in range(n_draws)]
    return torch.stack([sigma_t[0, 0] for _, sigma_t, _, _ in outputs]), torch.stack([x_t for x_t, _, _, _ in outputs])


def test_default_sigma_draws_are_the_truncated_log_normal_ones():
    # the random stream before anchors and small-sigma draws existed: sigma, then the noise
    sampler = VarianceExplodingGeometries(sigma_max=0.5)
    sigmas, x_t = draw(sampler, 30, seed=3)

    torch.manual_seed(3)
    expected_sigmas, expected_x_t = [], []
    for _ in range(30):
        kwargs = dict(size=(1,), dtype=torch.float64)
        sigma = torch.exp(torch.normal(mean=-0.7, std=1.2, **kwargs))
        while sigma > 0.5:
            sigma = torch.exp(torch.normal(mean=-0.7, std=1.2, **kwargs))
        expected_sigmas.append(sigma[0])
        expected_x_t.append(torch.randn(4, 3, dtype=torch.float64) * sigma)

    torch.testing.assert_close(sigmas, torch.stack(expected_sigmas))
    torch.testing.assert_close(x_t, torch.stack(expected_x_t))


def test_anchors_keep_the_reference_structure():
    torch.manual_seed(0)
    x_0 = torch.randn(6, 3, dtype=torch.float64)

    x_t, sigma_t, t, _ = VarianceExplodingGeometries(sigma_max=30.0, anchor_fraction=1.0)(x_0=x_0)

    torch.testing.assert_close(x_t, x_0)
    assert sigma_t.shape == t.shape == (6, 1)
    assert sigma_t.abs().max() == 0 and t.abs().max() == 0


def test_anchors_and_small_sigma_draws_replace_their_shares_of_the_log_normal_draws():
    # a log-normal far above the small range, so that the draws tell the three kinds apart
    sampler = VarianceExplodingGeometries(
        sigma_max=30.0,
        sigma_mean=2.0,
        sigma_std=0.5,
        anchor_fraction=0.2,
        small_sigma_fraction=0.3,
        small_sigma_range=(0.01, 0.1),
    )
    sigmas, _ = draw(sampler, 2000)

    anchors = sigmas == 0
    small = (sigmas >= 0.01) & (sigmas <= 0.1)
    assert anchors.double().mean().item() == pytest.approx(0.2, abs=0.05)
    assert small.double().mean().item() == pytest.approx(0.3, abs=0.05)
    assert (sigmas[~anchors & ~small] > 0.1).all()
    # log-uniform: the mean of ln(sigma) is the middle of the range in log space
    assert sigmas[small].log().mean().item() == pytest.approx(0.5 * math.log(0.01 * 0.1), abs=0.1)


def test_a_given_t_overrides_the_anchors():
    sampler = VarianceExplodingGeometries(sigma_max=30.0, anchor_fraction=1.0)
    _, sigma_t, _, _ = sampler(x_0=torch.zeros(4, 3, dtype=torch.float64), t=0.5)

    torch.testing.assert_close(sigma_t, torch.full((4, 1), 15.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor_fraction": -0.1},
        {"anchor_fraction": 0.7, "small_sigma_fraction": 0.5},
        {"anchor_fraction": 0.1, "sample_uniformly": True},
        {"small_sigma_fraction": 0.1, "small_sigma_range": (0.1, 0.01)},
        {"small_sigma_fraction": 0.1, "small_sigma_range": (0.0, 0.1)},
    ],
)
def test_ve_refuses_mixtures_that_make_no_sense(kwargs):
    with pytest.raises(ValueError):
        VarianceExplodingGeometries(sigma_max=30.0, **kwargs)
