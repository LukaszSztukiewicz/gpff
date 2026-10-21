import pytest
import torch

from gpff.pseudo_pes.geometries import VarianceExplodingGeometries


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
