import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from gpff.pseudo_pes import VarianceExplodingGeometries


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sigma_max=30),  # the experiments' sampler: barely truncated
        dict(sigma_max=2, sigma_mean=0.0),  # truncation removes ~28% of the normal
        dict(sigma_max=30, sample_uniformly=True),
    ],
)
def test_sigma_quantile_matches_training_draws(kwargs):
    # the pinned validation noise draws its levels through sigma_quantile, so it
    # must follow the distribution that training samples
    sampler = VarianceExplodingGeometries(**kwargs)
    torch.manual_seed(0)
    x_0 = torch.zeros(1, 3, dtype=torch.float64)
    drawn = np.array([sampler.get_noisy_geometry(x_0)[1].item() for _ in range(20_000)])
    quantiles = sampler.sigma_quantile(np.random.default_rng(0).random(20_000))
    assert ks_2samp(drawn, quantiles).pvalue > 1e-3
