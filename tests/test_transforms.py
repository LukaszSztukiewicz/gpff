import torch
from schnetpack import properties

from gpff import keys
from gpff.data.transform import PseudoPESTransform
from gpff.pseudo_pes import VarianceExplodingGeometries, VEPseudoCalculator


def transform(align_permutation):
    torch.manual_seed(0)
    inputs = {
        properties.R: torch.randn(12, 3, dtype=torch.float64),
        properties.Z: torch.tensor([6] * 4 + [1] * 8),
    }
    pseudo_pes = PseudoPESTransform(
        VarianceExplodingGeometries(sigma_max=30.0, mean_free_noise=True),
        VEPseudoCalculator(),
        align_permutation=align_permutation,
    )
    return pseudo_pes(inputs)


def test_noise_target_matches_the_noisy_geometry():
    outputs = transform(align_permutation=False)
    noise = (outputs[properties.R] - outputs[keys.x_0]) / outputs[keys.sigma_curr]
    torch.testing.assert_close(outputs[keys.noise], noise)


def test_aligned_geometry_has_no_noise_target():
    # the drawn noise does not match an aligned x_t; a noise model trained with
    # alignment must fail instead of learning from mismatched targets
    outputs = transform(align_permutation=True)
    assert keys.noise not in outputs
    torch.testing.assert_close(outputs[keys.forces], 2 * (outputs[keys.x_0] - outputs[properties.R]))
