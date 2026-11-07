import pytest
import torch
from schnetpack import properties

from gpff import keys
from gpff.data.transform import PseudoPESTransform
from gpff.pseudo_pes import VarianceExplodingGeometries, VEPseudoCalculator
from gpff.utils.alignment import align_rotation


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


def rotation_aligned(t, max_sigma):
    """The outputs for the noise level t * 30 and the x_t the same noise gives without any alignment."""
    torch.manual_seed(0)
    inputs = {
        properties.R: torch.randn(12, 3, dtype=torch.float64),
        properties.Z: torch.tensor([6] * 4 + [1] * 8),
    }
    noise = torch.randn(12, 3, dtype=torch.float64)
    pseudo_pes = PseudoPESTransform(
        VarianceExplodingGeometries(sigma_max=30.0, mean_free_noise=True),
        VEPseudoCalculator(),
        align_rotation=True,
        align_rotation_max_sigma=max_sigma,
    )
    outputs = pseudo_pes(inputs, t=t, noise=noise)
    unaligned = outputs[keys.x_0] + outputs[keys.sigma_curr] * (noise - noise.mean(0, keepdim=True))
    return outputs, unaligned


@pytest.mark.parametrize("t, max_sigma", [(0.01, None), (0.1, None), (0.01, 1.0)])
def test_rotation_is_aligned_below_max_sigma(t, max_sigma):
    outputs, unaligned = rotation_aligned(t, max_sigma)

    aligned = align_rotation(unaligned, outputs[keys.x_0])
    torch.testing.assert_close(outputs[properties.R], aligned)
    assert not torch.allclose(aligned, unaligned)


def test_rotation_is_left_alone_from_max_sigma_on():
    outputs, unaligned = rotation_aligned(t=0.1, max_sigma=1.0)

    torch.testing.assert_close(outputs[properties.R], unaligned)
    # the labels stay exact, and no noise target is kept whether or not this sample was aligned
    torch.testing.assert_close(outputs[keys.forces], 2 * (outputs[keys.x_0] - outputs[properties.R]))
    assert keys.noise not in outputs
