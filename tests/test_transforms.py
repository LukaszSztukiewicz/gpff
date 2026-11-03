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


def exchanged_cloud(align_by_element):
    # sigma = 1 and noise = x_0[perm] - x_0 make x_t a copy of x_0 with a carbon (atom 0)
    # and a hydrogen (atom 2) exchanged
    torch.manual_seed(0)
    z = torch.tensor([6, 6, 1, 1, 1, 8])
    x0 = torch.randn(6, 3, dtype=torch.float64)
    x0 = x0 - x0.mean(0)
    noise = x0[torch.tensor([2, 1, 0, 3, 4, 5])] - x0
    pseudo_pes = PseudoPESTransform(
        VarianceExplodingGeometries(sigma_max=1.0, mean_free_noise=True),
        VEPseudoCalculator(),
        align_permutation=True,
        align_by_element=align_by_element,
    )
    outputs = pseudo_pes({properties.R: x0.clone(), properties.Z: z.clone()}, t=1.0, noise=noise)
    return x0, z, outputs


def test_anonymous_alignment_pairs_by_position_and_keeps_the_elements():
    x0, z, outputs = exchanged_cloud(align_by_element=False)
    torch.testing.assert_close(outputs[properties.R], x0)
    torch.testing.assert_close(outputs[keys.forces], torch.zeros_like(x0))
    assert torch.equal(outputs[properties.Z], z)


def test_alignment_by_element_cannot_undo_an_exchange_of_elements():
    x0, z, outputs = exchanged_cloud(align_by_element=True)
    assert not torch.allclose(outputs[properties.R], x0)
    assert torch.equal(outputs[properties.Z], z)
