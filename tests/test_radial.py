import pytest
import torch
from schnetpack.nn.radial import GaussianRBF

from gpff.nn.radial import GridGaussianRBF, HybridGaussianRBF, LogGaussianRBF, UniformGaussianRBF


@pytest.mark.parametrize("n_rbf, cutoff", [(20, 5.0), (64, 300.0), (600, 150.0)])
def test_uniform_basis_is_schnetpacks_gaussian_basis(n_rbf, cutoff):
    distances = torch.rand(50, 1) * cutoff

    values = UniformGaussianRBF(n_rbf, cutoff)(distances)

    assert values.shape == (50, 1, n_rbf)
    torch.testing.assert_close(values, GaussianRBF(n_rbf, cutoff)(distances), rtol=0, atol=0)


def test_hybrid_basis_is_uniform_up_to_the_switch_and_geometric_beyond():
    rbf = HybridGaussianRBF()
    offsets = rbf.offsets.double()

    assert rbf.n_rbf == 42
    uniform = offsets[offsets <= 6.0 + 1e-4]
    torch.testing.assert_close(uniform, torch.arange(25, dtype=torch.float64) * 0.25, rtol=1e-6, atol=1e-6)
    geometric = offsets[offsets > 6.0 + 1e-4]
    assert len(geometric) == 17
    torch.testing.assert_close(geometric[0] / 6.0, torch.tensor(1.25, dtype=torch.float64), rtol=1e-6, atol=0)
    torch.testing.assert_close(geometric[1:] / geometric[:-1], torch.full((16,), 1.25, dtype=torch.float64), rtol=1e-5, atol=0)
    # the last offset is the last one within the cutoff
    assert geometric[-1] <= 300.0 < geometric[-1] * 1.25


def test_log_basis_has_a_constant_ratio_from_start_to_cutoff():
    rbf = LogGaussianRBF()
    offsets = rbf.offsets.double()

    assert rbf.n_rbf == 32
    torch.testing.assert_close(offsets[[0, -1]], torch.tensor([0.5, 300.0], dtype=torch.float64), rtol=1e-6, atol=0)
    ratios = offsets[1:] / offsets[:-1]
    torch.testing.assert_close(ratios, ratios[:1].expand_as(ratios), rtol=1e-5, atol=0)
    # and so are the widths relative to the offsets, up to the two ends
    relative = (rbf.widths / rbf.offsets)[1:-1]
    torch.testing.assert_close(relative, relative[:1].expand_as(relative), rtol=1e-4, atol=0)


@pytest.mark.parametrize("rbf, lowest, highest", [(HybridGaussianRBF(), 0.0, 300.0), (LogGaussianRBF(), 0.5, 300.0)])
def test_basis_resolves_every_distance_of_its_range(rbf, lowest, highest):
    distances = torch.linspace(lowest, highest, 5000)[:, None]

    values = rbf(distances)

    assert values.shape == (5000, 1, rbf.n_rbf)
    # every distance is near the center of a Gaussian
    assert values.sum(-1).min() > 0.5


@pytest.mark.parametrize("offsets", [[1.0], [0.0, 1.0, 1.0], [2.0, 1.0, 3.0]])
def test_grid_needs_increasing_offsets(offsets):
    with pytest.raises(ValueError):
        GridGaussianRBF(offsets)


def test_trainable_grid_has_parameters():
    rbf = GridGaussianRBF([0.0, 1.0, 3.0], trainable=True)

    assert {name for name, _ in rbf.named_parameters()} == {"offsets", "widths"}
    # the mean of the gaps to the neighbors, the gap itself at the ends
    torch.testing.assert_close(rbf.widths.detach(), torch.tensor([1.0, 1.5, 2.0]))
