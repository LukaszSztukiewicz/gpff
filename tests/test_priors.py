import torch
from schnetpack import properties

from gpff import keys
from gpff.sample.priors import ScaffoldGaussianPrior


def test_scaffold_gaussian_prior_centers_each_molecule_on_its_scaffold():
    g = torch.Generator().manual_seed(0)
    # the atoms of a molecule need not be contiguous
    idx_m = torch.tensor([0, 1, 0, 2, 1, 0, 2, 1, 2])
    positions = torch.randn(9, 3, generator=g, dtype=torch.float64) + 10
    scaffold_mask = torch.tensor([True, True, True, False, False, False, True, True, True])
    batch = {properties.R: positions.clone(), properties.idx_m: idx_m}

    torch.manual_seed(0)
    out = ScaffoldGaussianPrior(sigma=2.0).sample_from_batch(
        batch, conditions={keys.scaffold_mask: scaffold_mask}
    )[properties.R]

    expected = positions.clone()
    for molecule in range(3):
        mask = idx_m == molecule
        expected[mask] -= positions[mask & scaffold_mask].mean(dim=0)
    torch.manual_seed(0)
    expected[~scaffold_mask] = torch.randn(int((~scaffold_mask).sum()), 3, dtype=torch.float64) * 2.0
    torch.testing.assert_close(out, expected)
