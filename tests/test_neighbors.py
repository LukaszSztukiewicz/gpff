import pytest
import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn

from gpff.utils.neighbors import AllToAllNeighborList, compute_neighbors


def molecules(sizes):
    g = torch.Generator().manual_seed(0)
    return [
        {
            properties.idx: torch.tensor([k]),
            properties.n_atoms: torch.tensor([n]),
            properties.Z: torch.randint(1, 10, (n,), generator=g),
            properties.R: torch.randn(n, 3, generator=g),
            properties.cell: torch.zeros(1, 3, 3),
            properties.pbc: torch.zeros(3, dtype=torch.bool),
        }
        for k, n in enumerate(sizes)
    ]


@pytest.mark.parametrize("sizes", [[5, 1, 7, 2], [1, 1], [29, 29, 29]])
def test_fully_connected_matches_the_per_molecule_transform(sizes):
    expected = _atoms_collate_fn([AllToAllNeighborList()(dict(m)) for m in molecules(sizes)])
    batch = compute_neighbors(_atoms_collate_fn(molecules(sizes)), fully_connected=True, device="cpu")

    assert batch.keys() == expected.keys()
    for key, value in expected.items():
        assert batch[key].dtype == value.dtype, key
        assert torch.equal(batch[key], value), key
