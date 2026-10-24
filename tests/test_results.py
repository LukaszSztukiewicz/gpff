import torch
from schnetpack import properties

from gpff import keys
from gpff.sample.results import batches_to_relaxed


def test_relaxed_metadata_is_split_per_molecule():
    torch.manual_seed(0)
    # the atoms of a molecule need not be contiguous
    idx_m = torch.tensor([0, 1, 0, 1, 1])
    batch = {
        properties.idx_m: idx_m,
        properties.Z: torch.tensor([6, 1, 8, 1, 7]),
        properties.R: torch.randn(5, 3),
        properties.cell: torch.zeros(2, 3, 3),
        keys.nfe: torch.tensor([3, 4]),
        keys.converged: torch.tensor([True, False]),
        keys.forces: torch.randn(5, 3),
        "scalar_representation": torch.randn(5, 8),
        "vector_representation": torch.randn(5, 3, 8),
    }

    relaxed = batches_to_relaxed([batch])

    assert len(relaxed) == 2
    for molecule, (atoms, data) in enumerate(relaxed):
        mask = idx_m == molecule
        assert set(data) == {keys.nfe, keys.converged, keys.forces}
        assert data[keys.nfe] == batch[keys.nfe][molecule].item()
        assert data[keys.converged] == batch[keys.converged][molecule].item()
        assert data[keys.forces] == batch[keys.forces][mask].tolist()
        assert (atoms.numbers == batch[properties.Z][mask].numpy()).all()
        assert (atoms.positions == batch[properties.R][mask].numpy()).all()
