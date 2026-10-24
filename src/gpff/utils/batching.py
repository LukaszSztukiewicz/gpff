"""
Utility functions for batching and unbatching ase.Atoms objects to schnetpack inputs.
"""
import torch
from schnetpack import properties
from ase import Atoms
import numpy as np


__all__ = [
    "unbatch_to_dicts",
    "atoms_to_dict",
    "dict_to_atoms",
    "batch_to_atoms_list",
]


def unbatch_to_dicts(batch):
    batch_list = []
    for i, idx_m in enumerate(batch[properties.idx_m].unique()):
        mask = batch[properties.idx_m] == idx_m
        batch_list.append(
            {
                properties.idx_m: batch[properties.idx_m][mask],
                properties.R: batch[properties.R][mask],
                properties.Z: batch[properties.Z][mask],
                properties.cell: batch[properties.cell][i].unsqueeze(0),
            }
        )

    return batch_list


def dict_to_atoms(inputs):
    numpy_inputs = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
    return Atoms(
        numbers=numpy_inputs[properties.Z],
        positions=numpy_inputs[properties.R],
        cell=np.squeeze(numpy_inputs[properties.cell]),
    )


def atoms_to_dict(atoms):
    return {
        properties.R: torch.tensor(atoms.positions),
        properties.Z: torch.tensor(atoms.numbers),
        properties.cell: torch.tensor(atoms.cell).unsqueeze(0),
        properties.n_atoms: torch.tensor(len(atoms)).unsqueeze(0),
        properties.pbc: torch.tensor(atoms.pbc),
    }


def batch_to_atoms_list(batch):
    """
    One ase.Atoms per molecule, ordered like idx_m.unique(); what
    dict_to_atoms makes of each of unbatch_to_dicts(batch), with every tensor
    converted once and sliced rather than masked once per molecule.
    """
    _, inverse, counts = batch[properties.idx_m].unique(return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    bounds = [0] + torch.cumsum(counts, 0).tolist()
    numbers = batch[properties.Z][order].detach().cpu().numpy()
    positions = batch[properties.R][order].detach().cpu().numpy()
    cells = batch[properties.cell].detach().cpu().numpy()
    return [
        Atoms(numbers=numbers[start:end], positions=positions[start:end], cell=cells[m])
        for m, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]))
    ]
