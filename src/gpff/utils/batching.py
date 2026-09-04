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
    batch_list = unbatch_to_dicts(batch)
    atoms_list = [dict_to_atoms(inputs) for inputs in batch_list]

    return atoms_list
