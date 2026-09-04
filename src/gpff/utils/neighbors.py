from typing import Optional
import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import NeighborListTransform
from schnetpack.transform import MatScipyNeighborList


__all__ = ["compute_neighbors"]


class AllToAllNeighborList(NeighborListTransform):
    def __init__(self):
        super().__init__(cutoff=1e8)

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        n_atoms = Z.shape[0]
        idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
        idx_j = torch.arange(n_atoms).repeat(n_atoms)

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        offset = torch.zeros(n_atoms * (n_atoms - 1), 3, dtype=positions.dtype)
        return idx_i, idx_j, offset


def compute_neighbors(
    old_batch,
    cutoff=5.0,
    fully_connected=False,
    device=None,
):
    """
    function to compute the neighbors for a batch of systems

    Args:
        old_batch: batch of systems to compute the neighbors for
        cutoff: cutoff radius for the neighbor list
        fully_connected: if True, all atoms are connected to each other.
                            Ignores the cutoff.
        device: Pytorch device
    """
    if device is None:
        device = old_batch[properties.R].device

    # get the float precision
    f_dtype = old_batch[properties.R].dtype

    # initialize the neighbor list transform
    if fully_connected:
        neighbors_calculator = AllToAllNeighborList()
    else:
        neighbors_calculator = MatScipyNeighborList(
            cutoff=cutoff
        )

    batch = []

    # compute the neighbors for each molecule in the batch
    for j, i in enumerate(old_batch[properties.idx_m].unique()):
        mask = old_batch[properties.idx_m] == i
        inp = {
            properties.idx: old_batch[properties.idx][[j]].detach().cpu(),
            properties.n_atoms: old_batch[properties.n_atoms][[j]].detach().cpu(),
            properties.Z: old_batch[properties.Z][mask].detach().cpu(),
            properties.R: old_batch[properties.R][mask].detach().cpu(),
            properties.cell: old_batch[properties.cell][[j]].detach().to(f_dtype).cpu(),
            properties.pbc: old_batch[properties.pbc].view(-1, 3)[j].detach().cpu(),
        }

        inp = neighbors_calculator(inp)

        batch.append(inp)

    # create the new batch
    batch = _atoms_collate_fn(batch)

    # add original batch keys
    skipped_keys = [k for k in old_batch.keys() if k not in batch.keys()]
    batch.update({k: old_batch[k] for k in skipped_keys})

    batch = {p: batch[p].to(device) for p in batch}

    return batch
