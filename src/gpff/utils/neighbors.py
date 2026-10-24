from typing import Optional
import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import NeighborListTransform
from schnetpack.transform import MatScipyNeighborList


__all__ = ["AllToAllNeighborList", "compute_neighbors"]


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


def _fully_connected_batch(old_batch, f_dtype):
    """
    The batch that collating AllToAllNeighborList over the molecules of
    old_batch gives (same keys, pairs and pair order), built for all molecules
    at once instead of one molecule at a time.
    """
    idx_m = old_batch[properties.idx_m]
    n_atoms = old_batch[properties.n_atoms]
    device = idx_m.device

    # atoms grouped by molecule, as the per-molecule masks of the loop took them
    order = torch.argsort(idx_m, stable=True)
    counts = n_atoms.long()
    n_pairs = counts * (counts - 1)

    # Pair p of a molecule with n atoms joins atom i = p // (n - 1) to the
    # (p % (n - 1))-th of the other atoms: i-major over all j != i, like
    # AllToAllNeighborList. Single-atom molecules have no pairs, so n - 1 > 0.
    pair_mol = torch.repeat_interleave(torch.arange(len(counts), device=device), n_pairs)
    p = torch.arange(len(pair_mol), device=device) - (n_pairs.cumsum(0) - n_pairs)[pair_mol]
    n_others = (counts - 1)[pair_mol]
    idx_i_local = p // n_others
    idx_j_local = p % n_others
    idx_j_local = idx_j_local + (idx_j_local >= idx_i_local)
    atom_offset = (counts.cumsum(0) - counts)[pair_mol]

    return {
        properties.idx: old_batch[properties.idx].detach().clone(),
        properties.n_atoms: n_atoms.detach().clone(),
        properties.Z: old_batch[properties.Z].detach()[order],
        properties.R: old_batch[properties.R].detach()[order],
        properties.cell: old_batch[properties.cell].detach().to(f_dtype, copy=True),
        properties.pbc: old_batch[properties.pbc].detach().reshape(-1).clone(),
        properties.idx_i + "_local": idx_i_local,
        properties.idx_j + "_local": idx_j_local,
        properties.offsets: torch.zeros(len(pair_mol), 3, dtype=f_dtype, device=device),
        properties.idx_m: torch.repeat_interleave(torch.arange(len(counts), device=device), counts),
        properties.idx_i: idx_i_local + atom_offset,
        properties.idx_j: idx_j_local + atom_offset,
    }


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

    if fully_connected:
        batch = _fully_connected_batch(old_batch, f_dtype)
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
