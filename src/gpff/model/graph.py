from typing import Tuple

import torch
import torch.nn as nn
from schnetpack.nn.cutoff import cosine_cutoff

from gpff.model.geometry import radius_of_gyration


__all__ = ["NeighborGraph", "KNNGraph", "SizeRadiusGraph"]

# The distance filled in for the pairs that are no pair of two atoms of a molecule.
_FAR = 1e30


class NeighborGraph(nn.Module):
    """
    A neighbor graph built from the current positions, inside the model, on the
    device of the batch: in training, from the noisy positions, and at every step
    of a sampler, from the positions of that step. Nothing is stored in the batch,
    so the graph follows the atoms as they move.

    Every pair (i, j) of a graph has a weight w_ij = f(d_ij / r_ij) of the cosine
    cutoff f (1 at 0, 0 from 1 on) and a radius r_ij, chosen such that w_ij is
    exactly 0 when the pair is about to leave the graph. A graph that changes with
    the positions then changes continuously: a pair never enters or leaves with
    a message of full weight.

    The graph is symmetric, so a model can compute one filter per pair and use it
    for both directions (the half-pair filters of PaiNN). Subclasses define which
    pairs belong to it, in ``_adjacency``.

    Atoms of a molecule must be contiguous and molecules in order, like in every
    batch schnetpack collates (and the samplers select from).
    """

    @torch.compiler.disable  # data-dependent shapes: runs eagerly between two compiled graphs
    def forward(
        self,
        positions: torch.Tensor,
        idx_m: torch.Tensor,
        n_atoms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The pairs i < j of the graph, once each, and their weights.

        Args:
            positions: (total_atoms, 3) positions.
            idx_m: (total_atoms,) molecule of every atom.
            n_atoms: (n_molecules,) number of atoms of every molecule.

        Returns:
            idx_i, idx_j: (n_pairs,) atoms of the pairs, idx_i < idx_j. The graph
                holds the reversed pairs too.
            weight: (n_pairs,) w_ij in [0, 1], the same for (j, i).
        """
        counts = n_atoms.long()
        starts = torch.cumsum(counts, 0) - counts
        n_max = int(counts.max())
        device = positions.device
        slot = torch.arange(len(idx_m), device=device) - starts[idx_m]

        # molecules zero-padded to a common size: all pairs of a molecule at once, none across molecules
        padded = positions.new_zeros(len(counts), n_max, 3)
        padded[idx_m, slot] = positions
        atom = torch.arange(n_max, device=device) < counts[:, None]
        pair = atom[:, :, None] & atom[:, None, :]
        pair = pair & ~torch.eye(n_max, dtype=torch.bool, device=device)
        # the difference of the coordinates, not the matmul identity: exact for close atoms
        dist = torch.cdist(padded, padded, compute_mode="donot_use_mm_for_euclid_dist")
        dist = dist.masked_fill(~pair, _FAR)

        adjacency, rho = self._adjacency(dist, positions, idx_m, counts)

        upper = torch.arange(n_max, device=device)
        upper = upper[:, None] < upper[None, :]
        molecule, i, j = (adjacency & upper).nonzero(as_tuple=True)
        radius = torch.maximum(rho[molecule, i], rho[molecule, j])
        weight = cosine_cutoff(dist[molecule, i, j] / radius, 1.0)
        return starts[molecule] + i, starts[molecule] + j, weight

    def _adjacency(
        self,
        dist: torch.Tensor,
        positions: torch.Tensor,
        idx_m: torch.Tensor,
        counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dist: (n_molecules, n_max, n_max) distances, _FAR where i = j or an
                atom is padding.
            positions, idx_m, counts: as in forward, counts as integers.

        Returns:
            adjacency: (n_molecules, n_max, n_max) True for the pairs of the graph;
                symmetric, False on the diagonal and for padding.
            rho: (n_molecules, n_max) radius of every atom: a pair (i, j) is in the
                graph iff d_ij < max(rho_i, rho_j), and its weight is
                cosine_cutoff(d_ij / max(rho_i, rho_j)).
        """
        raise NotImplementedError


class KNNGraph(NeighborGraph):
    """
    The union of the k nearest neighbors of every atom, within its molecule: the
    pair (i, j) belongs to the graph if j is among the k nearest atoms of i or i is
    among those of j. Every atom has at least k neighbors and the graph is
    symmetric.

    rho_i is the distance of atom i to its (k+1)-th nearest atom, the first one
    outside its k neighbors. A pair is in the graph iff d_ij < max(rho_i, rho_j), so
    w_ij = cosine_cutoff(d_ij / max(rho_i, rho_j)) is exactly 0 when the pair leaves
    it, and positive while it is in. A molecule with at most k other atoms is fully
    connected with all weights 1: no pair can leave.
    """

    def __init__(self, k: int = 16):
        """
        Args:
            k: number of nearest neighbors of every atom.
        """
        super().__init__()
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def extra_repr(self) -> str:
        return f"k={self.k}"

    def _adjacency(self, dist, positions, idx_m, counts):
        n_max = dist.shape[-1]
        n_values = min(self.k + 1, n_max)
        values, index = dist.topk(n_values, dim=-1, largest=False)
        if n_values == self.k + 1:
            rho = values[..., -1]
            values, index = values[..., :-1], index[..., :-1]
        else:
            # every other atom is a neighbor
            rho = dist.new_full(dist.shape[:-1], _FAR)
        # neighbors a molecule does not have are padding or the atom itself, at _FAR
        adjacency = torch.zeros_like(dist, dtype=torch.bool).scatter_(-1, index, values < _FAR)
        return adjacency | adjacency.mT, rho


class SizeRadiusGraph(NeighborGraph):
    """
    All pairs within a radius that grows and shrinks with the size of the molecule:

    r = radius * R_g(x) / R_g,ref(N),    R_g,ref(N) = rg_ref_coefficient * N ** rg_ref_exponent,

    with R_g(x) the radius of gyration of the positions and R_g,ref(N) that of a
    clean molecule of N atoms. For a clean molecule it is the usual radius graph
    (r = ``radius`` Angstrom); at a noise level where the molecule has grown to a
    cloud of 10 times its size, so has the radius. Scaling all positions of a
    molecule scales r with them, so the graph does not change. This is the graph
    one gets by building it from x_t / R_g(x_t) * R_g,ref(N), the positions of a
    variance-preserving process rescaled to the scale of the data.

    The weight of a pair is cosine_cutoff(d_ij / r), 0 when the pair leaves.
    """

    def __init__(self, radius: float, rg_ref_coefficient: float, rg_ref_exponent: float):
        """
        Args:
            radius: radius in Angstrom of a molecule of its reference size.
            rg_ref_coefficient: c in R_g,ref(N) = c N^e, in Angstrom.
            rg_ref_exponent: e in R_g,ref(N) = c N^e.
        """
        super().__init__()
        if radius <= 0 or rg_ref_coefficient <= 0:
            raise ValueError("radius and rg_ref_coefficient must be positive")
        self.radius = radius
        self.rg_ref_coefficient = rg_ref_coefficient
        self.rg_ref_exponent = rg_ref_exponent

    def extra_repr(self) -> str:
        return (
            f"radius={self.radius}, "
            f"rg_ref={self.rg_ref_coefficient} * N ** {self.rg_ref_exponent}"
        )

    def _adjacency(self, dist, positions, idx_m, counts):
        rg = radius_of_gyration(positions, idx_m, counts)
        reference = self.rg_ref_coefficient * counts.to(rg.dtype) ** self.rg_ref_exponent
        radius = self.radius * rg / reference
        # a pair is no pair of the molecule at _FAR, which is beyond every radius
        adjacency = dist < radius[:, None, None]
        return adjacency, radius[:, None].expand(dist.shape[:2])
