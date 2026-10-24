from typing import Callable, Dict, Optional, Union, List
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing


__all__ = ["PaiNN"]

# weak references to the (idx_i, idx_j) of the last call, and its _upper_pairs
_upper_pairs_cache = (lambda: None, lambda: None, None)


# Data-dependent shapes and control flow: under torch.compile this runs
# eagerly between two compiled graphs, which keeps the half-sized filters.
@torch.compiler.disable
def _upper_pairs(idx_i: torch.Tensor, idx_j: torch.Tensor, n_atoms: int) -> Optional[torch.Tensor]:
    """
    Indices of the pairs with i < j if every pair of the neighbor list is also
    listed reversed and no atom is paired with itself, else None.

    Remembered for the last neighbor list: a sampler passes the same idx_i and
    idx_j tensors at every step (and never modifies them).
    """
    global _upper_pairs_cache
    cached_i, cached_j, cached = _upper_pairs_cache
    if cached_i() is idx_i and cached_j() is idx_j:
        return cached
    upper = idx_i < idx_j
    key = torch.minimum(idx_i, idx_j) * n_atoms + torch.maximum(idx_i, idx_j)
    upper_idx = upper.nonzero().squeeze(-1)
    lower_idx = (~upper).nonzero().squeeze(-1)
    if not torch.equal(key[upper_idx].sort().values, key[lower_idx].sort().values):
        upper_idx = None
    _upper_pairs_cache = (weakref.ref(idx_i), weakref.ref(idx_j), upper_idx)
    return upper_idx


def _interaction(
    interaction: PaiNNInteraction,
    q: torch.Tensor,
    mu: torch.Tensor,
    Wij: torch.Tensor,
    dir_ij: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    n_atoms: int,
):
    """
    PaiNNInteraction.forward, except that Wij may hold the filters of only the
    first half of the pairs when the second half is the same pairs reversed.
    """
    x = interaction.interatomic_context_net(q)
    xj = x[idx_j]
    muj = mu[idx_j]
    if Wij.shape[0] == xj.shape[0]:
        x = Wij * xj
    else:
        # one filter for both directions of a pair
        x = (Wij * xj.unflatten(0, (2, -1))).flatten(0, 1)

    dq, dmuR, dmumu = torch.split(x, interaction.n_atom_basis, dim=-1)
    dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)
    dmu = dmuR * dir_ij[..., None] + dmumu * muj
    dmu = snn.scatter_add(dmu, idx_i, dim_size=n_atoms)

    q = q + dq
    mu = mu + dmu

    return q, mu


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = F.silu,
        shared_interactions: bool = False,
        return_vector_representation: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        norm_epsilon: float = 1,
        nuclear_embedding: Optional[nn.Module] = None,
        electronic_embeddings: Optional[List] = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: numerical stability parameter
            nuclear_embedding: custom nuclear embedding (e.g. spk.nn.embeddings.NuclearEmbedding)
            electronic_embeddings: list of electronic embeddings. E.g. for spin and
                charge (see spk.nn.embeddings.ElectronicEmbedding)
        """
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis
        self.return_vector_representation = return_vector_representation
        self.epsilon = epsilon
        self.norm_epsilon = norm_epsilon

        # initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(100, n_atom_basis)
        self.embedding = nuclear_embedding
        if electronic_embeddings is None:
            electronic_embeddings = []
        electronic_embeddings = nn.ModuleList(electronic_embeddings)
        self.electronic_embeddings = electronic_embeddings

        # initialize filter layers
        self.share_filters = shared_filters
        if shared_filters:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        # initialize interaction blocks
        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.n_atom_basis,
                activation=activation,
                epsilon=self.epsilon,
            ),
            self.n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        # A filter depends on |r_ij| alone, so the pair (j, i) shares the filter
        # of (i, j). If every pair is listed in both directions (like in the
        # fully connected lists gpff trains and samples with), compute the
        # filters of the pairs i < j only: the pair list becomes those pairs
        # followed by their reverses, whose r_ji = -r_ij exactly.
        upper = _upper_pairs(idx_i, idx_j, n_atoms)
        if upper is not None:
            r_ij, idx_i, idx_j = r_ij[upper], idx_i[upper], idx_j[upper]
            idx_i, idx_j = torch.cat([idx_i, idx_j]), torch.cat([idx_j, idx_i])

        # compute atom and pair features
        d_ij = torch.sqrt(torch.sum(r_ij**2, dim=-1, keepdim=True) + self.epsilon)
        dir_ij = r_ij / (d_ij + self.norm_epsilon)
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)[..., None]
        if upper is not None:
            dir_ij = torch.cat([dir_ij, -dir_ij])

        # filter_net(phi_ij) * fcut as one matmul: the inputs are scaled by fcut
        # and fcut is the input of the bias, so the cutoff takes no pass over the
        # (pairs, 3 * n_interactions * n_atom_basis) filters. Zero-padding the
        # inputs to a multiple of 8 features keeps the GPU on aligned kernels.
        n_pad = -(phi_ij.shape[-1] + 1) % 8
        filters = F.linear(
            torch.cat([phi_ij * fcut, fcut, fcut.new_zeros(*fcut.shape[:-1], n_pad)], dim=-1),
            F.pad(torch.cat([self.filter_net.weight, self.filter_net.bias[:, None]], dim=1), (0, n_pad)),
        )
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # compute initial embeddings
        q = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = _interaction(interaction, q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
        q = q.squeeze(1)

        # collect results
        inputs["scalar_representation"] = q
        if self.return_vector_representation:
            inputs["vector_representation"] = mu

        return inputs
