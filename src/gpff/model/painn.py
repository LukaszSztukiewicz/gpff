from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing


__all__ = ["PaiNN", "VectorRMSNorm"]


class VectorRMSNorm(nn.Module):
    r"""Rotation-equivariant RMS normalization of vector features (n_atoms, 3, n_channels):
    :math:`\mu / \sqrt{\mathrm{mean}_c \lVert\mu_c\rVert^2 + \epsilon}` with a per-channel gain.

    With :math:`\epsilon = 1` (in squared feature units, like ``norm_epsilon`` of the
    directions) it is a soft normalization: about the identity for small features, so
    near-zero vectors (e.g. by symmetry) are not blown up, and bounded for large ones.
    """

    def __init__(self, n_channels: int, epsilon: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.epsilon = epsilon

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        ms = mu.pow(2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        return mu * torch.rsqrt(ms + self.epsilon) * self.weight


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    # class-level default, so models pickled before the option existed still load
    prenorm: bool = False

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
        prenorm: bool = False,
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
            prenorm: if True, every interaction and mixing block reads LayerNorm'ed
                scalars and VectorRMSNorm'ed vectors and adds its update to the
                unnormalized features, and the output features are normalized too.
                PaiNN's blocks multiply features with each other, so without it a
                feature excursion compounds over the blocks into an unbounded output;
                with it every update is bounded and the features grow at most
                linearly with depth.
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

        # one norm per block input (interaction, mixing) and one for the output
        self.prenorm = prenorm
        if prenorm:
            n_norms = 2 * self.n_interactions + 1
            self.scalar_norms = nn.ModuleList(nn.LayerNorm(n_atom_basis) for _ in range(n_norms))
            self.vector_norms = nn.ModuleList(VectorRMSNorm(n_atom_basis) for _ in range(n_norms))

    def _block(self, block: nn.Module, k: int, q: torch.Tensor, mu: torch.Tensor, *args):
        if not self.prenorm:
            return block(q, mu, *args)
        # schnetpack's blocks return their input plus the update; add the update to
        # the unnormalized features
        q_in, mu_in = self.scalar_norms[k](q), self.vector_norms[k](mu)
        q_out, mu_out = block(q_in, mu_in, *args)
        return q + (q_out - q_in), mu + (mu_out - mu_in)

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

        # compute atom and pair features
        d_ij = torch.sqrt(torch.sum(r_ij**2, dim=-1, keepdim=True) + self.epsilon)
        dir_ij = r_ij / (d_ij + self.norm_epsilon)
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
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
            q, mu = self._block(interaction, 2 * i, q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = self._block(mixing, 2 * i + 1, q, mu)
        if self.prenorm:
            q, mu = self.scalar_norms[-1](q), self.vector_norms[-1](mu)
        q = q.squeeze(1)

        # collect results
        inputs["scalar_representation"] = q
        if self.return_vector_representation:
            inputs["vector_representation"] = mu

        return inputs
