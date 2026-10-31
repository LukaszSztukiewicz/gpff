from typing import Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing


__all__ = ["PaiNN"]


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
        embedding_key: str = properties.Z,
        sigma_conditioning: bool = False,
        sigma_floor: float = 1e-4,
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
            embedding_key: input the nuclear embedding reads, e.g. noisy one-hot
                atom types for a linear embedding instead of atomic numbers
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
        self.embedding_key = embedding_key
        if sigma_floor <= 0:
            raise ValueError("sigma_floor must be positive")
        self.sigma_floor = sigma_floor
        # No new parameters or required inputs when disabled. The zero output
        # initially preserves the pretrained representation when enabled.
        if sigma_conditioning:
            self.sigma_embedding = nn.Sequential(
                nn.Linear(1, n_atom_basis), nn.SiLU(),
                nn.Linear(n_atom_basis, n_atom_basis),
            )
            nn.init.zeros_(self.sigma_embedding[-1].weight)
            nn.init.zeros_(self.sigma_embedding[-1].bias)
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
        # models pickled before embedding_key existed embed the atomic numbers
        q = self.embedding(inputs[getattr(self, "embedding_key", properties.Z)])
        sigma_embedding = getattr(self, "sigma_embedding", None)
        if sigma_embedding is not None:
            from gpff import keys
            # Training transforms store one sigma per molecule. Scheduled
            # inference uses the same convention; per-atom storage is accepted
            # for compatibility with the original calculator.
            sigma = inputs[keys.sigma_curr].reshape(-1, 1)
            if len(sigma) != n_atoms:
                sigma = sigma[inputs[properties.idx_m]]
            if not bool(torch.isfinite(sigma).all()) or bool((sigma < 0).any()):
                raise ValueError("Declared sigma must be finite and nonnegative")
            q = q + sigma_embedding(sigma.clamp_min(self.sigma_floor).log())
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
        q = q.squeeze(1)

        # collect results
        inputs["scalar_representation"] = q
        if self.return_vector_representation:
            inputs["vector_representation"] = mu

        return inputs
