from typing import Callable, Dict, Optional, Union, List
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing

from gpff.model.geometry import MoleculeGeometry, molecule_geometry


__all__ = ["PaiNN", "VectorRMSNorm", "GlobalChannel"]


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


class GlobalChannel(nn.Module):
    r"""A virtual node per molecule, O(n_atoms), equivariant: pools the features of a molecule
    and its shape into one global state and gates it back into every atom.

    The molecule level reads the means of the (normalized) scalars and the norms of the means
    of the (normalized) vectors, plus the invariants of the geometry: log R_g and two
    invariants of the gyration tensor (see :class:`gpff.model.geometry.MoleculeGeometry`).
    That is how a time-agnostic model sees the size of a noisy molecule at a glance, whatever
    its graph. The atom level reads its scalars, the global state and its distance from the
    centroid in units of R_g, and gives

    - an update of the scalars,
    - a gate for the mean vector of the molecule,
    - a gate for the direction from the centroid to the atom, :math:`(x_i - c) / R_g`.

    The vector update is a sum of vectors times invariant gates, so it is equivariant, and the
    channel is invariant to translations and permutations of the atoms. The update starts at
    zero, so a model with a global channel starts as the one without.
    """

    def __init__(self, n_atom_basis: int, activation: Optional[Callable] = F.silu):
        super().__init__()
        self.scalar_norm = nn.LayerNorm(n_atom_basis)
        self.vector_norm = VectorRMSNorm(n_atom_basis)
        # scalar means, vector mean norms, log R_g and two shape invariants
        self.molecule_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis + 3, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        # scalars, global state and |x_i - c| / R_g
        self.atom_net = nn.Sequential(
            snn.Dense(2 * n_atom_basis + 1, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        nn.init.zeros_(self.atom_net[-1].weight)
        nn.init.zeros_(self.atom_net[-1].bias)

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        geometry: MoleculeGeometry,
        idx_m: torch.Tensor,
        n_atoms: torch.Tensor,
    ):
        """
        Args:
            q: (total_atoms, 1, n_atom_basis) scalar features.
            mu: (total_atoms, 3, n_atom_basis) vector features.
            geometry: molecule_geometry of the positions.
            idx_m: (total_atoms,) molecule of every atom.
            n_atoms: (n_molecules,) number of atoms of every molecule.

        Returns:
            q, mu: the features plus the update.
        """
        n_molecules = n_atoms.shape[0]
        count = n_atoms.to(q.dtype)
        s = self.scalar_norm(q.squeeze(1))
        v = self.vector_norm(mu)
        s_mean = snn.scatter_add(s, idx_m, dim_size=n_molecules) / count[:, None]
        v_mean = snn.scatter_add(v, idx_m, dim_size=n_molecules) / count[:, None, None]
        v_norm = torch.sqrt(v_mean.pow(2).sum(dim=-2) + 1e-8)
        shape = torch.cat([geometry.log_rg[:, None], geometry.shape], dim=-1).to(q.dtype)
        state = self.molecule_net(torch.cat([s_mean, v_norm, shape], dim=-1))

        radial = geometry.radial.to(q.dtype)
        dq, gate_mean, gate_direction = self.atom_net(torch.cat([s, state[idx_m], radial], dim=-1)).chunk(3, dim=-1)
        direction = geometry.direction.to(q.dtype)
        q = q + dq.unsqueeze(1)
        mu = mu + gate_mean.unsqueeze(1) * v_mean[idx_m] + gate_direction.unsqueeze(1) * direction.unsqueeze(-1)
        return q, mu


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

    # class-level default, so models pickled before the option existed still load
    prenorm: bool = False
    degree_norm: bool = False
    global_channel: bool = False

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
        degree_norm: bool = False,
        filter_hidden: Optional[int] = None,
        graph: Optional[nn.Module] = None,
        global_channel: bool = False,
        embedding_key: str = properties.Z,
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
            degree_norm: if True, the messages an interaction block sums over the
                neighbors of an atom are divided by the square root of their number,
                as NequIP divides by the square root of the average number of neighbors.
                The sum is not normalized otherwise, so with all-to-all neighbors the
                update of an atom grows with the size of its molecule (QM9: 4-28
                neighbors, GEOM-Drugs: up to ~180) while the features it is added to
                are normalized by ``prenorm``. This takes the exact degree of every
                atom, so no dataset constant enters. The count is not recoverable from a
                normalized sum, the network has to read it from the distances.
            filter_hidden: if set, the filters are a two-layer MLP of the radial basis
                with this many hidden units (and the activation) instead of one linear
                layer. A linear layer makes every filter a fixed combination of the basis
                functions; with a basis of few, wide functions (see
                :mod:`gpff.nn.radial`) the MLP can still make a sharp filter.
            graph: if set (see :mod:`gpff.model.graph`), the neighbor graph is built here,
                from the positions of every call, and the pairs in the batch (idx_i, idx_j,
                Rij) are not read. Its weights replace the radial ``cutoff_fn`` as the
                factor that takes a filter to zero when its pair leaves the graph.
            global_channel: if True, every interaction block is followed by a
                :class:`GlobalChannel` (a virtual node per molecule with the size and shape
                of the molecule), which needs ``idx_m`` and ``n_atoms`` in the inputs.
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
        if electronic_embeddings is None:
            electronic_embeddings = []
        electronic_embeddings = nn.ModuleList(electronic_embeddings)
        self.electronic_embeddings = electronic_embeddings

        # initialize filter layers
        self.share_filters = shared_filters
        n_filter_inputs = self.radial_basis.n_rbf
        if filter_hidden is not None:
            # the first layer of the filter MLP; filter_net is its last, as the only layer of the linear one
            self.filter_hidden_net = snn.Dense(n_filter_inputs, filter_hidden, activation=activation)
            n_filter_inputs = filter_hidden
        if shared_filters:
            self.filter_net = snn.Dense(
                n_filter_inputs, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                n_filter_inputs,
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

        self.degree_norm = degree_norm

        # one norm per block input (interaction, mixing) and one for the output
        self.prenorm = prenorm
        if prenorm:
            n_norms = 2 * self.n_interactions + 1
            self.scalar_norms = nn.ModuleList(nn.LayerNorm(n_atom_basis) for _ in range(n_norms))
            self.vector_norms = nn.ModuleList(VectorRMSNorm(n_atom_basis) for _ in range(n_norms))

        # last, so the weights above are initialized as without these options
        self.graph = graph
        self.global_channel = global_channel
        if global_channel:
            self.global_channels = snn.replicate_module(
                lambda: GlobalChannel(n_atom_basis, activation=activation),
                self.n_interactions,
                shared_interactions,
            )

    def _block(self, block: nn.Module, k: int, q: torch.Tensor, mu: torch.Tensor, *args, update_scale=None):
        # schnetpack's blocks return their input plus the update; add the update to
        # the unnormalized features
        q_in, mu_in = (self.scalar_norms[k](q), self.vector_norms[k](mu)) if self.prenorm else (q, mu)
        if isinstance(block, PaiNNInteraction):
            q_out, mu_out = _interaction(block, q_in, mu_in, *args)
        else:
            q_out, mu_out = block(q_in, mu_in, *args)
        if not self.prenorm and update_scale is None:
            return q_out, mu_out
        dq, dmu = q_out - q_in, mu_out - mu_in
        if update_scale is not None:
            dq, dmu = dq * update_scale, dmu * update_scale
        return q + dq, mu + dmu

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
        n_atoms = atomic_numbers.shape[0]

        # A filter depends on |r_ij| alone, so the pair (j, i) shares the filter
        # of (i, j). If every pair is listed in both directions (like in the
        # fully connected lists gpff trains and samples with, and in the graphs
        # of gpff.model.graph), compute the filters of the pairs i < j only: the
        # pair list becomes those pairs followed by their reverses, whose
        # r_ji = -r_ij exactly.
        graph = getattr(self, "graph", None)
        weight = None
        if graph is not None:
            # the graph of the positions of this call, as the pairs i < j of a symmetric list
            positions = inputs[properties.R]
            idx_i, idx_j, weight = graph(positions, inputs[properties.idx_m], inputs[properties.n_atoms])
            r_ij = positions[idx_j] - positions[idx_i]
            half_pairs = True
        else:
            r_ij = inputs[properties.Rij]
            idx_i = inputs[properties.idx_i]
            idx_j = inputs[properties.idx_j]
            upper = _upper_pairs(idx_i, idx_j, n_atoms)
            half_pairs = upper is not None
            if half_pairs:
                r_ij, idx_i, idx_j = r_ij[upper], idx_i[upper], idx_j[upper]
        if half_pairs:
            idx_i, idx_j = torch.cat([idx_i, idx_j]), torch.cat([idx_j, idx_i])

        # compute atom and pair features
        d_ij = torch.sqrt(torch.sum(r_ij**2, dim=-1, keepdim=True) + self.epsilon)
        dir_ij = r_ij / (d_ij + self.norm_epsilon)
        phi_ij = self.radial_basis(d_ij)
        filter_hidden_net = getattr(self, "filter_hidden_net", None)
        if filter_hidden_net is not None:
            phi_ij = filter_hidden_net(phi_ij)
        # the graph's weights are its cutoff: 0 exactly when a pair leaves the graph
        if weight is None:
            fcut = self.cutoff_fn(d_ij)[..., None]
        else:
            fcut = weight.to(d_ij.dtype)[:, None, None]
        if half_pairs:
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

        # the interactions sum their messages over the neighbors of an atom: divide by sqrt(degree)
        update_scale = None
        if self.degree_norm:
            # idx_i lists both directions even when d_ij holds only half the pairs.
            degree = snn.scatter_add(torch.ones_like(idx_i, dtype=d_ij.dtype), idx_i, dim_size=n_atoms)
            update_scale = torch.rsqrt(degree.clamp(min=1))[:, None, None]

        # compute initial embeddings
        # models pickled before embedding_key existed embed the atomic numbers
        q = self.embedding(inputs[getattr(self, "embedding_key", properties.Z)])
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        global_channels = getattr(self, "global_channels", None)
        if global_channels is not None:
            idx_m, molecule_sizes = inputs[properties.idx_m], inputs[properties.n_atoms]
            geometry = molecule_geometry(inputs[properties.R], idx_m, molecule_sizes)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = self._block(
                interaction, 2 * i, q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms, update_scale=update_scale
            )
            q, mu = self._block(mixing, 2 * i + 1, q, mu)
            if global_channels is not None:
                q, mu = global_channels[i](q, mu, geometry, idx_m, molecule_sizes)
        if self.prenorm:
            q, mu = self.scalar_norms[-1](q), self.vector_norms[-1](mu)
        q = q.squeeze(1)

        # collect results
        inputs["scalar_representation"] = q
        if self.return_vector_representation:
            inputs["vector_representation"] = mu

        return inputs
