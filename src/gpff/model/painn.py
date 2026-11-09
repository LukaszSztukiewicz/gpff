from typing import Callable, Dict, Optional, Union, List, Tuple
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing
from gpff.nn.normalized import NormalizedDense, ScalarRMSNorm, normalize_linears, soft_unit_vectors


__all__ = ["PaiNN", "VectorRMSNorm", "centered_positions"]


def centered_positions(inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The positions relative to the centroid of their molecule, (n_atoms, 3), and the mean
    square distance of every molecule's atoms from it, :math:`\mathrm{mean}_i \lVert x_i - \bar x\rVert^2`,
    (n_molecules, 1). For VE noise of level :math:`\sigma` added to a molecule of per-axis RMS
    radius :math:`\rho_0`, the latter is about :math:`3 (\rho_0^2 + \sigma^2)`.
    """
    idx_m = inputs[properties.idx_m]
    n_atoms = inputs[properties.n_atoms]
    pos = inputs[properties.R]
    count = n_atoms.unsqueeze(-1).to(pos.dtype)
    centroid = snn.scatter_add(pos, idx_m, dim_size=n_atoms.shape[0]) / count
    x = pos - centroid[idx_m]
    ms = snn.scatter_add(x.pow(2).sum(-1, keepdim=True), idx_m, dim_size=n_atoms.shape[0]) / count
    return x, ms


class VectorRMSNorm(nn.Module):
    r"""Rotation-equivariant RMS normalization of vector features (n_atoms, 3, n_channels):
    :math:`\mu / \sqrt{\mathrm{mean}_c \lVert\mu_c\rVert^2 + \epsilon}` with a per-channel gain.

    With :math:`\epsilon = 1` (in squared feature units, like ``norm_epsilon`` of the
    directions) it is a soft normalization: about the identity for small features, so
    near-zero vectors (e.g. by symmetry) are not blown up, and bounded for large ones.
    """

    def __init__(self, n_channels: int, epsilon: float = 1.0, affine: bool = True):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
        else:
            self.register_buffer("weight", torch.ones(n_channels), persistent=False)
        self.epsilon = epsilon

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        ms = mu.pow(2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        return mu * torch.rsqrt(ms + self.epsilon) * self.weight

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
    gate=None,
    normalized_products: bool = False,
):
    """
    PaiNNInteraction.forward, except that Wij may hold the filters of only the
    first half of the pairs when the second half is the same pairs reversed, and
    that the message of every pair may be scaled by a gate.

    gate: None, or (atoms, radial): the weights (n_atom_basis, 2) that read the
    scalars of the receiving atom i and of the sending atom j, and the distance
    term of every pair (one per pair, or one per direction), all in the logit
    of the gate 2 * sigmoid(logit). The gate is an invariant number, so it
    scales the scalar and the vector parts of a message alike.
    """
    x = interaction.interatomic_context_net(q)
    xj = x[idx_j]
    muj = (soft_unit_vectors(mu) if normalized_products else mu)[idx_j]
    if Wij.shape[0] == xj.shape[0]:
        x = Wij * xj
    else:
        # one filter for both directions of a pair
        x = (Wij * xj.unflatten(0, (2, -1))).flatten(0, 1)
    if gate is not None:
        atoms, radial = gate
        scores = q @ atoms
        logit = scores[idx_i][..., :1] + scores[idx_j][..., 1:]
        if radial.shape[0] != logit.shape[0]:
            # one term for both directions of a pair
            radial = torch.cat([radial, radial])
        x = x * (2 * torch.sigmoid(logit + radial))

    dq, dmuR, dmumu = torch.split(x, interaction.n_atom_basis, dim=-1)
    dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)
    dmu = dmuR * dir_ij[..., None] + dmumu * muj
    dmu = snn.scatter_add(dmu, idx_i, dim_size=n_atoms)

    q = q + dq
    mu = mu + dmu

    return q, mu


class ProductNormalizedMixing(PaiNNMixing):
    """Separate invariant magnitude from vector direction in tensor products.

    Raw magnitudes still enter the scalar MLP. Direction/correlation operands
    have nonsingular unit-ball coordinates, removing the cubic amplification
    from (scalar gate) * (V dot W) and the quadratic gate * W feedback.
    """

    def forward(self, q, mu):
        v, w = self.mu_channel_mix(mu).split(self.n_atom_basis, dim=-1)
        magnitude = torch.sqrt(v.square().sum(dim=-2, keepdim=True) + self.epsilon)
        dq, dv, dqw = self.intraatomic_context_net(torch.cat([q, magnitude], dim=-1)).split(
            self.n_atom_basis, dim=-1
        )
        v, w = soft_unit_vectors(v), soft_unit_vectors(w)
        return q + dq + dqw * (v * w).sum(dim=-2, keepdim=True), mu + dv * w


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    # class-level default, so models pickled before the option existed still load
    prenorm: Union[bool, str] = False
    degree_norm: Union[bool, str] = False
    neighbor_gate: bool = False
    residual_scaling: bool = False
    normalized_products: bool = False
    radial_norm: bool = False
    sigma_data: Optional[float] = None

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
        prenorm: Union[bool, str] = False,
        degree_norm: Union[bool, str] = False,
        neighbor_gate: bool = False,
        weight_norm: bool = False,
        residual_scaling: bool = False,
        normalized_products: bool = False,
        radial_norm: bool = False,
        sigma_data: Optional[float] = None,
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
                'rms' uses mean-preserving soft scalar RMS and gain-free vector
                RMS instead: learned norm gains cannot undo the normalization.
            degree_norm: if True, the messages an interaction block sums over the
                neighbors of an atom are divided by the square root of their number,
                as NequIP divides by the square root of the average number of neighbors.
                The sum is not normalized otherwise, so with all-to-all neighbors the
                update of an atom grows with the size of its molecule (QM9: 4-28
                neighbors, GEOM-Drugs: up to ~180) while the features it is added to
                are normalized by ``prenorm``. This takes the exact degree of every
                atom, so no dataset constant enters. The count is not recoverable from a
                normalized sum, the network has to read it from the distances.
                'mean' divides by degree instead, also controlling coherent sums
                of correlated messages (e.g. same-element neighbors).
            neighbor_gate: if True, every interaction block scales the message of
                a neighbor j to an atom i by a gate in (0, 2), the sigmoid of the
                sum of a linear function of the scalars of i, one of the scalars
                of j and one of the radial basis of their distance (plus a bias),
                times two. The gate is invariant, so equivariance is kept, and it
                is not normalized over the neighbors: the sum still grows with
                their number, unlike an attention softmax. All its weights start
                at zero, so every gate starts at one and the model starts as the
                one without gates; they are created after all other weights, so
                that flag does not change the initialization of these.
            weight_norm: normalize linear weight rows on use and after optimizer
                updates (with AtomisticTask), following EDM2. Also covers filters.
            residual_scaling: scale each of the 2L residual updates by 1/sqrt(2L)
                and divide each sum by sqrt(1 + 1/(2L)). This limits the variance
                budget across depth; it is not a bound for correlated branches.
            normalized_products: use smooth direction coordinates in vector
                products, with magnitudes still supplied to the mixing MLP.
            radial_norm: unit-RMS radial features before the cutoff. Gaussian
                bases are sparse, so their active energy does not scale with fan-in.
            sigma_data: if set (in A), EDM's c_in and c_noise without the noise level as
                input. Distances are measured in units of the molecule's length scale
                s = sqrt(sigma_data^2 + max(rho^2 - sigma_data^2, 0)) = max(rho, sigma_data),
                with rho the per-axis RMS radius of the input molecule. VE noise gives
                rho^2 ~ rho_0^2 + sigma^2, so s is EDM's 1 / c_in = sqrt(sigma^2 + sigma_data^2)
                with the positive-part estimate of sigma^2 read off the geometry. A molecule
                no larger than sigma_data is seen in the fixed units sigma_data, so bond
                lengths keep their absolute size, as under EDM's c_in at small sigma; a
                larger noise cloud is seen in units of its own radius, so all large noise
                levels look alike. log(s / sigma_data), zero for the former, is added to the
                atom embeddings by a linear map that starts at zero (EDM's c_noise). Set
                sigma_data to the largest per-axis RMS radius of the clean molecules: below
                it, units that vary with the molecule would hide the absolute bond lengths.
                The cutoff, the radial basis and norm_epsilon are then in units of s.
        """
        super(PaiNN, self).__init__()
        if prenorm not in (False, True, "rms"):
            raise ValueError("prenorm must be false, true or 'rms'")
        if degree_norm not in (False, True, "mean"):
            raise ValueError("degree_norm must be false, true or 'mean'")

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
        mixing_cls = ProductNormalizedMixing if normalized_products else PaiNNMixing
        self.mixing = snn.replicate_module(
            lambda: mixing_cls(
                n_atom_basis=self.n_atom_basis,
                activation=activation,
                epsilon=self.epsilon,
            ),
            self.n_interactions,
            shared_interactions,
        )

        self.degree_norm = degree_norm
        self.normalized_products = normalized_products
        self.radial_norm = radial_norm
        self.residual_scaling = residual_scaling
        if residual_scaling and n_interactions < 1:
            raise ValueError("residual_scaling requires n_interactions >= 1")
        if weight_norm:
            normalize_linears(self)

        self.sigma_data = sigma_data
        if sigma_data is not None:
            if not sigma_data > 0:
                raise ValueError(f"sigma_data must be positive, got {sigma_data}")
            # no bias: a molecule seen in units of sigma_data gets no embedding at all.
            # skip_init: the random init of nn.Linear, overwritten by zeros anyway, would draw
            # from the global RNG and change the weights built after it (the output heads)
            # against the same run without sigma_data
            self.noise_embedding = nn.utils.skip_init(nn.Linear, 1, n_atom_basis, bias=False)
            nn.init.zeros_(self.noise_embedding.weight)

        # one norm per block input (interaction, mixing) and one for the output
        self.prenorm = prenorm
        if prenorm:
            n_norms = 2 * self.n_interactions + 1
            self.scalar_norms = nn.ModuleList(
                ScalarRMSNorm() if prenorm == "rms" else nn.LayerNorm(n_atom_basis) for _ in range(n_norms)
            )
            self.vector_norms = nn.ModuleList(
                VectorRMSNorm(n_atom_basis, affine=prenorm != "rms") for _ in range(n_norms)
            )

        # parameters, not layers, and zero: the gates draw no random numbers and start at one
        self.neighbor_gate = neighbor_gate
        if neighbor_gate:
            self.gate_atoms = nn.Parameter(torch.zeros(n_interactions, n_atom_basis, 2))
            self.gate_radial = nn.Parameter(torch.zeros(self.radial_basis.n_rbf, n_interactions))
            self.gate_bias = nn.Parameter(torch.zeros(n_interactions))

    def _block(
        self, block: nn.Module, k: int, q: torch.Tensor, mu: torch.Tensor, *args, update_scale=None, gate=None
    ):
        # schnetpack's blocks return their input plus the update; add the update to
        # the unnormalized features
        q_in, mu_in = (self.scalar_norms[k](q), self.vector_norms[k](mu)) if self.prenorm else (q, mu)
        if isinstance(block, PaiNNInteraction):
            q_out, mu_out = _interaction(
                block, q_in, mu_in, *args, gate=gate, normalized_products=self.normalized_products
            )
        else:
            q_out, mu_out = block(q_in, mu_in, *args)
        if not self.prenorm and update_scale is None and not self.residual_scaling:
            return q_out, mu_out
        dq, dmu = q_out - q_in, mu_out - mu_in
        if update_scale is not None:
            dq, dmu = dq * update_scale, dmu * update_scale
        if self.residual_scaling:
            scale = (2 * self.n_interactions) ** -0.5
            denom = (1 + scale**2) ** 0.5
            return (q + scale * dq) / denom, (mu + scale * dmu) / denom
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
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        # EDM's c_in: the pairs of a molecule in units of its length scale
        if self.sigma_data is not None:
            idx_m = inputs[properties.idx_m]
            _, ms = centered_positions(inputs)
            # clamped before the root, so a molecule at the floor gets no gradient through it
            scale = torch.sqrt(torch.clamp(ms / 3, min=self.sigma_data**2))
            r_ij = r_ij / scale[idx_m[idx_i]]

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
        if self.radial_norm:
            phi_ij = phi_ij * torch.rsqrt(phi_ij.square().mean(dim=-1, keepdim=True) + 1e-8)
        fcut = self.cutoff_fn(d_ij)[..., None]
        if upper is not None:
            dir_ij = torch.cat([dir_ij, -dir_ij])

        # filter_net(phi_ij) * fcut as one matmul: the inputs are scaled by fcut
        # and fcut is the input of the bias, so the cutoff takes no pass over the
        # (pairs, 3 * n_interactions * n_atom_basis) filters. Zero-padding the
        # inputs to a multiple of 8 features keeps the GPU on aligned kernels.
        n_pad = -(phi_ij.shape[-1] + 1) % 8
        filter_weight, filter_bias = (self.filter_net.normalized_parameters()
                                      if isinstance(self.filter_net, NormalizedDense) else
                                      (self.filter_net.weight, self.filter_net.bias))
        filters = F.linear(
            torch.cat([phi_ij * fcut, fcut, fcut.new_zeros(*fcut.shape[:-1], n_pad)], dim=-1),
            F.pad(torch.cat([filter_weight, filter_bias[:, None]], dim=1), (0, n_pad)),
        )
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # the distance term of the gates of every interaction: (pairs, 1, n_interactions)
        gate_radial = phi_ij @ self.gate_radial + self.gate_bias if self.neighbor_gate else None

        # the interactions sum their messages over the neighbors of an atom: divide by sqrt(degree)
        update_scale = None
        if self.degree_norm:
            # idx_i lists both directions even when d_ij holds only half the pairs.
            degree = snn.scatter_add(torch.ones_like(idx_i, dtype=d_ij.dtype), idx_i, dim_size=n_atoms)
            degree = degree.clamp(min=1)
            update_scale = (degree.reciprocal() if self.degree_norm == "mean" else
                            torch.rsqrt(degree))[:, None, None]

        # compute initial embeddings
        q = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            q = q + embedding(q, inputs)
        # EDM's c_noise: the absolute scale the units hide, zero below sigma_data
        if self.sigma_data is not None:
            q = q + self.noise_embedding(torch.log(scale / self.sigma_data))[idx_m]
        q = q.unsqueeze(1)

        # compute interaction blocks and update atomic embeddings
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            gate = (self.gate_atoms[i], gate_radial[..., i : i + 1]) if self.neighbor_gate else None
            q, mu = self._block(
                interaction, 2 * i, q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms,
                update_scale=update_scale, gate=gate,
            )
            q, mu = self._block(mixing, 2 * i + 1, q, mu)
        if self.prenorm:
            q, mu = self.scalar_norms[-1](q), self.vector_norms[-1](mu)
        q = q.squeeze(1)

        # collect results
        inputs["scalar_representation"] = q
        if self.return_vector_representation:
            inputs["vector_representation"] = mu

        return inputs
