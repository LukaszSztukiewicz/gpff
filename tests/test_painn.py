import pytest
import torch
from schnetpack import properties
from schnetpack.atomistic import PairwiseDistances
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF

import gpff.model.painn as painn_module
from gpff.model.painn import PaiNN, _upper_pairs
from gpff.utils.neighbors import compute_neighbors


def painn():
    torch.manual_seed(0)
    model = PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=10, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        return_vector_representation=True,
    ).double()
    with torch.no_grad():
        model.filter_net.bias.normal_()
    return model


def reference(model, inputs):
    """PaiNN.forward over every pair, with the cutoff applied to the filters."""
    r_ij, idx_i, idx_j = inputs[properties.Rij], inputs[properties.idx_i], inputs[properties.idx_j]
    d_ij = torch.sqrt(torch.sum(r_ij**2, dim=-1, keepdim=True) + model.epsilon)
    dir_ij = r_ij / (d_ij + model.norm_epsilon)
    filters = model.filter_net(model.radial_basis(d_ij)) * model.cutoff_fn(d_ij)[..., None]
    q = model.embedding(inputs[properties.Z]).unsqueeze(1)
    mu = torch.zeros((q.shape[0], 3, q.shape[2]))
    filter_list = torch.split(filters, 3 * model.n_atom_basis, dim=-1)
    for filters_i, interaction, mixing in zip(filter_list, model.interactions, model.mixing):
        q, mu = interaction(q, mu, filters_i, dir_ij, idx_i, idx_j, q.shape[0])
        q, mu = mixing(q, mu)
    return q.squeeze(1), mu


def batch(drop_every=None):
    g = torch.Generator().manual_seed(0)
    molecules = [
        {
            properties.idx: torch.tensor([k]),
            properties.n_atoms: torch.tensor([n]),
            properties.Z: torch.randint(1, 10, (n,), generator=g),
            properties.R: torch.randn(n, 3, generator=g, dtype=torch.float64) * 2,
            properties.cell: torch.zeros(1, 3, 3, dtype=torch.float64),
            properties.pbc: torch.zeros(3, dtype=torch.bool),
        }
        for k, n in enumerate([7, 2, 11])
    ]
    inputs = compute_neighbors(_atoms_collate_fn(molecules), fully_connected=True, device="cpu")
    if drop_every is not None:
        keep = torch.arange(len(inputs[properties.idx_i])) % drop_every != 0
        for key in (properties.idx_i, properties.idx_j, properties.offsets):
            inputs[key] = inputs[key][keep]
    return PairwiseDistances()(inputs)


@pytest.mark.parametrize("drop_every, half_pairs", [(None, True), (5, False)])
def test_matches_the_all_pairs_forward(drop_every, half_pairs):
    model, inputs = painn(), batch(drop_every)
    assert (_upper_pairs(inputs[properties.idx_i], inputs[properties.idx_j], 20) is not None) == half_pairs

    out = model(dict(inputs))
    q, mu = reference(model, inputs)
    torch.testing.assert_close(out["scalar_representation"], q)
    torch.testing.assert_close(out["vector_representation"], mu)

    # and so do the gradients of a training loss
    loss = lambda q, mu: (q**2).sum() + (mu**2).sum()
    grads = torch.autograd.grad(
        loss(out["scalar_representation"], out["vector_representation"]), model.parameters()
    )
    expected = torch.autograd.grad(loss(q, mu), model.parameters())
    for grad, grad_expected in zip(grads, expected):
        torch.testing.assert_close(grad, grad_expected)


def test_pairs_of_an_atom_with_itself_use_every_pair():
    idx_i, idx_j = torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2])
    assert _upper_pairs(idx_i, idx_j, 3) is None


def test_compiled_forward_keeps_the_half_pairs():
    model, inputs = painn(), batch()
    painn_module._upper_pairs_cache = (lambda: None, lambda: None, None)

    out = torch.compile(model, backend="eager", dynamic=True)(dict(inputs))

    # the symmetry check ran, eagerly, on this neighbor list
    assert painn_module._upper_pairs_cache[0]() is inputs[properties.idx_i]
    assert painn_module._upper_pairs_cache[2] is not None
    q, mu = reference(model, inputs)
    torch.testing.assert_close(out["scalar_representation"], q)
    torch.testing.assert_close(out["vector_representation"], mu)
