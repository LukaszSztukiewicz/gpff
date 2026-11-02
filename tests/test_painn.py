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


def painn(prenorm=False, degree_norm=False, neighbor_gate=False):
    torch.manual_seed(0)
    model = PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=10, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        return_vector_representation=True,
        prenorm=prenorm,
        degree_norm=degree_norm,
        neighbor_gate=neighbor_gate,
    ).double()
    with torch.no_grad():
        model.filter_net.bias.normal_()
        for name, weight in model.named_parameters():
            if name.startswith("gate_"):  # zero at first: gates of one
                weight.normal_(std=0.3)
    return model


def zero_gates(model):
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if name.startswith("gate_"):
                weight.zero_()
    return model


def representations(model, inputs):
    out = model(dict(inputs))
    return out["scalar_representation"], out["vector_representation"]


def reference(model, inputs):
    """PaiNN.forward over every pair, with the cutoff applied to the filters."""
    r_ij, idx_i, idx_j = inputs[properties.Rij], inputs[properties.idx_i], inputs[properties.idx_j]
    d_ij = torch.sqrt(torch.sum(r_ij**2, dim=-1, keepdim=True) + model.epsilon)
    dir_ij = r_ij / (d_ij + model.norm_epsilon)
    phi_ij = model.radial_basis(d_ij)
    filters = model.filter_net(phi_ij) * model.cutoff_fn(d_ij)[..., None]
    q = model.embedding(inputs[properties.Z]).unsqueeze(1)
    mu = q.new_zeros((q.shape[0], 3, q.shape[2]))
    degree = torch.bincount(idx_i, minlength=q.shape[0]).to(q.dtype)
    scale = degree.clamp(min=1).rsqrt()[:, None, None] if model.degree_norm else 1.0
    filter_list = [filters] * model.n_interactions if model.share_filters else torch.split(
        filters, 3 * model.n_atom_basis, dim=-1
    )
    for i, (filters_i, interaction, mixing) in enumerate(zip(filter_list, model.interactions, model.mixing)):
        q_in, mu_in = q, mu
        if model.prenorm:
            q_in, mu_in = model.scalar_norms[2 * i](q), model.vector_norms[2 * i](mu)
        if model.neighbor_gate:
            # the gate scales the message of the pair, and so its filter
            scores = q_in @ model.gate_atoms[i]
            distance = (phi_ij @ model.gate_radial + model.gate_bias)[..., i : i + 1]
            logit = scores[idx_i][..., :1] + scores[idx_j][..., 1:] + distance
            filters_i = filters_i * (2 * torch.sigmoid(logit))
        q_out, mu_out = interaction(q_in, mu_in, filters_i, dir_ij, idx_i, idx_j, q.shape[0])
        if model.prenorm or model.degree_norm:
            q, mu = q + (q_out - q_in) * scale, mu + (mu_out - mu_in) * scale
        else:
            q, mu = q_out, mu_out
        if model.prenorm:
            q_in, mu_in = model.scalar_norms[2 * i + 1](q), model.vector_norms[2 * i + 1](mu)
            q_out, mu_out = mixing(q_in, mu_in)
            q, mu = q + (q_out - q_in), mu + (mu_out - mu_in)
        else:
            q, mu = mixing(q, mu)
    if model.prenorm:
        q, mu = model.scalar_norms[-1](q), model.vector_norms[-1](mu)
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
@pytest.mark.parametrize("prenorm", [False, True])
@pytest.mark.parametrize("degree_norm", [False, True])
@pytest.mark.parametrize("neighbor_gate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)])
def test_matches_the_all_pairs_forward(drop_every, half_pairs, prenorm, degree_norm, neighbor_gate, dtype, device):
    model = painn(prenorm, degree_norm, neighbor_gate).to(device=device, dtype=dtype)
    inputs = {key: value.to(device=device) for key, value in batch(drop_every).items()}
    inputs[properties.R] = inputs[properties.R].to(dtype=dtype).detach().requires_grad_()
    inputs[properties.offsets] = inputs[properties.offsets].to(dtype=dtype)
    inputs = PairwiseDistances()(inputs)
    reference_inputs = dict(inputs)
    reference_inputs[properties.R] = inputs[properties.R].detach().clone().requires_grad_()
    reference_inputs = PairwiseDistances()(reference_inputs)
    assert (_upper_pairs(inputs[properties.idx_i], inputs[properties.idx_j], 20) is not None) == half_pairs

    out = model(dict(inputs))
    q, mu = reference(model, reference_inputs)
    tolerance = dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(out["scalar_representation"], q, **tolerance)
    torch.testing.assert_close(out["vector_representation"], mu, **tolerance)

    # and so do the gradients of a training loss
    loss = lambda q, mu: (q**2).sum() + (mu**2).sum()
    grads = torch.autograd.grad(
        loss(out["scalar_representation"], out["vector_representation"]),
        (*model.parameters(), inputs[properties.R]),
    )
    expected = torch.autograd.grad(loss(q, mu), (*model.parameters(), reference_inputs[properties.R]))
    names = [name for name, _ in model.named_parameters()] + ["positions"]
    for name, grad, grad_expected in zip(names, grads, expected):
        # a gradient sums many terms that cancel: its rounding error scales with its largest entry
        atol = tolerance["atol"] * max(1.0, grad_expected.abs().max().item())
        torch.testing.assert_close(
            grad, grad_expected, rtol=tolerance["rtol"], atol=atol, msg=lambda msg: f"{name}: {msg}"
        )


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


def test_gates_add_three_parameters_and_leave_every_other_weight_as_it_was():
    plain, gated = dict(painn().named_parameters()), dict(painn(neighbor_gate=True).named_parameters())

    assert set(gated) - set(plain) == {"gate_atoms", "gate_radial", "gate_bias"}
    assert set(plain) <= set(gated)
    for name, weight in plain.items():
        torch.testing.assert_close(gated[name], weight, rtol=0, atol=0)
    assert gated["gate_atoms"].shape == (2, 16, 2)  # interactions, features, (receiver, sender)
    assert gated["gate_radial"].shape == (10, 2)  # radial functions, interactions
    assert gated["gate_bias"].shape == (2,)
    assert not any(name.startswith("gate_") for name in painn().state_dict())


@pytest.mark.parametrize("drop_every", [None, 5])
@pytest.mark.parametrize("prenorm", [False, True])
def test_untrained_gates_leave_the_model_as_it_is(drop_every, prenorm):
    plain = painn(prenorm=prenorm, degree_norm=True)
    gated = zero_gates(painn(prenorm=prenorm, degree_norm=True, neighbor_gate=True))
    inputs = batch(drop_every)

    for want, got in zip(representations(plain, inputs), representations(gated, inputs)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("neighbor_gate", [False, True])
@pytest.mark.parametrize("prenorm", [False, True])
def test_scalars_are_invariant_and_vectors_equivariant_under_rotation_and_translation(neighbor_gate, prenorm):
    model = painn(prenorm=prenorm, degree_norm=True, neighbor_gate=neighbor_gate)
    inputs = batch()
    g = torch.Generator().manual_seed(1)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    rotation[:, 0] *= torch.det(rotation).sign()  # a proper rotation
    moved = dict(inputs)
    moved[properties.R] = inputs[properties.R] @ rotation.T + torch.randn(3, generator=g, dtype=torch.float64)
    moved = PairwiseDistances()(moved)

    q, mu = representations(model, inputs)
    q_moved, mu_moved = representations(model, moved)

    torch.testing.assert_close(q_moved, q)
    torch.testing.assert_close(mu_moved, torch.einsum("ij,njf->nif", rotation, mu))
    assert mu.abs().max() > 1e-3  # the vectors are not trivially zero


def test_gates_change_the_output_and_read_the_receiver_the_sender_and_the_distance():
    inputs = batch()
    gated = painn(neighbor_gate=True)
    base = representations(gated, inputs)[0]

    assert not torch.allclose(base, representations(painn(), inputs)[0])
    parts = {
        "receiver": lambda: gated.gate_atoms[..., 0],
        "sender": lambda: gated.gate_atoms[..., 1],
        "distance": lambda: gated.gate_radial,
        "bias": lambda: gated.gate_bias,
    }
    for name, get in parts.items():
        with torch.no_grad():
            get().add_(0.5)
        changed = representations(gated, inputs)[0]
        with torch.no_grad():
            get().sub_(0.5)
        assert not torch.allclose(changed, base), name


def test_untrained_gates_receive_gradients():
    model = zero_gates(painn(neighbor_gate=True))
    q, mu = representations(model, batch())

    ((q**2).sum() + (mu**2).sum()).backward()

    # every interaction's gate reads both atoms, the distance and has a bias
    assert (model.gate_atoms.grad.abs().sum(dim=1) > 0).all()
    assert (model.gate_radial.grad.abs().sum(dim=0) > 0).all()
    assert (model.gate_bias.grad.abs() > 0).all()


@pytest.mark.parametrize("neighbor_gate", [False, True])
def test_compiled_forward_matches_eager(neighbor_gate):
    model, inputs = painn(neighbor_gate=neighbor_gate), batch()
    eager = representations(model, inputs)

    out = torch.compile(model, backend="eager", dynamic=True)(dict(inputs))

    torch.testing.assert_close(out["scalar_representation"], eager[0])
    torch.testing.assert_close(out["vector_representation"], eager[1])
