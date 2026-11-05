import pytest
import torch
from schnetpack import properties
from schnetpack.model import NeuralNetworkPotential
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF

from gpff import keys
from gpff.model.heads import GatedEquivariant
from gpff.model.painn import GlobalChannel, PaiNN
from gpff.model.graph import KNNGraph, SizeRadiusGraph
from gpff.sample.base import Calculator

GRAPHS = {
    "knn": lambda: KNNGraph(3),
    "size_radius": lambda: SizeRadiusGraph(radius=2.5, rg_ref_coefficient=1.2, rg_ref_exponent=0.45),
}


def inputs_for(sizes, seed=0, dtype=torch.float64):
    """a batch without a neighbor list: the model builds the graph"""
    g = torch.Generator().manual_seed(seed)
    n_atoms = torch.tensor(sizes)
    total = int(n_atoms.sum())
    return {
        properties.Z: torch.randint(1, 10, (total,), generator=g),
        properties.R: torch.randn(total, 3, generator=g, dtype=dtype) * 2,
        properties.n_atoms: n_atoms,
        properties.idx_m: torch.repeat_interleave(torch.arange(len(sizes)), n_atoms),
    }


def painn(graph=None, global_channel=False, filter_hidden=None, prenorm=False, degree_norm=False):
    torch.manual_seed(0)
    model = PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=10, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        return_vector_representation=True,
        prenorm=prenorm,
        degree_norm=degree_norm,
        filter_hidden=filter_hidden,
        graph=graph,
        global_channel=global_channel,
    ).double()
    with torch.no_grad():
        model.filter_net.bias.normal_()
    return model


def reference(model, inputs):
    """PaiNN.forward pair by pair, over both directions of every pair of the graph, weighed by the graph."""
    positions = inputs[properties.R]
    i, j, w = model.graph(positions, inputs[properties.idx_m], inputs[properties.n_atoms])
    idx_i, idx_j, weight = torch.cat([i, j]), torch.cat([j, i]), torch.cat([w, w])
    r_ij = positions[idx_j] - positions[idx_i]
    d_ij = torch.sqrt(torch.sum(r_ij**2, dim=-1, keepdim=True) + model.epsilon)
    dir_ij = r_ij / (d_ij + model.norm_epsilon)
    phi = model.radial_basis(d_ij)
    if getattr(model, "filter_hidden_net", None) is not None:
        phi = model.filter_hidden_net(phi)
    filters = model.filter_net(phi) * weight[:, None, None]
    q = model.embedding(inputs[properties.Z]).unsqueeze(1)
    mu = q.new_zeros((q.shape[0], 3, q.shape[2]))
    degree = torch.bincount(idx_i, minlength=q.shape[0]).to(q.dtype)
    scale = degree.clamp(min=1).rsqrt()[:, None, None] if model.degree_norm else 1.0
    for k, (filters_k, interaction, mixing) in enumerate(
        zip(torch.split(filters, 3 * model.n_atom_basis, dim=-1), model.interactions, model.mixing)
    ):
        q_in, mu_in = q, mu
        if model.prenorm:
            q_in, mu_in = model.scalar_norms[2 * k](q), model.vector_norms[2 * k](mu)
        q_out, mu_out = interaction(q_in, mu_in, filters_k, dir_ij, idx_i, idx_j, q.shape[0])
        if model.prenorm or model.degree_norm:
            q, mu = q + (q_out - q_in) * scale, mu + (mu_out - mu_in) * scale
        else:
            q, mu = q_out, mu_out
        if model.prenorm:
            q_in, mu_in = model.scalar_norms[2 * k + 1](q), model.vector_norms[2 * k + 1](mu)
            q_out, mu_out = mixing(q_in, mu_in)
            q, mu = q + (q_out - q_in), mu + (mu_out - mu_in)
        else:
            q, mu = mixing(q, mu)
    if model.prenorm:
        q, mu = model.scalar_norms[-1](q), model.vector_norms[-1](mu)
    return q.squeeze(1), mu


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("filter_hidden", [None, 8])
@pytest.mark.parametrize("prenorm, degree_norm", [(False, False), (True, True)])
def test_model_with_a_graph_matches_the_pair_by_pair_forward(graph, filter_hidden, prenorm, degree_norm):
    model = painn(GRAPHS[graph](), filter_hidden=filter_hidden, prenorm=prenorm, degree_norm=degree_norm)
    inputs = inputs_for([7, 2, 11, 1])
    inputs[properties.R].requires_grad_()
    reference_inputs = dict(inputs)
    reference_inputs[properties.R] = inputs[properties.R].detach().clone().requires_grad_()

    out = model(dict(inputs))
    q, mu = reference(model, reference_inputs)

    tolerance = dict(rtol=1e-9, atol=1e-10)
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
        torch.testing.assert_close(grad, grad_expected, **tolerance, msg=lambda msg: f"{name}: {msg}")


def test_graph_decides_the_pairs_the_batch_does_not():
    model = painn(KNNGraph(3))
    inputs = inputs_for([9, 6])
    # a neighbor list in the batch is not read
    inputs[properties.idx_i] = torch.tensor([0])
    inputs[properties.idx_j] = torch.tensor([0])
    inputs[properties.Rij] = torch.zeros(1, 3, dtype=torch.float64)

    out = model(dict(inputs))

    q, mu = reference(model, inputs)
    torch.testing.assert_close(out["scalar_representation"], q)


def random_rotation(reflect, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if reflect:
        q[:, 0] = -q[:, 0]
    return q


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("graph", GRAPHS)
def test_representation_is_equivariant_and_does_not_depend_on_the_order_of_atoms(graph, reflect):
    model = painn(GRAPHS[graph](), global_channel=True, filter_hidden=8)
    # a channel whose update is not zero
    torch.manual_seed(5)
    for parameter in model.global_channels.parameters():
        parameter.data.normal_(0, 0.3)
    sizes = [12, 7, 1, 9]
    inputs = inputs_for(sizes)
    out = model(dict(inputs))

    # rotated, reflected and moved: scalars stay, vectors turn
    q, shift = random_rotation(reflect), torch.tensor([5.0, -3.0, 40.0], dtype=torch.float64)
    moved = dict(inputs)
    moved[properties.R] = inputs[properties.R] @ q.T + shift
    out_moved = model(moved)
    torch.testing.assert_close(out_moved["scalar_representation"], out["scalar_representation"], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        out_moved["vector_representation"],
        torch.einsum("ab,nbf->naf", q, out["vector_representation"]),
        rtol=1e-8,
        atol=1e-8,
    )

    # atoms of a molecule in another order
    starts = torch.cumsum(torch.tensor(sizes), 0) - torch.tensor(sizes)
    g = torch.Generator().manual_seed(2)
    perm = torch.cat([s + torch.randperm(n, generator=g) for s, n in zip(starts.tolist(), sizes)])
    shuffled = dict(inputs)
    shuffled[properties.Z] = inputs[properties.Z][perm]
    shuffled[properties.R] = inputs[properties.R][perm]
    out_shuffled = model(shuffled)
    torch.testing.assert_close(out_shuffled["scalar_representation"], out["scalar_representation"][perm], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(out_shuffled["vector_representation"], out["vector_representation"][perm], rtol=1e-8, atol=1e-8)


def test_global_channel_starts_as_the_model_without_it_and_learns():
    plain = painn(KNNGraph(3))
    with_channel = painn(KNNGraph(3), global_channel=True)
    with_channel.load_state_dict(plain.state_dict(), strict=False)
    inputs = inputs_for([12, 7, 9])

    out_plain = plain(dict(inputs))
    out = with_channel(dict(inputs))

    torch.testing.assert_close(out["scalar_representation"], out_plain["scalar_representation"], rtol=0, atol=0)
    torch.testing.assert_close(out["vector_representation"], out_plain["vector_representation"], rtol=0, atol=0)
    # the update has gradients even though it starts at 0
    (out["scalar_representation"].pow(2).sum() + out["vector_representation"].pow(2).sum()).backward()
    assert all(
        channel.atom_net[-1].weight.grad.abs().sum() > 0 for channel in with_channel.global_channels
    )


def test_global_channel_is_per_block_and_shared_with_the_interactions():
    assert len(painn(KNNGraph(3), global_channel=True).global_channels) == 2
    shared = PaiNN(
        n_atom_basis=16,
        n_interactions=3,
        radial_basis=GaussianRBF(n_rbf=10, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        shared_interactions=True,
        global_channel=True,
    )
    assert shared.global_channels[0] is shared.global_channels[2]
    assert isinstance(shared.global_channels[0], GlobalChannel)


def test_defaults_add_nothing_to_the_model():
    model = painn()

    assert getattr(model, "graph", None) is None
    assert not model.global_channel
    assert not any("global" in name or "hidden" in name for name in model.state_dict())
    assert model.filter_net.in_features == 10


def test_filter_mlp_has_a_hidden_layer_of_the_given_size():
    model = painn(filter_hidden=8)

    assert model.filter_hidden_net.in_features == 10 and model.filter_hidden_net.out_features == 8
    assert model.filter_net.in_features == 8


def test_calculator_with_a_graph_model_needs_no_neighbor_list():
    torch.manual_seed(0)
    model = NeuralNetworkPotential(
        representation=painn(KNNGraph(3), global_channel=True),
        input_modules=[],
        output_modules=[GatedEquivariant(n_in=16)],
    ).double()
    calculator = Calculator(model=model, device="cpu", dtype=torch.float64)
    inputs = inputs_for([5, 1, 7, 3, 9])
    inputs[keys.converged] = torch.zeros(5, dtype=torch.bool)

    assert calculator.builds_graph
    assert properties.idx_i not in calculator.update_neighbors(inputs)
    full = calculator._predict(dict(inputs))[keys.forces]

    # converged molecules skip the model, and the others are predicted as in the full batch
    inputs[keys.converged] = torch.tensor([False, True, True, False, False])
    inputs[keys.forces] = full
    prediction = calculator._predict(inputs)[keys.forces]

    active = ~inputs[keys.converged][inputs[properties.idx_m]]
    torch.testing.assert_close(prediction[active], full[active])
    assert (prediction[~active] == 0).all()
    assert properties.idx_i not in inputs


def test_calculator_builds_a_neighbor_list_for_a_model_without_a_graph():
    model = NeuralNetworkPotential(
        representation=painn(), input_modules=[], output_modules=[GatedEquivariant(n_in=16)]
    )

    assert not Calculator(model=model, device="cpu").builds_graph
    assert not Calculator(device="cpu").builds_graph


def test_compiled_model_with_a_graph_matches_the_eager_one():
    model = painn(KNNGraph(3), global_channel=True, filter_hidden=8)
    inputs = inputs_for([7, 2, 11])

    expected = model(dict(inputs))
    out = torch.compile(model, backend="eager", dynamic=True)(dict(inputs))

    torch.testing.assert_close(out["scalar_representation"], expected["scalar_representation"])
    torch.testing.assert_close(out["vector_representation"], expected["vector_representation"])


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@needs_cuda
@pytest.mark.parametrize("graph", GRAPHS)
def test_model_with_a_graph_trains_on_the_gpu_in_deterministic_mode_like_on_the_cpu(graph):
    cpu_model = painn(GRAPHS[graph](), global_channel=True, filter_hidden=8)
    torch.manual_seed(5)
    for parameter in cpu_model.global_channels.parameters():
        parameter.data.normal_(0, 0.3)
    model = painn(GRAPHS[graph](), global_channel=True, filter_hidden=8).float().cuda()
    model.load_state_dict(cpu_model.state_dict())
    inputs = inputs_for([40, 7, 25, 1])
    inputs_cuda = {k: (v.float() if v.is_floating_point() else v).cuda() for k, v in inputs.items()}
    inputs_cuda[properties.R].requires_grad_()

    torch.use_deterministic_algorithms(True)
    try:
        out = model(inputs_cuda)
        (out["scalar_representation"].pow(2).sum() + out["vector_representation"].pow(2).sum()).backward()
    finally:
        torch.use_deterministic_algorithms(False)

    expected = cpu_model(dict(inputs))
    for key in ("scalar_representation", "vector_representation"):
        torch.testing.assert_close(out[key].detach().double().cpu(), expected[key], rtol=1e-3, atol=1e-3)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.isfinite(inputs_cuda[properties.R].grad).all()
