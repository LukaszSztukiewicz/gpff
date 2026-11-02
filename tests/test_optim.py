import copy
import math

import pytest
import torch
import torch.nn.functional as F
from schnetpack.atomistic import PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.nn import build_gated_equivariant_mlp
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from schnetpack.representation.painn import PaiNNInteraction, PaiNNMixing
from torch import nn

from gpff.model.heads import GatedEquivariant
from gpff.model.painn import PaiNN
from gpff.optim import (
    BlockAdamW,
    block_param_groups,
    eigenbasis,
    newton_schulz,
    normalized,
    orthogonalized,
    refreshed_basis,
    spectral_norm,
)
from gpff.task import AtomisticTask

LR = 1e-2


def last_delta(block_update, steps=3, shape=(12, 8), block=True, **kwargs):
    """The change of a weight by the last of a few steps on random gradients (the same for every update)."""
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(shape, generator=g, dtype=torch.float64))
    optimizer = BlockAdamW([{"params": [weight], "block": block}], lr=LR, block_update=block_update, **kwargs)
    for _ in range(steps):
        before = weight.detach().clone()
        weight.grad = torch.randn(shape, generator=g, dtype=torch.float64)
        optimizer.step()
    return before - weight.detach()


@pytest.mark.parametrize("block", [False, True])
def test_adam_steps_match_torch_adamw(block):
    torch.manual_seed(0)
    reference = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    model = copy.deepcopy(reference)
    kwargs = dict(lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
    expected = torch.optim.AdamW(reference.parameters(), **kwargs)
    weights = [p for p in model.parameters() if p.ndim == 2]
    biases = [p for p in model.parameters() if p.ndim == 1]
    actual = BlockAdamW([{"params": weights, "block": block}, {"params": biases}], block_update="adam", **kwargs)

    for _ in range(5):
        x = torch.randn(7, 6, dtype=torch.float64)
        for net, optimizer in ((reference, expected), (model, actual)):
            optimizer.zero_grad()
            net(x).pow(2).sum().backward()
            optimizer.step()

    for want, got in zip(reference.parameters(), model.parameters()):
        torch.testing.assert_close(got.detach(), want.detach())


def test_rms_step_has_rms_lr_and_the_direction_of_adam():
    adam, rms = last_delta("adam"), last_delta("rms")

    assert not torch.allclose(adam.pow(2).mean().sqrt(), torch.tensor(LR, dtype=torch.float64))
    assert rms.pow(2).mean().sqrt().item() == pytest.approx(LR)
    torch.testing.assert_close(rms, adam * LR / adam.pow(2).mean().sqrt())


def test_spectral_step_norm_is_at_most_lr_and_the_direction_is_adams():
    adam, step = last_delta("adam"), last_delta("spectral")

    spectral = torch.linalg.matrix_norm(step, 2).item()
    assert LR / 8 ** (1 / 16) * (1 - 1e-9) <= spectral <= LR * (1 + 1e-9)  # the matrix has rank 8
    torch.testing.assert_close(step / step.norm(), adam / adam.norm())


@pytest.mark.parametrize("block_update", ["rms", "spectral", "muon", "soap"])
def test_zero_gradient_leaves_a_block_unchanged(block_update):
    weight = nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
    weight.grad = torch.zeros_like(weight)
    before = weight.detach().clone()

    BlockAdamW([{"params": [weight], "block": True}], lr=LR, block_update=block_update).step()

    assert torch.equal(weight.detach(), before)


def test_spectral_norm_bounds_the_largest_singular_value():
    g = torch.Generator().manual_seed(0)
    rank_one = torch.randn(9, 1, generator=g, dtype=torch.float64) @ torch.randn(1, 6, generator=g, dtype=torch.float64)
    torch.testing.assert_close(spectral_norm(rank_one), torch.linalg.matrix_norm(rank_one, 2))

    rank_five = torch.randn(40, 5, generator=g, dtype=torch.float64)
    for matrix in (rank_five, rank_five.T):
        exact = torch.linalg.matrix_norm(matrix, 2)
        assert exact * (1 - 1e-9) <= spectral_norm(matrix) <= exact * 5 ** (1 / 16) * (1 + 1e-9)

    # the 16th power of a large float32 matrix would overflow without scaling
    assert torch.isfinite(spectral_norm(1e6 * rank_five.float()))


def test_newton_schulz_puts_the_singular_values_near_one():
    g = torch.Generator().manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(20, 12, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(12, 12, generator=g, dtype=torch.float64))
    matrix = u @ torch.diag(torch.linspace(0.5, 2.0, 12, dtype=torch.float64)) @ v.T

    for m in (matrix, matrix.T):
        singular_values = torch.linalg.svdvals(newton_schulz(m))
        assert 0.4 < singular_values.min() and singular_values.max() < 1.6
    torch.testing.assert_close(newton_schulz(10 * matrix), newton_schulz(matrix))


def test_first_muon_step_is_the_scaled_orthogonalized_gradient():
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(12, 8, generator=g, dtype=torch.float64))
    weight.grad = torch.randn(12, 8, generator=g, dtype=torch.float64)
    before, grad = weight.detach().clone(), weight.grad.clone()

    BlockAdamW([{"params": [weight], "block": True}], lr=LR, block_update="muon", momentum=0.9).step()

    # Nesterov momentum only rescales the gradient of the first step
    expected = LR * 0.2 * math.sqrt(12) * newton_schulz(grad)
    torch.testing.assert_close(before - weight.detach(), expected)


def step_fused_and_separate(block_update, roles=3, group_roles=None, rows=6, columns=8, steps=4, **kwargs):
    """A weight of ``roles`` stacked matrices, stepped as one block group with ``group_roles`` (default: ``roles``),
    next to the same matrices as separate parameters. The gradient of each role has its own scale (0.1, 1, 10, ...)."""
    g = torch.Generator().manual_seed(0)
    fused = nn.Parameter(torch.randn(roles * rows, columns, generator=g, dtype=torch.float64))
    separate = [nn.Parameter(role.clone()) for role in fused.detach().chunk(roles)]
    optimizers = (
        BlockAdamW(
            [{"params": [fused], "block": True, "roles": group_roles or roles}],
            lr=LR, block_update=block_update, **kwargs,
        ),
        BlockAdamW([{"params": separate, "block": True}], lr=LR, block_update=block_update, **kwargs),
    )
    scales = torch.tensor([10.0 ** (k - 1) for k in range(roles)], dtype=torch.float64).repeat_interleave(rows)[:, None]
    for _ in range(steps):
        grad = torch.randn(roles * rows, columns, generator=g, dtype=torch.float64) * scales
        fused.grad = grad
        for p, role in zip(separate, grad.chunk(roles)):
            p.grad = role.clone()
        for optimizer in optimizers:
            optimizer.step()
    return fused.detach(), torch.cat([p.detach() for p in separate])


@pytest.mark.parametrize("block_update", ["adam", "rms", "spectral", "muon"])
def test_a_fused_layer_steps_like_separate_layers_for_its_roles(block_update):
    fused, separate = step_fused_and_separate(block_update)

    torch.testing.assert_close(fused, separate)


@pytest.mark.parametrize("block_update", ["rms", "spectral", "muon"])
def test_stepping_the_roles_apart_differs_from_stepping_their_stack(block_update):
    apart, _ = step_fused_and_separate(block_update)
    stack, separate = step_fused_and_separate(block_update, group_roles=1)

    assert not torch.allclose(apart, stack)
    assert not torch.allclose(stack, separate)


def test_first_muon_step_of_a_fused_layer_has_the_scale_of_each_role():
    g = torch.Generator().manual_seed(0)
    roles, rows, columns = 3, 6, 8
    weight = nn.Parameter(torch.randn(roles * rows, columns, generator=g, dtype=torch.float64))
    scales = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64).repeat_interleave(rows)[:, None]
    weight.grad = torch.randn(roles * rows, columns, generator=g, dtype=torch.float64) * scales
    before, grad = weight.detach().clone(), weight.grad.clone()

    BlockAdamW([{"params": [weight], "block": True, "roles": roles}], lr=LR, block_update="muon", momentum=0.9).step()

    steps = (before - weight.detach()).chunk(roles)
    # the scale is Muon's for a matrix of the role's shape (sqrt(8)), not of the whole weight (sqrt(18))
    expected = [LR * 0.2 * math.sqrt(columns) * newton_schulz(role) for role in grad.chunk(roles)]
    for step, want in zip(steps, expected):
        torch.testing.assert_close(step, want)
    # a role takes a step of the same size whatever the size of its gradient
    rms = torch.stack([step.pow(2).mean().sqrt() for step in steps])
    assert rms.max() / rms.min() < 1.5


def test_the_roles_of_a_group_have_to_split_its_rows():
    for roles in (3, 0):
        with pytest.raises(ValueError, match="roles"):
            BlockAdamW([{"params": [nn.Parameter(torch.zeros(10, 4))], "block": True, "roles": roles}])


@pytest.mark.parametrize(
    "block_update, roles",
    [(update, roles) for update in ("adam", "rms", "spectral", "muon") for roles in (1, 2)] + [("soap", 1)],
)
def test_every_update_fits_a_linear_regression(block_update, roles):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 8, generator=g, dtype=torch.float64)
    y = x @ torch.randn(8, 4, generator=g, dtype=torch.float64)
    layer = nn.Linear(8, 4, bias=False).double()
    optimizer = BlockAdamW(
        [{"params": list(layer.parameters()), "block": True, "roles": roles}], lr=0.05, block_update=block_update
    )

    losses = []
    for _ in range(300):
        optimizer.zero_grad()
        loss = (layer(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < 0.1 * losses[0]


def test_block_groups_have_to_be_matrices():
    with pytest.raises(ValueError):
        BlockAdamW([{"params": [nn.Parameter(torch.zeros(3))], "block": True}])
    with pytest.raises(ValueError):
        BlockAdamW([nn.Parameter(torch.zeros(3, 3))], block_update="sgd")


class Net(nn.Module):
    """Equivariant blocks next to the layers and parameters that are no blocks."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 8)
        self.filter_net = nn.Linear(4, 24)
        self.interaction = PaiNNInteraction(8, F.silu)
        self.mixing = PaiNNMixing(8, F.silu)
        self.head = build_gated_equivariant_mlp(n_in=8, n_out=1, n_layers=2)
        self.gain = nn.Parameter(torch.ones(8))


def ids(params):
    return {id(p) for p in params}


def test_vector_scope_selects_the_matrices_acting_on_vector_features():
    net = Net()

    blocks, rest = block_param_groups(net.parameters(), net, "vector")

    assert blocks["block"] and "block" not in rest
    assert ids(blocks["params"]) == ids([net.mixing.mu_channel_mix.weight] + [b.mix_vectors.weight for b in net.head])
    assert ids(blocks["params"]) | ids(rest["params"]) == ids(net.parameters())


def test_all_scope_selects_the_weights_of_every_linear_layer_in_the_blocks():
    net = Net()

    blocks, rest = block_param_groups(net.parameters(), net, "all")

    layers = [m for block in (net.interaction, net.mixing, *net.head) for m in block.modules() if isinstance(m, nn.Linear)]
    assert ids(blocks["params"]) == ids(layer.weight for layer in layers)
    assert ids([net.embedding.weight, net.filter_net.weight, net.filter_net.bias, net.gain]) <= ids(rest["params"])
    assert ids(blocks["params"]) | ids(rest["params"]) == ids(net.parameters())
    assert not ids(blocks["params"]) & ids(rest["params"])


def test_slices_give_the_fused_layers_their_roles():
    net = Net()

    unsliced, unsliced_rest = block_param_groups(net.parameters(), net, "all")
    *blocks, rest = block_param_groups(net.parameters(), net, "all", slices=True)

    assert unsliced["roles"] == 1
    assert [group["roles"] for group in blocks] == [1, 2, 3]
    by_roles = {group["roles"]: ids(group["params"]) for group in blocks}
    fused3 = [net.interaction.interatomic_context_net[1].weight, net.mixing.intraatomic_context_net[1].weight]
    fused2 = [net.mixing.mu_channel_mix.weight] + [block.mix_vectors.weight for block in net.head]
    assert by_roles[3] == ids(fused3)
    assert by_roles[2] == ids(fused2)
    # the other layers stay whole, among them the last of the scalar nets, whose outputs differ in size
    whole = [net.interaction.interatomic_context_net[0].weight, net.mixing.intraatomic_context_net[0].weight]
    whole += [layer.weight for block in net.head for layer in block.scalar_net]
    assert by_roles[1] == ids(whole)
    # the roles of the interaction and mixing blocks are as wide as the features
    assert all(weight.shape[0] == 3 * 8 for weight in fused3) and net.mixing.mu_channel_mix.weight.shape[0] == 2 * 8
    # nothing else changes: the same matrices and the same rest
    assert ids(p for group in blocks for p in group["params"]) == ids(unsliced["params"])
    assert ids(rest["params"]) == ids(unsliced_rest["params"])
    BlockAdamW([*blocks, rest], block_update="muon")  # every group's matrices split into its roles


def test_slices_with_the_vector_scope_are_the_two_roles_of_the_vector_layers():
    net = Net()

    blocks, rest = block_param_groups(net.parameters(), net, "vector", slices=True)

    assert blocks["roles"] == 2
    assert ids(blocks["params"]) == ids([net.mixing.mu_channel_mix.weight] + [b.mix_vectors.weight for b in net.head])


@pytest.mark.parametrize("block_update", ["adam", "rms", "muon", "soap"])
def test_weight_decay_shrinks_only_the_block_matrices(block_update):
    net = Net()
    blocks, rest = block_param_groups(net.parameters(), net, "all")
    before = {id(p): p.detach().clone() for p in net.parameters()}
    for p in net.parameters():
        p.grad = torch.zeros_like(p)  # no update: only the decay moves the weights

    BlockAdamW([blocks, rest], lr=LR, weight_decay=0.5, block_update=block_update).step()

    for p in blocks["params"]:
        torch.testing.assert_close(p.detach(), before[id(p)] * (1 - LR * 0.5))
    for p in rest["params"]:
        assert torch.equal(p.detach(), before[id(p)])


def test_weight_decay_is_off_by_default():
    net = Net()
    optimizer = BlockAdamW(block_param_groups(net.parameters(), net, "all"))

    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.0]


def test_block_scope_needs_blocks_and_a_known_name():
    layer = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="no weight matrices"):
        block_param_groups(layer.parameters(), layer, "all")
    with pytest.raises(ValueError, match="unknown block scope"):
        block_param_groups(layer.parameters(), layer, "scalar")


def task(**kwargs):
    representation = PaiNN(
        n_atom_basis=8,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=6, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        return_vector_representation=True,
    )
    model = NeuralNetworkPotential(
        representation=representation,
        input_modules=[PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=8, n_layers=2)],
    )
    return AtomisticTask(model=model, outputs=[], optimizer_args={"lr": 1e-3}, **kwargs)


def test_task_builds_the_optimizer_from_the_block_groups():
    blocked = task(optimizer_cls=BlockAdamW, block_scope="vector")

    optimizer = blocked.configure_optimizers()

    assert isinstance(optimizer, BlockAdamW)
    assert [group["block"] for group in optimizer.param_groups] == [True, False]
    assert len(optimizer.param_groups[0]["params"]) == 2 + 2  # mu_channel_mix of 2 interactions, mix_vectors of 2 head blocks
    assert ids(p for group in optimizer.param_groups for p in group["params"]) == ids(blocked.parameters())


def test_task_without_block_scope_keeps_its_optimizer():
    plain = task(optimizer_cls=torch.optim.AdamW)

    assert plain.optimizer_cls is torch.optim.AdamW
    assert len(plain.configure_optimizers().param_groups) == 1


def test_task_steps_the_roles_of_the_fused_layers_apart_on_request():
    sliced = task(optimizer_cls=BlockAdamW, block_scope="all", block_slices=True)

    optimizer = sliced.configure_optimizers()

    assert [group["roles"] for group in optimizer.param_groups if group["block"]] == [1, 2, 3]
    assert ids(p for group in optimizer.param_groups for p in group["params"]) == ids(sliced.parameters())
    unsliced = task(optimizer_cls=BlockAdamW, block_scope="all").configure_optimizers()
    assert [group["roles"] for group in unsliced.param_groups if group["block"]] == [1]


def test_slices_need_the_blocks():
    with pytest.raises(ValueError, match="block_slices"):
        task(optimizer_cls=BlockAdamW, block_slices=True)


def soap(weights, **kwargs):
    return BlockAdamW([{"params": list(weights), "block": True}], block_update="soap", **kwargs)


def identity(n):
    return torch.eye(n, dtype=torch.float64)


def test_the_first_soap_gradient_only_starts_the_preconditioner():
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(5, 7, generator=g, dtype=torch.float64))
    before = weight.detach().clone()
    optimizer = soap([weight], lr=LR)

    weight.grad = torch.randn(5, 7, generator=g, dtype=torch.float64)
    optimizer.step()

    assert torch.equal(weight.detach(), before)
    state = optimizer.state[weight]
    for gram, basis in ((state["gram_left"], state["basis_left"]), (state["gram_right"], state["basis_right"])):
        torch.testing.assert_close(basis.T @ basis, identity(len(basis)))
        rotated = basis.T @ gram @ basis
        torch.testing.assert_close(rotated, torch.diag(rotated.diagonal()), atol=1e-12, rtol=0)  # an eigenbasis
        assert (rotated.diagonal()[:-1] - rotated.diagonal()[1:] >= -1e-12).all()  # by decreasing eigenvalue


def test_the_gram_matrices_average_with_shampoo_beta():
    g = torch.Generator().manual_seed(0)
    grads = [torch.randn(3, 5, generator=g, dtype=torch.float64) for _ in range(2)]
    for shampoo_beta, beta in ((0.5, 0.5), (None, 0.99)):  # by default betas[1]
        weight = nn.Parameter(torch.randn(3, 5, generator=g, dtype=torch.float64))
        optimizer = soap([weight], lr=LR, betas=(0.9, 0.99), shampoo_beta=shampoo_beta)
        expected_left = expected_right = 0
        for grad in grads:
            weight.grad = grad
            optimizer.step()
            expected_left = beta * expected_left + (1 - beta) * grad @ grad.T
            expected_right = beta * expected_right + (1 - beta) * grad.T @ grad

        state = optimizer.state[weight]
        torch.testing.assert_close(state["gram_left"], expected_left)
        torch.testing.assert_close(state["gram_right"], expected_right)


@pytest.mark.parametrize("precondition_frequency", [1, 2, 100])
def test_soap_is_adam_when_the_gradients_are_diagonal(precondition_frequency):
    # the eigenbasis of diagonal Gram matrices is a permutation of the coordinates: SOAP's moments are Adam's
    g = torch.Generator().manual_seed(0)
    start = torch.randn(5, 5, generator=g, dtype=torch.float64)
    weight, reference = nn.Parameter(start.clone()), nn.Parameter(start.clone())
    kwargs = dict(lr=LR, betas=(0.9, 0.99), eps=1e-8)
    optimizer = soap([weight], precondition_frequency=precondition_frequency, **kwargs)
    adam = torch.optim.AdamW([reference], weight_decay=0.0, **kwargs)

    for k in range(9):
        grad = torch.diag(torch.randn(5, generator=g, dtype=torch.float64) * 3)
        weight.grad = grad
        optimizer.step()
        if k > 0:  # the first gradient only starts the preconditioner
            reference.grad = grad.clone()
            adam.step()

    assert not torch.equal(weight.detach(), start)
    torch.testing.assert_close(weight.detach(), reference.detach())


def test_soap_commutes_with_rotating_the_weight_space():
    # W -> U W V maps gradients to U G V, the Gram matrices to U G G^T U^T and V^T G^T G V, and so SOAP's update to U update V
    g = torch.Generator().manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    start = torch.randn(6, 6, generator=g, dtype=torch.float64)
    weight, rotated = nn.Parameter(start.clone()), nn.Parameter(u @ start @ v)
    optimizers = [soap([w], lr=LR, precondition_frequency=2) for w in (weight, rotated)]
    spread = torch.diag(torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.float64))  # well separated eigenvalues

    for _ in range(9):  # four refreshes of the bases
        grad = torch.randn(6, 6, generator=g, dtype=torch.float64) @ spread
        weight.grad, rotated.grad = grad, u @ grad @ v
        for optimizer in optimizers:
            optimizer.step()

    assert (weight.detach() - start).abs().max() > 1e-3
    torch.testing.assert_close(rotated.detach(), u @ weight.detach() @ v, rtol=1e-6, atol=1e-8)


def test_refreshing_the_basis_sorts_it_and_the_second_moment_with_it():
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    eigenvalues = 2.0 ** torch.arange(6, dtype=torch.float64)
    gram = q @ torch.diag(eigenvalues) @ q.T
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    second_moment = eigenvalues[order][:, None].expand(6, 4).clone()  # each row follows the eigenvalue of its column

    new_basis, new_moment = refreshed_basis(gram, q[:, order], second_moment, dim=0)

    torch.testing.assert_close(new_basis.T @ new_basis, identity(6))
    torch.testing.assert_close(new_moment[:, 0], eigenvalues.flip(0))  # now by decreasing eigenvalue
    torch.testing.assert_close((new_basis.T @ q.flip(1)).abs(), identity(6))  # and still the eigenvectors, up to sign
    torch.testing.assert_close((eigenbasis(gram).T @ q.flip(1)).abs(), identity(6))


def test_repeated_refreshes_converge_to_the_eigenbasis():
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    gram = q @ torch.diag(2.0 ** torch.arange(6, dtype=torch.float64)) @ q.T
    basis, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    second_moment = torch.zeros(6, 6, dtype=torch.float64)

    for _ in range(60):
        basis, second_moment = refreshed_basis(gram, basis, second_moment, dim=1)

    torch.testing.assert_close((basis.T @ q.flip(1)).abs(), identity(6), atol=1e-6, rtol=0)


def test_soap_resumes_exactly_from_its_state():
    g = torch.Generator().manual_seed(0)
    start = torch.randn(4, 6, generator=g, dtype=torch.float64)
    grads = [torch.randn(4, 6, generator=g, dtype=torch.float64) for _ in range(12)]

    def run(weight, optimizer, grads):
        for grad in grads:
            weight.grad = grad.clone()
            optimizer.step()

    straight = nn.Parameter(start.clone())
    run(straight, soap([straight], lr=LR, precondition_frequency=3), grads)

    interrupted = nn.Parameter(start.clone())
    optimizer = soap([interrupted], lr=LR, precondition_frequency=3)
    run(interrupted, optimizer, grads[:5])
    saved = copy.deepcopy(optimizer.state_dict())
    resumed = nn.Parameter(interrupted.detach().clone())
    optimizer = soap([resumed], lr=LR, precondition_frequency=3)
    optimizer.load_state_dict(saved)
    run(resumed, optimizer, grads[5:])

    torch.testing.assert_close(resumed.detach(), straight.detach(), rtol=0, atol=0)


def test_soap_leaves_a_weight_alone_while_its_gradients_are_zero():
    weight = nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
    before = weight.detach().clone()
    optimizer = soap([weight], lr=LR, precondition_frequency=2)

    for _ in range(5):
        weight.grad = torch.zeros_like(weight)
        optimizer.step()

    assert torch.equal(weight.detach(), before)


@pytest.mark.parametrize("shape", [(12, 4), (4, 12)])
def test_soap_stays_finite_with_orthogonal_bases_on_gradients_of_rank_one(shape):
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(*shape, generator=g, dtype=torch.float64))
    optimizer = soap([weight], lr=LR, precondition_frequency=3)

    for _ in range(20):  # Gram matrices of rank one: most of their eigenvectors are arbitrary
        weight.grad = torch.randn(shape[0], 1, generator=g, dtype=torch.float64) @ torch.randn(
            1, shape[1], generator=g, dtype=torch.float64
        )
        optimizer.step()

    assert torch.isfinite(weight).all()
    state = optimizer.state[weight]
    for basis in (state["basis_left"], state["basis_right"]):
        torch.testing.assert_close(basis.T @ basis, identity(len(basis)))


def test_soap_steps_every_block_matrix_of_the_network():
    net = Net()
    optimizer = BlockAdamW(
        block_param_groups(net.parameters(), net, "all"), lr=LR, block_update="soap", precondition_frequency=2
    )
    before = {id(p): p.detach().clone() for p in net.parameters()}

    for _ in range(6):
        for p in net.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()

    for p in net.parameters():
        assert torch.isfinite(p).all()
        assert not torch.equal(p.detach(), before[id(p)])
    assert all("basis_left" in optimizer.state[p] for p in optimizer.param_groups[0]["params"])
    assert all("basis_left" not in optimizer.state[p] for p in optimizer.param_groups[1]["params"])


def test_soap_options_are_checked():
    weight = nn.Parameter(torch.zeros(6, 2))
    with pytest.raises(ValueError, match="precondition_frequency"):
        BlockAdamW([weight], block_update="soap", precondition_frequency=0)
    with pytest.raises(ValueError, match="shampoo_beta"):
        BlockAdamW([weight], block_update="soap", shampoo_beta=1.0)
    with pytest.raises(ValueError, match="roles"):  # the roles of a fused layer are not preconditioned apart
        BlockAdamW([{"params": [weight], "block": True, "roles": 3}], block_update="soap")
    BlockAdamW([{"params": [weight], "block": False, "roles": 3}], block_update="soap")  # not a block: plain AdamW


# The step works on lists of tensors and batches the matrices of one shape. The loop over the parameters that it
# replaced is the reference: it has to take the same steps and leave the same state.


def ref_newton_schulz(matrix, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / matrix.norm().clamp_min(1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


def ref_spectral_norm(update):
    frobenius = update.norm()
    unit = update / frobenius.clamp_min(torch.finfo(update.dtype).tiny)
    gram = unit.T @ unit if unit.shape[0] >= unit.shape[1] else unit @ unit.T
    gram = gram @ gram
    gram = gram @ gram
    return frobenius * gram.norm() ** (1 / 8)


def ref_rms_norm(update):
    return update.pow(2).mean().sqrt()


def ref_refreshed_basis(gram, basis, second_moment, dim):
    estimate = (basis * (gram @ basis)).sum(dim=0)
    order = estimate.argsort(descending=True)
    return torch.linalg.qr(gram @ basis[:, order]).Q, second_moment.index_select(dim, order)


def ref_per_role(function, matrix, roles):
    return function(matrix) if roles == 1 else torch.cat([function(role) for role in matrix.chunk(roles)])


class LoopBlockAdamW(BlockAdamW):
    """BlockAdamW as it stepped before it was batched: one parameter after the other, matrix by matrix."""

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mode = group["block_update"] if group["block"] else "adam"
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                if mode == "muon":
                    update = self._loop_muon(p, group)
                elif mode == "soap":
                    update = self._loop_soap(p, group)
                else:
                    update = self._loop_adam(p, group)
                    norm = {"rms": ref_rms_norm, "spectral": ref_spectral_norm}.get(mode)
                    if norm is not None:
                        update = ref_per_role(
                            lambda role: role / norm(role).clamp_min(torch.finfo(role.dtype).tiny), update, group["roles"]
                        )
                p.add_(update, alpha=-lr)

    def _loop_adam(self, p, group):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        beta1, beta2 = group["betas"]
        state["step"] += 1
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(p.grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
        denominator = (exp_avg_sq / (1 - beta2 ** state["step"])).sqrt_().add_(group["eps"])
        return exp_avg / (1 - beta1 ** state["step"]) / denominator

    def _loop_muon(self, p, group):
        state = self.state[p]
        if not state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buffer = state["momentum_buffer"]
        buffer.lerp_(p.grad, 1 - group["momentum"])
        return ref_per_role(
            lambda role: ref_newton_schulz(role) * (0.2 * math.sqrt(max(role.shape))),
            p.grad.lerp(buffer, group["momentum"]),
            group["roles"],
        )

    def _loop_soap(self, p, group):
        state, grad = self.state[p], p.grad
        beta1, beta2 = group["betas"]
        gram_beta = beta2 if group["shampoo_beta"] is None else group["shampoo_beta"]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["gram_left"] = p.new_zeros(p.shape[0], p.shape[0])
            state["gram_right"] = p.new_zeros(p.shape[1], p.shape[1])
            state["gram_left"].lerp_(grad @ grad.T, 1 - gram_beta)
            state["gram_right"].lerp_(grad.T @ grad, 1 - gram_beta)
            state["basis_left"] = torch.linalg.eigh(state["gram_left"]).eigenvectors.flip(1)
            state["basis_right"] = torch.linalg.eigh(state["gram_right"]).eigenvectors.flip(1)
            return torch.zeros_like(p)
        left, right = state["basis_left"], state["basis_right"]
        state["step"] += 1
        step = state["step"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.lerp_(grad, 1 - beta1)
        rotated = left.T @ grad @ right
        exp_avg_sq.mul_(beta2).addcmul_(rotated, rotated, value=1 - beta2)
        denominator = (exp_avg_sq / (1 - beta2**step)).sqrt_().add_(group["eps"])
        update = left @ ((left.T @ exp_avg @ right) / (1 - beta1**step) / denominator) @ right.T
        state["gram_left"].lerp_(grad @ grad.T, 1 - gram_beta)
        state["gram_right"].lerp_(grad.T @ grad, 1 - gram_beta)
        if step % group["precondition_frequency"] == 0:
            state["basis_left"], state["exp_avg_sq"] = ref_refreshed_basis(state["gram_left"], left, exp_avg_sq, 0)
            state["basis_right"], state["exp_avg_sq"] = ref_refreshed_basis(
                state["gram_right"], right, state["exp_avg_sq"], 1
            )
        return update


# matrices of equal shapes next to each other, fused layers with 2 and 3 roles, and a group that is no block
SHAPES = {1: [(4, 4), (4, 4), (4, 8), (5, 7), (4, 4)], 3: [(12, 4), (12, 4), (6, 5)], 2: [(8, 4), (8, 4)]}


def parameter_groups(roles_of_groups=(1, 2, 3), seed=0):
    g = torch.Generator().manual_seed(seed)

    def parameters(shapes):
        return [nn.Parameter(torch.randn(shape, generator=g, dtype=torch.float64)) for shape in shapes]

    groups = [{"params": parameters(SHAPES[roles]), "block": True, "roles": roles} for roles in roles_of_groups]
    return groups + [{"params": parameters([(6, 5), (5,), (4, 4), (4, 4)]), "weight_decay": 0.0}]


def run_both(block_update, roles_of_groups, steps, skipped=(), **kwargs):
    """The batched optimizer and the loop on the same parameters and gradients; the first parameter of the last block
    group has no gradient in the steps ``skipped``. Returns the two sets of groups and the two optimizers."""
    groups = parameter_groups(roles_of_groups)
    copies = copy.deepcopy(groups)
    batched = BlockAdamW(groups, lr=LR, block_update=block_update, **kwargs)
    loop = LoopBlockAdamW(copies, lr=LR, block_update=block_update, **kwargs)
    g = torch.Generator().manual_seed(1)
    for k in range(steps):
        for group, twin in zip(groups, copies):
            for p, q in zip(group["params"], twin["params"]):
                p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * 3
                q.grad = p.grad.clone()
        if k in skipped:
            last = len(roles_of_groups) - 1
            groups[last]["params"][0].grad = copies[last]["params"][0].grad = None
        batched.step()
        loop.step()
    return groups, copies, batched, loop


def assert_same_steps_and_state(groups, copies, batched, loop):
    for group, twin in zip(groups, copies):
        for p, q in zip(group["params"], twin["params"]):
            torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-6, atol=1e-8)
            assert batched.state[p].keys() == loop.state[q].keys()
            for key, value in batched.state[p].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, loop.state[q][key], rtol=1e-6, atol=1e-8)
                else:
                    assert value == loop.state[q][key]


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("block_update", ["adam", "rms", "spectral", "muon"])
def test_the_batched_step_is_the_step_of_the_loop(block_update, weight_decay):
    assert_same_steps_and_state(*run_both(block_update, (1, 2, 3), steps=6, weight_decay=weight_decay))


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_the_batched_soap_step_is_the_step_of_the_loop(weight_decay):
    # a refresh of the bases at every other step, and the steps of the matrices of one shape batched
    assert_same_steps_and_state(
        *run_both("soap", (1,), steps=7, weight_decay=weight_decay, precondition_frequency=2)
    )


@pytest.mark.parametrize("block_update, roles", [("adam", (1, 3)), ("rms", (1, 3)), ("muon", (1, 2)), ("soap", (1,))])
def test_a_missing_gradient_leaves_the_parameter_and_its_step_count_behind(block_update, roles):
    # the first gradient of a parameter may come late, and the steps of the others go on: batches of equal step only
    kwargs = dict(precondition_frequency=2) if block_update == "soap" else {}
    results = run_both(block_update, roles, steps=6, skipped=(0, 2, 3), **kwargs)
    assert_same_steps_and_state(*results)
    groups, _, batched, _ = results
    if block_update != "muon":  # muon keeps no step count
        late, on_time = (batched.state[p]["step"] for p in groups[len(roles) - 1]["params"][:2])
        assert late < on_time


def test_the_batched_functions_are_the_functions_of_each_matrix():
    g = torch.Generator().manual_seed(0)

    for shape in ((7, 7), (9, 5), (5, 9)):
        batch = torch.randn(4, *shape, generator=g, dtype=torch.float64) * torch.tensor([0.1, 1, 10, 100.0]).view(4, 1, 1)
        for i, matrix in enumerate(batch):
            torch.testing.assert_close(newton_schulz(batch)[i], ref_newton_schulz(matrix))
            torch.testing.assert_close(spectral_norm(batch)[i], ref_spectral_norm(matrix))
            torch.testing.assert_close(normalized(batch, "rms")[i], matrix / ref_rms_norm(matrix))
            torch.testing.assert_close(normalized(batch, "spectral")[i], matrix / ref_spectral_norm(matrix))
            torch.testing.assert_close(
                orthogonalized(batch)[i], ref_newton_schulz(matrix) * (0.2 * math.sqrt(max(shape)))
            )


@pytest.mark.parametrize("dim", [0, 1])
def test_refreshing_a_batch_of_bases_is_refreshing_each_of_them(dim):
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(3, 6, 6, generator=g, dtype=torch.float64))
    grams = q @ torch.diag_embed(2.0 ** torch.rand(3, 6, generator=g, dtype=torch.float64) * 8) @ q.mT
    bases, _ = torch.linalg.qr(torch.randn(3, 6, 6, generator=g, dtype=torch.float64))
    moments = torch.randn(3, 6, 6, generator=g, dtype=torch.float64)

    new_bases, new_moments = refreshed_basis(grams, bases, moments, dim)

    for i in range(3):
        basis, moment = ref_refreshed_basis(grams[i], bases[i], moments[i], dim)
        torch.testing.assert_close(new_bases[i], basis)
        torch.testing.assert_close(new_moments[i], moment)


def test_the_parameters_of_a_group_share_their_dtype():
    with pytest.raises(ValueError, match="dtype"):
        BlockAdamW([{"params": [nn.Parameter(torch.zeros(2, 2)), nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))]}])


@pytest.mark.parametrize("block_update", ["rms", "muon", "soap"])
def test_a_compiled_step_is_the_step_of_the_eager_one(block_update):
    # one block group and the rest: each compiles once (SOAP: once more for the steps that refresh the bases)
    kwargs = dict(precondition_frequency=3) if block_update == "soap" else {}
    groups = parameter_groups((1,))
    copies = copy.deepcopy(groups)
    compiled = BlockAdamW(groups, lr=LR, block_update=block_update, weight_decay=0.1, compiled=True, **kwargs)
    eager = BlockAdamW(copies, lr=LR, block_update=block_update, weight_decay=0.1, **kwargs)
    g = torch.Generator().manual_seed(1)

    for _ in range(7):
        for group, twin in zip(groups, copies):
            for p, q in zip(group["params"], twin["params"]):
                p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * 3
                q.grad = p.grad.clone()
        compiled.step()
        eager.step()

    for group, twin in zip(groups, copies):
        for p, q in zip(group["params"], twin["params"]):
            torch.testing.assert_close(p.detach(), q.detach(), rtol=1e-6, atol=1e-8)
            assert compiled.state[p].keys() == eager.state[q].keys()
