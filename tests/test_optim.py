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
    polar_express,
    refreshed_basis,
    rms_norm,
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


@pytest.mark.parametrize("block_update", ["rms", "spectral", "muon", "normuon"])
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


def spread_matrix(n=16, low=2e-3):
    """A matrix with singular values spread from low to 1 (log-spaced): after the division by its Frobenius norm
    (about 1.4) they are inside the interval [1e-3, 1] that Polar Express is made for."""
    g = torch.Generator().manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return u @ torch.diag(torch.logspace(math.log10(low), 0, n, dtype=torch.float64)) @ v.T


def test_polar_express_puts_the_singular_values_closer_to_one_than_newton_schulz():
    matrix = spread_matrix()

    def worst(orthogonalize):  # the largest distance of a singular value from one, of a square and a tall matrix
        return max((torch.linalg.svdvals(orthogonalize(m)) - 1).abs().max().item() for m in (matrix, matrix[:, :10]))

    newton, polar = worst(newton_schulz), worst(polar_express)
    assert polar < newton, (newton, polar)
    # its later iterations are Newton-Schulz's exact quintic: it converges to the polar factor
    converged = torch.linalg.svdvals(polar_express(matrix, steps=10))
    torch.testing.assert_close(converged, torch.ones_like(converged), atol=1e-6, rtol=0)
    torch.testing.assert_close(polar_express(10 * matrix), polar_express(matrix))


def test_first_muon_step_with_polar_express_is_the_scaled_polar_express_of_the_gradient():
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(12, 8, generator=g, dtype=torch.float64))
    weight.grad = torch.randn(12, 8, generator=g, dtype=torch.float64)
    before, grad = weight.detach().clone(), weight.grad.clone()
    optimizer = BlockAdamW(
        [{"params": [weight], "block": True}], lr=LR, block_update="muon", orthogonalization="polar_express"
    )

    optimizer.step()

    torch.testing.assert_close(before - weight.detach(), LR * 0.2 * math.sqrt(12) * polar_express(grad))
    with pytest.raises(ValueError, match="orthogonalization"):
        BlockAdamW([weight], block_update="muon", orthogonalization="svd")


def test_first_normuon_step_moves_every_row_alike_by_the_size_of_muons_step():
    g = torch.Generator().manual_seed(0)
    start = torch.randn(12, 8, generator=g, dtype=torch.float64)
    scales = torch.logspace(-1, 1, 12, dtype=torch.float64)[:, None]  # rows of gradients of very different size
    grad = torch.randn(12, 8, generator=g, dtype=torch.float64) * scales
    steps = {}
    for update in ("muon", "normuon"):
        weight = nn.Parameter(start.clone())
        weight.grad = grad.clone()
        optimizer = BlockAdamW([{"params": [weight], "block": True}], lr=LR, block_update=update, row_beta=0.9)
        optimizer.step()
        steps[update] = start - weight.detach()

    muon_rows, normuon_rows = (steps[u].pow(2).mean(dim=1).sqrt() for u in ("muon", "normuon"))
    assert muon_rows.max() > 1.5 * muon_rows.min()  # Muon moves some neurons much more than others
    torch.testing.assert_close(normuon_rows, normuon_rows.mean().expand(12))  # NorMuon moves them alike
    torch.testing.assert_close(steps["normuon"].norm(), steps["muon"].norm())  # by the size of Muon's step
    assert optimizer.state[weight]["row_second_moment"].shape == (12, 1)
    with pytest.raises(ValueError, match="row_beta"):
        BlockAdamW([weight], block_update="normuon", row_beta=1.0)


def test_the_matrix_functions_of_a_batch_are_those_of_each_matrix():
    g = torch.Generator().manual_seed(0)
    for shape in ((3, 5, 8), (3, 8, 5)):
        batch = torch.randn(shape, generator=g, dtype=torch.float64)
        for function in (rms_norm, spectral_norm, newton_schulz, polar_express):
            torch.testing.assert_close(function(batch), torch.stack([function(matrix) for matrix in batch]))


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


@pytest.mark.parametrize("block_update", ["adam", "rms", "spectral", "muon", "normuon", "soap"])
@pytest.mark.parametrize("rows", [6, 12])  # roles wider and taller than they are long
def test_a_fused_layer_steps_like_separate_layers_for_its_roles(block_update, rows):
    fused, separate = step_fused_and_separate(block_update, rows=rows)

    torch.testing.assert_close(fused, separate)


@pytest.mark.parametrize("block_update", ["rms", "spectral", "muon", "normuon", "soap"])
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
    [(update, roles) for update in ("adam", "rms", "spectral", "muon", "normuon", "soap") for roles in (1, 2)],
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


@pytest.mark.parametrize("block_update", ["adam", "rms", "muon", "normuon"])
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


def potential(trunk_weight_norm=False, head_weight_norm=False):
    representation = PaiNN(
        n_atom_basis=8,
        n_interactions=2,
        radial_basis=GaussianRBF(n_rbf=6, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        return_vector_representation=True,
        weight_norm=trunk_weight_norm,
    )
    return NeuralNetworkPotential(
        representation=representation,
        input_modules=[PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=8, n_layers=2, weight_norm=head_weight_norm)],
    )


def task(model=None, **kwargs):
    model = potential() if model is None else model
    return AtomisticTask(model=model, outputs=[], optimizer_args={"lr": 1e-3}, **kwargs)


def test_task_builds_the_optimizer_from_the_block_groups():
    blocked = task(optimizer_cls=BlockAdamW, block_scope="vector")

    optimizer = blocked.configure_optimizers()

    assert isinstance(optimizer, BlockAdamW)
    assert [group["block"] for group in optimizer.param_groups] == [True, False]
    assert len(optimizer.param_groups[0]["params"]) == 2 + 2  # mu_channel_mix of 2 interactions, mix_vectors of 2 head blocks
    assert ids(p for group in optimizer.param_groups for p in group["params"]) == ids(blocked.parameters())


# the trunk alone would leave the head's matrices in the block group, the head alone the trunk's
@pytest.mark.parametrize("trunk, head", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("scope", ["all", "vector"])
def test_block_scope_refuses_weight_norm(trunk, head, scope):
    model = potential(trunk_weight_norm=trunk, head_weight_norm=head)
    message = "weight_norm=true cannot be combined with task/optimizer=block_adam"

    with pytest.raises(ValueError, match=message):
        block_param_groups(model.parameters(), model, scope)
    with pytest.raises(ValueError, match=message):
        task(model=model, optimizer_cls=BlockAdamW, block_scope=scope)


def test_weight_norm_without_block_scope_keeps_its_optimizer():
    plain = task(model=potential(trunk_weight_norm=True, head_weight_norm=True), optimizer_cls=torch.optim.AdamW)

    assert len(plain.configure_optimizers().param_groups) == 1


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
    optimizer = soap([weight], lr=LR, weight_decay=0.5)

    weight.grad = torch.randn(5, 7, generator=g, dtype=torch.float64)
    optimizer.step()

    assert torch.equal(weight.detach(), before)  # no step and no decay, as in the reference implementation
    state = optimizer.state[weight]
    for gram, basis in [(state[f"gram_{side}"][0], state[f"basis_{side}"][0]) for side in ("left", "right")]:
        torch.testing.assert_close(basis.T @ basis, identity(len(basis)))
        rotated = basis.T @ gram @ basis
        torch.testing.assert_close(rotated, torch.diag(rotated.diagonal()), atol=1e-12, rtol=0)  # an eigenbasis
        assert (rotated.diagonal()[:-1] - rotated.diagonal()[1:] >= -1e-12).all()  # by decreasing eigenvalue

    weight.grad = torch.zeros_like(weight)
    optimizer.step()
    torch.testing.assert_close(weight.detach(), before * (1 - LR * 0.5))  # from the second gradient on it decays


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
        torch.testing.assert_close(state["gram_left"][0], expected_left)
        torch.testing.assert_close(state["gram_right"][0], expected_right)


@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("precondition_frequency", [1, 2, 100])
def test_soap_is_adam_when_the_gradients_are_diagonal(precondition_frequency, one_sided):
    # the eigenbasis of diagonal Gram matrices is a permutation of the coordinates: SOAP's moments are Adam's
    g = torch.Generator().manual_seed(0)
    start = torch.randn(5, 5, generator=g, dtype=torch.float64)
    weight, reference = nn.Parameter(start.clone()), nn.Parameter(start.clone())
    kwargs = dict(lr=LR, betas=(0.9, 0.99), eps=1e-8)
    optimizer = soap([weight], precondition_frequency=precondition_frequency, one_sided=one_sided, **kwargs)
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
    # W -> U W V maps gradients to U G V, the Gram matrices to U G G^T U^T and V^T G^T G V, and so SOAP's update
    # to U update V
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


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
def test_one_sided_soap_commutes_with_rotating_the_smaller_side_of_the_weight(shape):
    # the smaller side of the weight is the one with a basis: W -> W Q (or Q W) maps the gradients and the Gram matrix
    # the same way, and so SOAP's update. The other side is not rotated, and the coordinates of Adam stay.
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.float64))
    spread = torch.diag(torch.tensor([1, 2, 4, 8], dtype=torch.float64))  # well separated eigenvalues
    if shape[0] > shape[1]:
        turn, scaled = (lambda x: x @ q), (lambda x: x @ spread)
    else:
        turn, scaled = (lambda x: q @ x), (lambda x: spread @ x)
    start = torch.randn(shape, generator=g, dtype=torch.float64)
    weight, rotated = nn.Parameter(start.clone()), nn.Parameter(turn(start))
    optimizers = [soap([w], lr=LR, precondition_frequency=2, one_sided=True) for w in (weight, rotated)]

    for _ in range(9):  # four refreshes of the basis
        grad = scaled(torch.randn(shape, generator=g, dtype=torch.float64))
        weight.grad, rotated.grad = grad, turn(grad)
        for optimizer in optimizers:
            optimizer.step()

    assert (weight.detach() - start).abs().max() > 1e-3
    torch.testing.assert_close(rotated.detach(), turn(weight.detach()), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("shape, side", [((4, 9), "left"), ((9, 4), "right"), ((5, 5), "right")])
def test_one_sided_soap_keeps_the_state_of_the_smaller_side_only(shape, side):
    other = "right" if side == "left" else "left"
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(shape, generator=g, dtype=torch.float64))
    optimizer = soap([weight], lr=LR, precondition_frequency=2, one_sided=True)

    for _ in range(5):
        weight.grad = torch.randn(shape, generator=g, dtype=torch.float64)
        optimizer.step()

    state = optimizer.state[weight]
    assert state[f"basis_{side}"].shape == (1,) + (min(shape),) * 2 == state[f"gram_{side}"].shape
    assert f"basis_{other}" not in state and f"gram_{other}" not in state
    assert torch.isfinite(weight).all()


@pytest.mark.parametrize("one_sided", [False, True])
def test_soap_preconditions_the_roles_of_a_fused_layer_apart(one_sided):
    fused, separate = step_fused_and_separate("soap", steps=7, precondition_frequency=2, one_sided=one_sided)

    torch.testing.assert_close(fused, separate)


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


@pytest.mark.parametrize("dim", [0, 1])
def test_refreshing_a_batch_of_bases_is_refreshing_each_of_them(dim):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 5, 5, generator=g, dtype=torch.float64)
    gram = a @ a.mT
    basis = torch.linalg.qr(torch.randn(3, 5, 5, generator=g, dtype=torch.float64)).Q
    second_moment = torch.rand(3, 5, 5, generator=g, dtype=torch.float64)

    batched = refreshed_basis(gram, basis, second_moment, dim)

    for k in range(3):
        for got, want in zip(batched, refreshed_basis(gram[k], basis[k], second_moment[k], dim)):
            torch.testing.assert_close(got[k], want)


def test_repeated_refreshes_converge_to_the_eigenbasis():
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    gram = q @ torch.diag(2.0 ** torch.arange(6, dtype=torch.float64)) @ q.T
    basis, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    second_moment = torch.zeros(6, 6, dtype=torch.float64)

    for _ in range(60):
        basis, second_moment = refreshed_basis(gram, basis, second_moment, dim=1)

    torch.testing.assert_close((basis.T @ q.flip(1)).abs(), identity(6), atol=1e-6, rtol=0)


@pytest.mark.parametrize("one_sided", [False, True])
def test_soap_resumes_exactly_from_its_state(one_sided):
    g = torch.Generator().manual_seed(0)
    start = torch.randn(4, 6, generator=g, dtype=torch.float64)
    grads = [torch.randn(4, 6, generator=g, dtype=torch.float64) for _ in range(12)]

    def run(weight, optimizer, grads):
        for grad in grads:
            weight.grad = grad.clone()
            optimizer.step()

    options = dict(lr=LR, precondition_frequency=3, one_sided=one_sided)
    straight = nn.Parameter(start.clone())
    run(straight, soap([straight], **options), grads)

    interrupted = nn.Parameter(start.clone())
    optimizer = soap([interrupted], **options)
    run(interrupted, optimizer, grads[:5])
    saved = copy.deepcopy(optimizer.state_dict())
    resumed = nn.Parameter(interrupted.detach().clone())
    # the options come back with the state: an optimizer built with the other one_sided resumes the same run
    optimizer = soap([resumed], lr=LR, precondition_frequency=3, one_sided=not one_sided)
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


@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("shape", [(12, 4), (4, 12)])
def test_soap_stays_finite_with_orthogonal_bases_on_gradients_of_rank_one(shape, one_sided):
    g = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(*shape, generator=g, dtype=torch.float64))
    optimizer = soap([weight], lr=LR, precondition_frequency=3, one_sided=one_sided)

    for _ in range(20):  # Gram matrices of rank one: most of their eigenvectors are arbitrary
        weight.grad = torch.randn(shape[0], 1, generator=g, dtype=torch.float64) @ torch.randn(
            1, shape[1], generator=g, dtype=torch.float64
        )
        optimizer.step()

    assert torch.isfinite(weight).all()
    state = optimizer.state[weight]
    bases = [state[key][0] for key in ("basis_left", "basis_right") if key in state]
    assert len(bases) == (1 if one_sided else 2)
    for basis in bases:
        torch.testing.assert_close(basis.T @ basis, identity(len(basis)))


@pytest.mark.parametrize("slices", [False, True])
def test_soap_steps_every_block_matrix_of_the_network(slices):
    net = Net()
    groups = block_param_groups(net.parameters(), net, "all", slices=slices)
    optimizer = BlockAdamW(groups, lr=LR, block_update="soap", precondition_frequency=2)
    before = {id(p): p.detach().clone() for p in net.parameters()}

    for _ in range(6):
        for p in net.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()

    for p in net.parameters():
        assert torch.isfinite(p).all()
        assert not torch.equal(p.detach(), before[id(p)])
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert ("basis_left" in optimizer.state[p]) == group["block"]
            if group["block"]:
                assert optimizer.state[p]["basis_left"].shape[0] == group["roles"]


def test_soap_options_are_checked():
    weight = nn.Parameter(torch.zeros(6, 2))
    with pytest.raises(ValueError, match="precondition_frequency"):
        BlockAdamW([weight], block_update="soap", precondition_frequency=0)
    with pytest.raises(ValueError, match="shampoo_beta"):
        BlockAdamW([weight], block_update="soap", shampoo_beta=1.0)


@pytest.mark.parametrize("block_update", ["adam", "rms", "spectral", "muon", "normuon", "soap"])
@pytest.mark.parametrize("roles", [1, 2])
def test_matrices_stepped_together_step_as_if_alone(block_update, roles):
    # a group batches its matrices of one shape; a gradient missing at some steps puts its matrix behind the others
    g = torch.Generator().manual_seed(0)
    shapes = [(8, 4), (8, 4), (4, 6), (8, 4), (4, 6)]
    starts = [torch.randn(shape, generator=g, dtype=torch.float64) for shape in shapes]
    together, alone = [nn.Parameter(s.clone()) for s in starts], [nn.Parameter(s.clone()) for s in starts]
    kwargs = dict(lr=LR, block_update=block_update, weight_decay=0.1, precondition_frequency=2)
    optimizers = [BlockAdamW([{"params": together, "block": True, "roles": roles}], **kwargs)]
    optimizers += [BlockAdamW([{"params": [p], "block": True, "roles": roles}], **kwargs) for p in alone]

    for step in range(7):
        for k, (a, b) in enumerate(zip(together, alone)):
            missing = k == 1 and step in (0, 3)
            a.grad = None if missing else torch.randn(a.shape, generator=g, dtype=torch.float64)
            b.grad = None if missing else a.grad.clone()
        for optimizer in optimizers:
            optimizer.step()

    for a, b, start in zip(together, alone, starts):
        assert not torch.equal(a.detach(), start)
        torch.testing.assert_close(a.detach(), b.detach())


def test_the_matrices_of_a_block_group_share_their_dtype():
    with pytest.raises(ValueError, match="dtype"):
        mixed = [nn.Parameter(torch.zeros(2, 2)), nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))]
        BlockAdamW([{"params": mixed, "block": True}])
