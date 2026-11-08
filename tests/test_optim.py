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
from gpff.optim import BlockAdamW, block_param_groups, newton_schulz, rms_norm, spectral_norm
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


@pytest.mark.parametrize("block_update", ["rms", "spectral", "muon"])
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


def test_the_matrix_functions_of_a_batch_are_those_of_each_matrix():
    g = torch.Generator().manual_seed(0)
    for shape in ((3, 5, 8), (3, 8, 5)):
        batch = torch.randn(shape, generator=g, dtype=torch.float64)
        for function in (rms_norm, spectral_norm, newton_schulz):
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


@pytest.mark.parametrize("block_update", ["adam", "rms", "spectral", "muon"])
@pytest.mark.parametrize("rows", [6, 12])  # roles wider and taller than they are long
def test_a_fused_layer_steps_like_separate_layers_for_its_roles(block_update, rows):
    fused, separate = step_fused_and_separate(block_update, rows=rows)

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
    "block_update, roles", [(update, roles) for update in ("adam", "rms", "spectral", "muon") for roles in (1, 2)]
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


@pytest.mark.parametrize("block_update", ["adam", "rms", "muon"])
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
