import copy
import io

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from schnetpack import properties
from schnetpack.atomistic import PairwiseDistances
from schnetpack.model import NeuralNetworkPotential
from schnetpack.nn import Dense
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.radial import GaussianRBF
from schnetpack.representation.painn import PaiNNMixing

from gpff.model import GatedEquivariant, PaiNN
from gpff.model.painn import ProductNormalizedMixing
from gpff.nn.normalized import NormalizedDense, project_weights, soft_unit_vectors
from gpff.task import AtomisticTask, ModelOutput
from test_model_growth import _batch, _molecules


def make_model(**overrides):
    options = dict(prenorm="rms", degree_norm=True, weight_norm=True,
                   residual_scaling=True, normalized_products=True, radial_norm=True)
    options.update(overrides)
    rep = PaiNN(n_atom_basis=32, n_interactions=4,
                radial_basis=GaussianRBF(20, 10.0), cutoff_fn=CosineCutoff(10.0),
                return_vector_representation=True, **options)
    head = GatedEquivariant(n_in=32, n_layers=3, weight_norm=options["weight_norm"],
                           normalized_products=options["normalized_products"],
                           output_gain=1.0, zero_net_force=True)
    return NeuralNetworkPotential(rep, [PairwiseDistances()], [head]).double()


@pytest.mark.parametrize("bias", [False, True])
def test_normalized_affine_gauge_gradients_and_projection(bias):
    torch.manual_seed(8)
    dense = Dense(13, 7, bias=bias, activation=None).double()
    if bias:
        with torch.no_grad():
            dense.bias.normal_()
    layer = NormalizedDense(dense)
    x = torch.randn(9, 13, dtype=torch.float64)
    expected = layer(x)
    # Scaling a complete affine row must leave the function unchanged.
    with torch.no_grad():
        layer.weight.mul_(19)
        if bias:
            layer.bias.mul_(19)
    torch.testing.assert_close(layer(x), expected)
    layer(x).square().sum().backward()
    radial_grad = (layer.weight.grad * layer.weight).sum(1)
    if bias:
        radial_grad = radial_grad + layer.bias.grad * layer.bias
    torch.testing.assert_close(radial_grad, torch.zeros_like(radial_grad), atol=1e-12, rtol=0)
    layer.project_()
    torch.testing.assert_close(layer(x), expected)
    raw = layer._augmented_weight()
    torch.testing.assert_close(raw.square().mean(1), torch.ones(7, dtype=torch.float64))
    w, b = layer.normalized_parameters()
    norm2 = w.square().sum(1) + (0 if b is None else b.square())
    torch.testing.assert_close(norm2, torch.ones_like(norm2))


def test_direction_coordinates_are_equivariant_and_nonsingular_at_zero():
    v = torch.zeros(5, 3, 11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(soft_unit_vectors, (v,))
    torch.testing.assert_close(torch.autograd.grad(soft_unit_vectors(v).sum(), v)[0], torch.ones_like(v))
    v = torch.randn_like(v) * 100
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=v.dtype))
    torch.testing.assert_close(soft_unit_vectors(rotation @ v), rotation @ soft_unit_vectors(v))


def test_zero_initialized_rows_require_an_explicit_output_gain():
    layer = Dense(4, 4, activation=None)
    with torch.no_grad():
        layer.weight.zero_()
    with pytest.raises(ValueError, match="nonzero affine rows"):
        NormalizedDense(layer)


def test_products_remove_cubic_vector_feedback():
    # An invariant gate proportional to |V| isolates the original cubic path.
    class MagnitudeGate(torch.nn.Module):
        def forward(self, ctx):
            _, magnitude = ctx.chunk(2, dim=-1)
            return torch.cat([torch.zeros_like(magnitude), magnitude, magnitude], dim=-1)

    original = PaiNNMixing(4, activation=torch.nn.functional.silu).double()
    native = ProductNormalizedMixing(4, activation=torch.nn.functional.silu).double()
    with torch.no_grad():
        original.mu_channel_mix.weight.copy_(torch.eye(4, dtype=torch.float64).repeat(2, 1))
    native.load_state_dict(original.state_dict())
    original.intraatomic_context_net = native.intraatomic_context_net = MagnitudeGate()
    q = torch.zeros(2, 1, 4, dtype=torch.float64)
    mu = torch.ones(2, 3, 4, dtype=torch.float64)
    old_growth = original(q, mu * 100)[0].norm() / original(q, mu)[0].norm()
    new_growth = native(q, mu * 100)[0].norm() / native(q, mu)[0].norm()
    assert old_growth > 9e5
    assert new_growth < 140  # asymptotically linear, not cubic


@pytest.mark.parametrize("overrides", [{}, {"prenorm": False}, {"degree_norm": "mean"},
                                      {"normalized_products": False}, {"weight_norm": False},
                                      {"shared_interactions": True, "shared_filters": True}])
def test_native_model_symmetries_and_batch_independence(overrides):
    torch.manual_seed(3)
    model = make_model(**overrides).eval()
    positions, numbers = _molecules()
    inputs = _batch(positions, numbers)
    forces = model(dict(inputs))["forces"]
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rotation[:, 0] *= -torch.linalg.det(rotation)  # include reflections
    moved = model(_batch([p @ rotation.T + 2.3 for p in positions], numbers))["forces"]
    torch.testing.assert_close(moved, forces @ rotation.T)
    perm = torch.randperm(len(numbers[0]))
    permuted = model(_batch([positions[0][perm], positions[1]], [numbers[0][perm], numbers[1]]))["forces"]
    torch.testing.assert_close(permuted[:len(perm)], forces[:len(perm)][perm])
    alone = model(_batch(positions[:1], numbers[:1]))["forces"]
    torch.testing.assert_close(alone, forces[:len(perm)])
    for part in forces.split([len(z) for z in numbers]):
        torch.testing.assert_close(part.sum(0), torch.zeros(3, dtype=forces.dtype), atol=1e-10, rtol=0)
    forces.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_radial_normalization_and_fused_affine_match_direct_calculation():
    model = make_model().representation
    inputs = PairwiseDistances()(_batch(*_molecules()))
    r = inputs[properties.Rij]
    d = (r.square().sum(-1, keepdim=True) + model.epsilon).sqrt()
    phi = model.radial_basis(d)
    phi = phi * (phi.square().mean(-1, keepdim=True) + 1e-8).rsqrt()
    cutoff = model.cutoff_fn(d)[..., None]
    with torch.no_grad():
        model.filter_net.bias.normal_()
    w, b = model.filter_net.normalized_parameters()
    direct = model.filter_net(phi) * cutoff
    fused = torch.nn.functional.linear(torch.cat([phi * cutoff, cutoff], -1), torch.cat([w, b[:, None]], 1))
    torch.testing.assert_close(fused, direct)
    grads = torch.autograd.grad(fused.square().sum(), tuple(model.filter_net.parameters()), retain_graph=True)
    ref = torch.autograd.grad(direct.square().sum(), tuple(model.filter_net.parameters()))
    for a, b in zip(grads, ref):
        torch.testing.assert_close(a, b)


def test_export_roundtrip_and_compiled_repeated_backward():
    model = make_model()
    inputs = _batch(*_molecules())
    data = io.BytesIO()
    torch.save(model, data)
    data.seek(0)
    restored = torch.load(data, weights_only=False)
    torch.testing.assert_close(model(dict(inputs))["forces"], restored(dict(inputs))["forces"])
    compiled = torch.compile(restored, backend="eager", dynamic=True)
    # Several calls before a single backward catches unsafe in-forward projection.
    loss = sum(compiled(dict(inputs))["forces"].square().sum() for _ in range(2))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in restored.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_training_matches_eager():
    torch.manual_seed(12)
    eager = make_model().float().cuda()
    compiled_model = copy.deepcopy(eager)
    compiled = torch.compile(compiled_model, dynamic=True)
    inputs = {k: (v.float() if v.is_floating_point() else v).cuda()
              for k, v in _batch(*_molecules()).items()}
    opt1 = torch.optim.AdamW(eager.parameters(), lr=0.003, weight_decay=0)
    opt2 = torch.optim.AdamW(compiled_model.parameters(), lr=0.003, weight_decay=0)
    for _ in range(2):
        opt1.zero_grad()
        opt2.zero_grad()
        a = eager(dict(inputs))["forces"]
        b = compiled(dict(inputs))["forces"]
        torch.testing.assert_close(a, b, rtol=1e-3, atol=2e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        for p, q in zip(eager.parameters(), compiled_model.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=2e-6)
        opt1.step()
        opt2.step()
        project_weights(eager)
        project_weights(compiled_model)


def test_coincident_and_isolated_atoms_have_finite_forces_and_derivatives():
    model = make_model()
    numbers = [torch.tensor([6, 1, 1]), torch.tensor([1])]
    inputs = _batch([torch.zeros(len(z), 3, dtype=torch.float64) for z in numbers], numbers)
    inputs[properties.R].requires_grad_()
    force = model(inputs)["forces"]
    torch.testing.assert_close(force, torch.zeros_like(force))
    force.sum().backward()
    assert torch.isfinite(inputs[properties.R].grad).all()


def test_optimizer_step_projects_weights_and_checkpoint_restores():
    model = make_model()
    task = AtomisticTask(model, [ModelOutput("forces", metrics={})], optimizer_args={"lr": 0.01})
    optimizer = task.configure_optimizers()
    inputs = _batch(*_molecules())

    def closure():
        optimizer.zero_grad()
        loss = model(dict(inputs))["forces"].square().sum()
        loss.backward()
        return loss

    for _ in range(3):
        task.optimizer_step(optimizer=optimizer, optimizer_closure=closure)
    for layer in model.modules():
        if isinstance(layer, NormalizedDense):
            torch.testing.assert_close(layer._augmented_weight().square().mean(1),
                                       torch.ones(layer.out_features, dtype=torch.float64))
    state = copy.deepcopy(model.state_dict())
    copy_model = make_model()
    copy_model.load_state_dict(state)
    torch.testing.assert_close(copy_model(dict(inputs))["forces"], model(dict(inputs))["forces"])


def test_stable_config_instantiates_without_training_remedies():
    import gpff.cli.train  # schnetpack's Hydra resolvers
    with initialize_config_module(config_module="gpff.configs", version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=gpff_stable"])
    assert cfg.task.scheduler_cls is None and cfg.task.warmup_steps == 0
    assert cfg.trainer.gradient_clip_val is None and not cfg.task.skip_exploding_batches
    model = instantiate(cfg.model)
    assert isinstance(model.representation.mixing[0], ProductNormalizedMixing)
