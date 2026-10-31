from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from gpff.callbacks import ExponentialMovingAverage
from gpff.posthoc_ema import (
    PostHocEMA, profile_inner_product, reconstruct, reconstruction_weights,
    sigma_rel_to_gamma,
)


@pytest.mark.parametrize("sigma", [0.015, 0.05, 0.1, 0.25])
def test_width_and_profile_integrals(sigma):
    gamma = sigma_rel_to_gamma(sigma)
    assert np.sqrt((gamma + 1) / ((gamma + 2)**2 * (gamma + 3))) == pytest.approx(sigma)
    # Independent numerical quadrature of the product of two densities.
    t = np.linspace(0, 70, 100001)
    other = float(sigma_rel_to_gamma(0.1))
    density_a = (gamma + 1) / 70 * (t / 70)**gamma
    density_b = (other + 1) / 100 * (t / 100)**other
    expected = np.trapezoid(density_a * density_b, t)
    assert profile_inner_product(70, gamma, 100, other) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("sigma", [0, -0.1, 12**-0.5, 1, float("nan"), float("inf")])
def test_invalid_widths(sigma):
    with pytest.raises(ValueError):
        PostHocEMA(sigma_rels=[sigma])


def test_reconstruction_recovers_source_and_is_scale_invariant():
    steps = np.repeat(np.arange(1, 33) * 100, 2)
    widths = np.tile([0.05, 0.1], 32)
    weights, error = reconstruction_weights(steps, widths, 3200, 0.1)
    np.testing.assert_allclose(weights, np.eye(1, 64, 63)[0], atol=1e-10)
    assert error < 1e-7
    a, _ = reconstruction_weights(steps, widths, 3200, 0.15)
    b, _ = reconstruction_weights(steps * 512, widths, 3200 * 512, 0.15)
    np.testing.assert_allclose(a, b, atol=1e-10)
    assert a.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="Duplicate"):
        reconstruction_weights([100, 100], [0.1, 0.1], 100, 0.15)


def test_online_and_posthoc_against_full_weight_history(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False).double()
    module = SimpleNamespace(model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(optimizers=[optimizer], global_step=0,
                              current_epoch=0, is_global_zero=True)
    cb = PostHocEMA(every_n_steps=100, dirpath=tmp_path)
    cb.on_train_start(trainer, module)
    model.weight.grad = torch.ones_like(model.weight)
    history = []
    for step in range(1, 3201):
        value = 0.001 * step + np.sin(step / 300)
        history.append(value)
        with torch.no_grad():
            model.weight.fill_(value)
        cb._after_step(model)
        trainer.global_step = step
        cb.on_train_batch_end(trainer, module, {}, None, step)
    cb.on_train_end(trainer, module)
    for sigma, shadow in zip(cb.sigma_rels, cb.shadows):
        gamma = sigma_rel_to_gamma(sigma)
        masses = np.diff((np.arange(3201) / 3200)**(gamma + 1))
        assert shadow["weight"].item() == pytest.approx(masses @ history, abs=1e-12)
    result, meta = reconstruct(tmp_path, 0.15)
    masses = np.diff((np.arange(3201) / 3200)**(sigma_rel_to_gamma(0.15) + 1))
    assert result.weight.item() == pytest.approx(masses @ history, abs=2e-3)
    assert meta["relative_profile_error"] < 0.03
    # A source width at an earlier endpoint must not consult future weights.
    result, meta = reconstruct(tmp_path, 0.1, step=1600)
    source = torch.load(tmp_path / "step_000001600.pt", weights_only=True)
    torch.testing.assert_close(result.weight, source["shadows"][1]["weight"].float())
    assert len(meta["sources"]) == 16


class TinyTask(pl.LightningModule):
    def __init__(self, skip=False):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)
        self.model.register_buffer("counter", torch.tensor(0))
        with torch.no_grad():
            self.model.weight.fill_(0.4)
            self.model.bias.fill_(0.2)
        self.skip = skip

    def training_step(self, batch, batch_idx):
        self.model.counter.add_(1)
        if self.skip and batch_idx == 1:
            return None
        return self.model(batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        return self.model(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.03, momentum=0.2)


def fit(directory, epochs, checkpoint=None, accumulate=1, skip=False, posthoc=True,
        every_n_steps=2, snapshot_epochs=()):
    cb = PostHocEMA(every_n_steps=every_n_steps, epochs=snapshot_epochs, dirpath=directory)
    callbacks = [ExponentialMovingAverage(decay=0.8)]
    if posthoc:
        callbacks.append(cb)
    trainer = pl.Trainer(accelerator="cpu", max_epochs=epochs, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0,
                         accumulate_grad_batches=accumulate, callbacks=callbacks)
    task = TinyTask(skip=skip)
    loader = DataLoader(TensorDataset(torch.arange(1., 5.).reshape(-1, 1)), batch_size=1)
    trainer.fit(task, loader, loader, ckpt_path=checkpoint, weights_only=False)
    return trainer, task, cb


def test_lightning_resume_validation_swap_and_export(tmp_path):
    full, task, ema = fit(tmp_path / "full", 3)
    first, _, _ = fit(tmp_path / "resumed", 1)
    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)
    resumed, resumed_task, resumed_ema = fit(tmp_path / "resumed", 3, checkpoint)
    assert ema.num_updates == resumed_ema.num_updates == 12
    for a, b in zip(ema.shadows, resumed_ema.shadows):
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    baseline, baseline_task, _ = fit(tmp_path / "unused", 3, posthoc=False)
    for p, q in zip(task.model.parameters(), baseline_task.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # The post-hoc source differs from the conventional validation EMA.
    assert not torch.equal(task.model.weight, ema.shadows[1]["weight"].float())
    model, _ = reconstruct(tmp_path / "full", 0.1)
    torch.testing.assert_close(model.weight, ema.shadows[1]["weight"].float())
    assert model.counter.item() == 12
    assert model.do_postprocessing and not model.training
    output = tmp_path / "inference_model"
    torch.save(model, output)
    loaded = torch.load(output, weights_only=False)
    torch.testing.assert_close(loaded(torch.ones(1, 1)), model(torch.ones(1, 1)))


@pytest.mark.parametrize("accumulate,skip,expected", [(2, False, 4), (1, True, 6), (2, True, 4)])
def test_only_actual_optimizer_updates_advance_clock(tmp_path, accumulate, skip, expected):
    _, _, ema = fit(tmp_path, 2, accumulate=accumulate, skip=skip)
    assert ema.num_updates == expected


def test_reject_missing_state_changed_widths_and_other_run(tmp_path):
    trainer, _, ema = fit(tmp_path / "source", 2)
    with pytest.raises(ValueError, match="source sigma_rels"):
        PostHocEMA(sigma_rels=[0.1, 0.15]).load_state_dict(ema.state_dict())
    with pytest.raises(ValueError, match="without its callback state"):
        PostHocEMA().on_train_start(trainer, TinyTask())
    # A new run must not write into another run's directory; checked at start.
    with pytest.raises(ValueError, match="different training run"):
        fit(tmp_path / "source", 1)


def test_resume_from_earlier_checkpoint_sets_later_snapshots_aside(tmp_path):
    directory = tmp_path / "run"
    first, _, _ = fit(directory, 1)
    checkpoint = tmp_path / "epoch1.ckpt"
    first.save_checkpoint(checkpoint)
    fit(directory, 2, checkpoint)  # continuation that is then abandoned
    # Resume from the earlier checkpoint, as hy resume does from a milestone.
    _, _, resumed = fit(directory, 3, checkpoint)
    _, _, full = fit(tmp_path / "full", 3)
    (archive,) = (directory / "abandoned").iterdir()
    assert sorted(p.name for p in archive.iterdir()) == ["step_000000006.pt", "step_000000008.pt"]
    for a, b in zip(full.shadows, resumed.shadows):
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    expected, _ = reconstruct(tmp_path / "full", 0.15)
    model, meta = reconstruct(directory, 0.15)
    torch.testing.assert_close(model.weight, expected.weight, rtol=0, atol=0)
    assert len(meta["sources"]) == 6


def test_epoch_end_snapshots(tmp_path):
    # 4 updates per epoch; cadence 3 saves 3, 6, 9, 12; epoch 2 ends at update
    # 8 and epoch 3 at update 12, which is also a cadence snapshot.
    fit(tmp_path, 3, every_n_steps=3, snapshot_epochs=[2, 3])
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == [
        f"step_{t:09d}.pt" for t in (3, 6, 8, 9, 12)]
    model, meta = reconstruct(tmp_path, 0.1, epoch=2)
    by_step, _ = reconstruct(tmp_path, 0.1, step=8)
    assert meta["step"] == 8 and meta["epoch_end"] == 2
    torch.testing.assert_close(model.weight, by_step.weight, rtol=0, atol=0)
    assert reconstruct(tmp_path, 0.1, epoch=3)[1]["step"] == 12
    with pytest.raises(ValueError, match="found 0"):
        reconstruct(tmp_path, 0.1, epoch=1)
    with pytest.raises(ValueError, match="not both"):
        reconstruct(tmp_path, 0.1, step=8, epoch=2)


def test_averages_keep_moving_late_in_training():
    # Late in a GPFF run, 1 - beta ~ (gamma + 1) / t moves a float32 average by
    # less than half a unit in the last place, so it would not change at all.
    model = torch.nn.Linear(1, 1, bias=False)
    module = SimpleNamespace(model=model)
    trainer = SimpleNamespace(optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)],
                              global_step=0, current_epoch=0, is_global_zero=True)
    with torch.no_grad():
        model.weight.fill_(1.0)
    cb = PostHocEMA()
    cb.on_train_start(trainer, module)
    start = 3_000_000
    cb.num_updates = start
    with torch.no_grad():
        model.weight.fill_(1.01)
    model.weight.grad = torch.ones_like(model.weight)
    for _ in range(1000):
        cb._after_step(model)
    theta = model.weight.item()
    for gamma, shadow in zip(cb.gammas, cb.shadows):
        remaining = (start / (start + 1000))**(float(gamma) + 1)
        assert shadow["weight"].item() == pytest.approx(theta - (theta - 1) * remaining, abs=1e-12)
        assert shadow["weight"].item() > 1 + 1e-6


def test_float16_snapshots_keep_checkpoint_precision(tmp_path):
    cb = PostHocEMA(dirpath=tmp_path, snapshot_dtype="float16")
    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)],
                              global_step=0, current_epoch=0, is_global_zero=True)
    module = SimpleNamespace(model=model)
    cb.on_train_start(trainer, module)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    cb._after_step(model)
    cb.on_train_end(trainer, module)
    saved = torch.load(tmp_path / "step_000000001.pt", weights_only=True)
    assert saved["shadows"][0]["weight"].dtype == torch.float16
    assert cb.state_dict()["shadows"][0]["weight"].dtype == torch.float64


def test_config_is_additive():
    import gpff.cli  # register the project's Hydra resolvers
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    with initialize_config_module(config_module="gpff.configs", version_base=None):
        config = compose(config_name="train", overrides=["experiment=gpff", "+callbacks=posthoc_ema"])
        with_milestones = compose(config_name="train", overrides=[
            "experiment=gpff", "+callbacks=[milestones,posthoc_ema]"])
    assert config.callbacks.ema._target_ == "gpff.callbacks.ExponentialMovingAverage"
    callback = instantiate(config.callbacks.posthoc_ema)
    assert isinstance(callback, PostHocEMA) and callback.epochs == frozenset()
    # Snapshots are also saved at the milestone epochs, for comparison there.
    callback = instantiate(with_milestones.callbacks.posthoc_ema)
    assert callback.epochs == set(with_milestones.callbacks.milestone_checkpoint.epochs)


@pytest.mark.parametrize("compiled", [False, True])
def test_painn_cli_export_loads_for_generation(tmp_path, monkeypatch, compiled):
    from schnetpack import properties
    from schnetpack.atomistic import PairwiseDistances
    from schnetpack.model import NeuralNetworkPotential
    from schnetpack.nn import CosineCutoff, GaussianRBF
    from schnetpack.utils import load_model
    from gpff.model import GatedEquivariant, PaiNN
    from gpff.cli.posthoc_ema import main

    model = NeuralNetworkPotential(
        representation=PaiNN(n_atom_basis=8, n_interactions=1,
                             radial_basis=GaussianRBF(n_rbf=4, cutoff=5.),
                             cutoff_fn=CosineCutoff(5.), return_vector_representation=True),
        input_modules=[PairwiseDistances()],
        output_modules=[GatedEquivariant(n_in=8, n_layers=2)],
    )
    if compiled:
        model.compile(backend="eager")
    inputs = {
        properties.Z: torch.tensor([1, 1]),
        properties.R: torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
        properties.n_atoms: torch.tensor([2]), properties.idx_m: torch.tensor([0, 0]),
        properties.idx_i: torch.tensor([0, 1]), properties.idx_j: torch.tensor([1, 0]),
        properties.offsets: torch.zeros(2, 3),
    }
    expected = model(dict(inputs))["forces"].detach()
    cb = PostHocEMA(dirpath=tmp_path / "sources")
    trainer = SimpleNamespace(optimizers=[torch.optim.Adam(model.parameters())],
                              global_step=0, current_epoch=0, is_global_zero=True)
    module = SimpleNamespace(model=model)
    cb.on_train_start(trainer, module)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    cb._after_step(model)
    cb.on_train_end(trainer, module)
    output = tmp_path / "phema_model"
    monkeypatch.setattr("sys.argv", ["posthoc_ema", "--input", str(cb.dirpath),
                                    "--output", str(output), "--sigma-rel", "0.1"])
    main()
    loaded = load_model(str(output), device="cpu")
    torch.testing.assert_close(loaded(dict(inputs))["forces"], expected)
    assert (tmp_path / "phema_model.json").exists()
    with pytest.raises(SystemExit):
        main()  # refuse to overwrite a previous export
