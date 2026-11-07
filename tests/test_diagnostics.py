import csv
import math
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from schnetpack import properties

import gpff.cli  # registers the resolvers the configs use
from gpff import keys
from gpff.analysis import field
from gpff.diagnostics import DenoisingErrorCurve, EveryNSamples, SigmaCalibration
from gpff.sample.base import Calculator

CONFIG_DIR = Path(gpff.cli.__file__).parents[1] / "configs"


# --- helpers -----------------------------------------------------------------


def make_batch(sizes=(3, 4, 5), sigmas=(0.05, 0.5, 5.0), dtype=torch.float32, seed=0):
    """Validation molecules: clean x_0, noisy R = x_0 + sigma * noise, and the
    per-atom noise level, like the training transform writes them."""
    generator = torch.Generator().manual_seed(seed)
    n_atoms = torch.tensor(sizes)
    idx_m = torch.repeat_interleave(torch.arange(len(sizes)), n_atoms)
    n = int(n_atoms.sum())
    x_0 = field.center_per_molecule(torch.randn(n, 3, generator=generator, dtype=dtype), idx_m, n_atoms)
    noise = field.mean_free_noise(x_0, idx_m, n_atoms, generator)
    sigma = torch.tensor(sigmas, dtype=dtype)[idx_m].unsqueeze(-1)
    return {
        properties.Z: torch.randint(0, 2, (n,), generator=generator) * 5 + 1,
        properties.idx_m: idx_m,
        properties.n_atoms: n_atoms,
        keys.x_0: x_0,
        properties.R: x_0 + sigma * noise,
        keys.sigma_curr: sigma,
    }


class FakeModule:
    """The parts of the task the diagnostics use."""

    def __init__(self, forces, outputs=None, include_time=False):
        self.forces = forces
        self.outputs = [SimpleNamespace(name=keys.forces)] if outputs is None else outputs
        self.model = SimpleNamespace(output_modules=[SimpleNamespace(include_time=include_time)])
        self.logged = []
        self.forward_calls = 0

    def predict_without_postprocessing(self, inputs):
        self.forward_calls += 1
        return {keys.forces: self.forces(inputs)}

    def log_dict(self, values, **kwargs):
        self.logged.append(dict(values))


def trainer(world_size=1, sanity_checking=False):
    return SimpleNamespace(world_size=world_size, sanity_checking=sanity_checking)


def train(callback, trainer_, n_batches, size=4):
    for batch_idx in range(n_batches):
        callback.on_train_batch_end(
            trainer_, None, None, {properties.n_atoms: torch.zeros(size, dtype=torch.long)}, batch_idx
        )


def validate(callback, trainer_, module=None, batches=(None, None)):
    callback.on_validation_start(trainer_, module)
    for batch_idx, batch in enumerate(batches):
        callback.on_validation_batch_end(trainer_, module, None, batch, batch_idx)
    callback.on_validation_epoch_end(trainer_, module)
    callback.on_validation_end(trainer_, module)


def run_due(callback, module, batches, trainer_=None):
    """One epoch that passes exactly one multiple of ``every_n_samples``."""
    trainer_ = trainer_ or trainer()
    train(callback, trainer_, 1, size=callback.every_n_samples)
    validate(callback, trainer_, module, batches)


def perfect(inputs):
    return 2 * (inputs[keys.x_0] - inputs[properties.R])


def no_forces(inputs):
    return torch.zeros_like(inputs[properties.R])


class Recorder(EveryNSamples):
    def __init__(self, every_n_samples):
        super().__init__(every_n_samples)
        self.calls = []

    def on_due_validation_start(self, trainer, pl_module):
        self.calls.append("start")

    def on_due_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.calls.append(("batch", batch_idx))

    def on_due_validation_epoch_end(self, trainer, pl_module):
        self.calls.append("end")


# --- EveryNSamples -------------------------------------------------------------


def test_runs_only_in_the_validation_after_the_interval_is_passed():
    callback, trainer_ = Recorder(30), trainer()
    runs = []
    for _ in range(5):
        train(callback, trainer_, 3)  # 12 samples per epoch: 12, 24, 36, 48, 60
        validate(callback, trainer_)
        runs.append(callback.calls.count("start"))

    assert runs == [0, 0, 1, 1, 2]
    assert callback.calls[:4] == ["start", ("batch", 0), ("batch", 1), "end"]


def test_several_intervals_between_two_validations_run_once():
    callback, trainer_ = Recorder(5), trainer()
    train(callback, trainer_, 3)  # 12 samples pass two multiples of 5
    validate(callback, trainer_)
    assert callback.calls.count("start") == 1
    assert not callback.due

    validate(callback, trainer_)
    assert callback.calls.count("start") == 1


def test_sanity_check_neither_runs_nor_consumes_the_interval():
    callback = Recorder(5)
    train(callback, trainer(), 3)

    validate(callback, trainer(sanity_checking=True))
    assert callback.calls == []
    assert callback.due

    validate(callback, trainer())
    assert callback.calls.count("start") == 1


def test_samples_are_counted_over_all_devices():
    callback = Recorder(1000)
    train(callback, trainer(world_size=4), 2, size=4)
    assert callback.samples_seen == 32


def test_hooks_do_nothing_when_nothing_is_due():
    callback = Recorder(1000)
    train(callback, trainer(), 3)
    validate(callback, trainer(), batches=(None,) * 5)
    assert callback.calls == []


def test_a_resumed_run_continues_the_count_and_the_unserved_interval():
    callback, trainer_ = Recorder(10), trainer()
    train(callback, trainer_, 3)  # 12 samples, a checkpoint before the validation
    resumed = Recorder(10)
    resumed.load_state_dict(callback.state_dict())
    assert resumed.samples_seen == 12 and resumed.due

    # a checkpoint saved in the due validation, after its epoch end hook
    # (ModelCheckpoint runs in on_validation_end): the interval is served
    resumed.on_validation_start(trainer_, None)
    resumed.on_validation_epoch_end(trainer_, None)
    again = Recorder(10)
    again.load_state_dict(resumed.state_dict())
    assert again.samples_seen == 12 and not again.due


def test_interval_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        Recorder(0)


# --- field ---------------------------------------------------------------------


def test_center_per_molecule_removes_the_mean_of_each_molecule():
    batch = make_batch(dtype=torch.float64)
    x = torch.randn(12, 3, dtype=torch.float64)
    centered = field.center_per_molecule(x, batch[properties.idx_m], batch[properties.n_atoms])
    for m in range(3):
        torch.testing.assert_close(
            centered[batch[properties.idx_m] == m].mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0
        )


def test_std_per_molecule_is_the_unbiased_std_of_the_flattened_values():
    batch = make_batch(dtype=torch.float64)
    idx_m, n_atoms = batch[properties.idx_m], batch[properties.n_atoms]
    values = torch.randn(12, 3, dtype=torch.float64)

    expected = torch.stack([values[idx_m == m].flatten().std() for m in range(3)])
    torch.testing.assert_close(field.std_per_molecule(values, idx_m, n_atoms), expected)


class _ForceModel(torch.nn.Module):
    def forward(self, inputs):
        return {keys.forces: torch.zeros_like(inputs[properties.R])}


def test_std_per_molecule_is_the_estimate_of_the_samplers():
    batch = make_batch(dtype=torch.float64)
    idx_m, n_atoms = batch[properties.idx_m], batch[properties.n_atoms]
    values = torch.randn(12, 3, dtype=torch.float64)

    calculator = Calculator(model=_ForceModel(), device="cpu")
    sampler = calculator._compute_std_per_molecule(idx_m, values)
    torch.testing.assert_close(field.std_per_molecule(values, idx_m, n_atoms)[idx_m].unsqueeze(-1), sampler)


def test_noisy_positions_without_alignment_are_x_0_plus_sigma_noise():
    batch = make_batch(dtype=torch.float64)
    x_0, n_atoms = batch[keys.x_0], batch[properties.n_atoms]
    noise = field.mean_free_noise(x_0, batch[properties.idx_m], n_atoms, torch.Generator().manual_seed(0))

    positions = field.noisy_positions(x_0, batch[properties.Z], n_atoms, [0.1, 3.0], noise)

    assert positions.shape == (2, 12, 3)
    torch.testing.assert_close(positions[0], x_0 + 0.1 * noise)
    torch.testing.assert_close(positions[1], x_0 + 3.0 * noise)


def test_alignment_permutes_the_atoms_and_never_increases_the_displacement():
    batch = make_batch(sizes=(6, 8), dtype=torch.float64)
    x_0, idx_m, n_atoms = batch[keys.x_0], batch[properties.idx_m], batch[properties.n_atoms]
    noise = field.mean_free_noise(x_0, idx_m, n_atoms, torch.Generator().manual_seed(0))
    args = (x_0, batch[properties.Z], n_atoms, [1.0, 10.0], noise)

    plain = field.noisy_positions(*args)
    permuted = field.noisy_positions(*args, align_permutation=True)
    rotated = field.noisy_positions(*args, align_rotation=True)

    assert not torch.equal(plain[1], permuted[1])
    for k in range(2):
        for m in range(2):
            mask = idx_m == m
            before, target = plain[k][mask], x_0[mask]
            after = permuted[k][mask]
            # the same atoms in another order
            torch.testing.assert_close(after.sum(0), before.sum(0))
            torch.testing.assert_close(after.square().sum(), before.square().sum())
            assert (after - target).square().sum() <= (before - target).square().sum() + 1e-9
            assert (rotated[k][mask] - target).square().sum() <= (before - target).square().sum() + 1e-9


def test_denoising_errors_of_known_models():
    batch = make_batch(dtype=torch.float64)
    x_0, idx_m, n_atoms = batch[keys.x_0], batch[properties.idx_m], batch[properties.n_atoms]
    noise = field.mean_free_noise(x_0, idx_m, n_atoms, torch.Generator().manual_seed(0))
    sigmas = [0.1, 1.0, 10.0]
    positions = field.noisy_positions(x_0, batch[properties.Z], n_atoms, sigmas, noise)

    def errors(forces):
        return field.denoising_errors(lambda inputs: {keys.forces: forces(inputs)}, batch, x_0, sigmas, positions)

    # D(x) = x removes nothing: e = |noise|^2 / (3N - 3), whatever sigma
    kept = torch.stack([noise[idx_m == m].square().sum() for m in range(3)]) / (3 * n_atoms - 3)
    identity = errors(no_forces)
    assert identity.shape == (3, 3)
    for k in range(3):
        torch.testing.assert_close(identity[:, k], kept)

    # D(x) = x_0 is perfect; D(x) = x_0 + (x - x_0) / 2 keeps half of the noise
    torch.testing.assert_close(errors(perfect), torch.zeros(3, 3, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(errors(lambda inputs: x_0 - inputs[properties.R]), 0.25 * identity)


# --- DenoisingErrorCurve -----------------------------------------------------------


def curve(**kwargs):
    kwargs = {"sigmas": [0.1, 1.0, 10.0], "every_n_samples": 3, "align_permutation": False, **kwargs}
    return DenoisingErrorCurve(**kwargs)


def test_curve_logs_the_error_per_sigma_and_its_integral():
    module = FakeModule(no_forces)
    run_due(curve(), module, [make_batch()])

    (logged,) = module.logged
    assert set(logged) == {"val/e_sigma_0.1", "val/e_sigma_1", "val/e_sigma_10", "val/e_log_sigma_integral"}
    # nothing is removed, so the error does not depend on sigma
    # (up to the rounding of float32 positions, which is relatively largest at small sigma)
    error = logged["val/e_sigma_1"]
    assert logged["val/e_sigma_0.1"] == pytest.approx(error, rel=1e-3)
    assert logged["val/e_sigma_10"] == pytest.approx(error, rel=1e-3)
    assert logged["val/e_log_sigma_integral"] == pytest.approx(error * math.log(100), rel=1e-3)


def test_curve_follows_the_model():
    errors = {}
    for name, forces in {
        "none": no_forces,
        "half": lambda inputs: inputs[keys.x_0] - inputs[properties.R],
        "perfect": perfect,
    }.items():
        module = FakeModule(forces)
        run_due(curve(), module, [make_batch()])
        errors[name] = module.logged[0]["val/e_sigma_1"]

    assert errors["half"] == pytest.approx(0.25 * errors["none"], rel=1e-4)
    assert errors["perfect"] == pytest.approx(0.0, abs=1e-8)


def test_curve_averages_the_molecules_of_all_batches(monkeypatch):
    # a batch of 2 molecules with errors (1, 2, 3) and one of 1 molecule with (4, 5, 6)
    def errors(model, batch, x_0, sigmas, positions):
        n_molecules = batch[properties.n_atoms].shape[0]
        return torch.tensor([[1.0, 2.0, 3.0] if n_molecules == 2 else [4.0, 5.0, 6.0]]).repeat(n_molecules, 1)

    monkeypatch.setattr(field, "denoising_errors", errors)
    module = FakeModule(no_forces)
    run_due(curve(), module, [make_batch(sizes=(3, 4)), make_batch(sizes=(5,))])

    (logged,) = module.logged
    # (2 * (1, 2, 3) + (4, 5, 6)) / 3 = (2, 3, 4): weighted by molecules, not by batch
    assert [logged[f"val/e_sigma_{sigma}"] for sigma in ("0.1", "1", "10")] == pytest.approx([2.0, 3.0, 4.0])
    # trapezoid over log(sigma): ((2 + 3) / 2 + (3 + 4) / 2) * log(10)
    assert logged["val/e_log_sigma_integral"] == pytest.approx(6 * math.log(10))


def test_curve_leaves_the_global_rng_alone():
    module = FakeModule(no_forces)
    state = torch.get_rng_state()
    run_due(curve(align_permutation=True), module, [make_batch()])
    assert torch.equal(torch.get_rng_state(), state)


def test_curve_does_not_touch_the_model_when_nothing_is_due():
    module = FakeModule(no_forces)
    callback, trainer_ = curve(every_n_samples=1000), trainer()
    train(callback, trainer_, 2)
    validate(callback, trainer_, module, [make_batch()])
    assert module.forward_calls == 0 and module.logged == []


def test_curve_aligns_the_noisy_molecules_once(monkeypatch):
    calls = []
    original = field.noisy_positions
    monkeypatch.setattr(field, "noisy_positions", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))

    module = FakeModule(no_forces)
    callback = curve(align_permutation=True)
    run_due(callback, module, [make_batch()])
    run_due(callback, module, [make_batch()])
    assert len(calls) == 1
    assert module.logged[0] == module.logged[1]

    # another batch under the same index is not the cached one
    run_due(callback, module, [make_batch(sizes=(2, 3))])
    assert len(calls) == 2

    calls.clear()
    uncached = curve(align_permutation=True, cache_alignment=False)
    run_due(uncached, module, [make_batch()])
    run_due(uncached, module, [make_batch()])
    assert len(calls) == 2
    assert module.logged[-1] == module.logged[-2]


@pytest.mark.parametrize("sigmas", [[1.0], [0.0, 1.0], []])
def test_curve_needs_two_positive_noise_levels(sigmas):
    with pytest.raises(ValueError, match="two positive"):
        DenoisingErrorCurve(sigmas=sigmas)


@pytest.mark.parametrize("callback_cls", [DenoisingErrorCurve, SigmaCalibration])
def test_diagnostics_need_a_time_agnostic_force_model(callback_cls):
    kwargs = {"sigmas": [0.1, 1.0]} if callback_cls is DenoisingErrorCurve else {}
    callback_cls(**kwargs).on_fit_start(None, FakeModule(no_forces))

    with pytest.raises(ValueError, match="force model"):
        callback_cls(**kwargs).on_fit_start(
            None, FakeModule(no_forces, outputs=[SimpleNamespace(name=keys.eps_pred)])
        )
    with pytest.raises(ValueError, match="time-agnostic"):
        callback_cls(**kwargs).on_fit_start(None, FakeModule(no_forces, include_time=True))


# --- SigmaCalibration ------------------------------------------------------------


def test_calibration_of_a_perfect_model_is_one():
    module = FakeModule(perfect)
    run_due(SigmaCalibration(every_n_samples=3), module, [make_batch()])

    (logged,) = module.logged
    assert logged.pop("val/sigma_hat_log_ratio_mean") == pytest.approx(0.0, abs=1e-5)
    assert logged.pop("val/sigma_hat_log_ratio_rms") == pytest.approx(0.0, abs=1e-5)
    # the molecules of the batch are in three bins
    assert set(logged) == {
        "val/sigma_hat_ratio_0.03_0.1",
        "val/sigma_hat_ratio_0.3_1",
        "val/sigma_hat_ratio_3_10",
    }
    assert list(logged.values()) == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)


def test_calibration_measures_a_model_that_overshoots():
    module = FakeModule(lambda inputs: 2 * perfect(inputs))
    run_due(SigmaCalibration(every_n_samples=3), module, [make_batch()])

    (logged,) = module.logged
    assert logged["val/sigma_hat_ratio_0.3_1"] == pytest.approx(2.0, rel=1e-5)
    assert logged["val/sigma_hat_log_ratio_mean"] == pytest.approx(math.log(2), rel=1e-5)
    assert logged["val/sigma_hat_log_ratio_rms"] == pytest.approx(math.log(2), rel=1e-5)


def test_calibration_attributes_molecules_to_their_noise_level_bin():
    # right below sigma = 1, the model is exact; above it the estimate is 4 times too large
    module = FakeModule(lambda inputs: perfect(inputs) * torch.where(inputs[keys.sigma_curr] > 1.0, 4.0, 1.0))
    run_due(SigmaCalibration(every_n_samples=3), module, [make_batch()])

    (logged,) = module.logged
    assert logged["val/sigma_hat_ratio_0.03_0.1"] == pytest.approx(1.0, rel=1e-5)
    assert logged["val/sigma_hat_ratio_0.3_1"] == pytest.approx(1.0, rel=1e-5)
    assert logged["val/sigma_hat_ratio_3_10"] == pytest.approx(4.0, rel=1e-5)
    assert logged["val/sigma_hat_log_ratio_mean"] == pytest.approx(math.log(4) / 3, rel=1e-5)
    assert logged["val/sigma_hat_log_ratio_rms"] == pytest.approx(math.log(4) / math.sqrt(3), rel=1e-5)


def test_calibration_puts_noise_levels_outside_the_edges_into_the_end_bins():
    module = FakeModule(perfect)
    run_due(SigmaCalibration(every_n_samples=3), module, [make_batch(sigmas=(0.001, 0.5, 100.0))])

    ratios = {name for name in module.logged[0] if name.startswith("val/sigma_hat_ratio_")}
    assert ratios == {"val/sigma_hat_ratio_0.01_0.03", "val/sigma_hat_ratio_0.3_1", "val/sigma_hat_ratio_10_30"}


def test_calibration_pools_the_batches_and_supports_a_single_bin():
    module = FakeModule(perfect)
    batches = [make_batch(sizes=(3, 4), sigmas=(0.05, 0.5), seed=1), make_batch(sizes=(5,), sigmas=(5.0,), seed=2)]
    run_due(SigmaCalibration(every_n_samples=3, sigma_edges=[0.01, 30.0]), module, batches)

    (logged,) = module.logged
    assert set(logged) == {
        "val/sigma_hat_log_ratio_mean",
        "val/sigma_hat_log_ratio_rms",
        "val/sigma_hat_ratio_0.01_30",
    }
    assert logged["val/sigma_hat_ratio_0.01_30"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("edges", [[1.0], [1.0, 1.0], [3.0, 1.0, 2.0]])
def test_calibration_needs_increasing_edges(edges):
    with pytest.raises(ValueError, match="increasing"):
        SigmaCalibration(sigma_edges=edges)


# --- configs -----------------------------------------------------------------------


def compose_train(experiment, *overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        return compose(
            "train", overrides=[f"experiment={experiment}", "run.path=/runs", "run.data_dir=/data", *overrides]
        )


def test_diagnostics_are_off_by_default():
    config = compose_train("gpff")
    assert "denoising_error" not in config.callbacks
    assert "sigma_calibration" not in config.callbacks


@pytest.mark.parametrize("experiment, permutation", [("gpff", True), ("gpff_no_permutation", False)])
def test_denoising_error_config_follows_the_alignment_of_the_experiment(experiment, permutation):
    config = compose_train(experiment, "+callbacks=denoising_error")
    callback = hydra.utils.instantiate(config.callbacks.denoising_error)

    assert isinstance(callback, DenoisingErrorCurve)
    assert callback.every_n_samples == 5_000_000
    assert callback.align_permutation is permutation
    assert callback.align_rotation is False
    assert callback.sigmas == [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def test_diagnostics_are_enabled_next_to_the_milestones_with_the_list_form():
    # Hydra rejects two +callbacks= overrides; the list appends to the defaults
    config = compose_train("gpff", "+callbacks=[milestones,denoising_error,sigma_calibration]")
    assert {"milestone_checkpoint", "denoising_error", "sigma_calibration", "model_checkpoint"} <= set(config.callbacks)

    callbacks = [hydra.utils.instantiate(config.callbacks[name]) for name in ("denoising_error", "sigma_calibration")]
    assert [type(callback) for callback in callbacks] == [DenoisingErrorCurve, SigmaCalibration]


def test_sigma_calibration_config_builds():
    config = compose_train("gpff", "+callbacks=sigma_calibration")
    callback = hydra.utils.instantiate(config.callbacks.sigma_calibration)

    assert isinstance(callback, SigmaCalibration)
    assert callback.every_n_samples == 5_000_000
    assert callback.sigma_edges[0] == 0.01 and callback.sigma_edges[-1] == 30.0


# --- in a real trainer ---------------------------------------------------------------


class _Loader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _TinyTask(pl.LightningModule):
    """A force model that is exactly right, behind the interface of the task."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.outputs = [SimpleNamespace(name=keys.forces)]
        self.model = SimpleNamespace(output_modules=[SimpleNamespace(include_time=False)])

    def predict_without_postprocessing(self, inputs):
        return {keys.forces: perfect(inputs)}

    def training_step(self, batch, batch_idx):
        return self.layer(torch.ones(1, 1)).sum()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", torch.tensor(1.0), on_epoch=True, batch_size=3)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _fit(tmp_path, name, max_epochs, callbacks, ckpt_path=None):
    batches = [make_batch(), make_batch(seed=1)]
    logger = CSVLogger(tmp_path, name=name)
    pl.Trainer(
        max_epochs=max_epochs, accelerator="cpu", devices=1, logger=logger, callbacks=callbacks,
        num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=any(isinstance(c, ModelCheckpoint) for c in callbacks),
        default_root_dir=tmp_path,
    ).fit(_TinyTask(), _Loader(batches), _Loader(batches), ckpt_path=ckpt_path)
    with open(Path(logger.log_dir) / "metrics.csv") as file:
        return list(csv.DictReader(file))


def _epochs_with(rows, name):
    return [int(float(row["epoch"])) for row in rows if row.get(name)]


def curve_in_trainer():
    return DenoisingErrorCurve(sigmas=[0.1, 1.0, 10.0], every_n_samples=15)


def test_diagnostics_log_in_due_epochs_only_and_resume_from_a_checkpoint(tmp_path):
    # 3 molecules per batch, 2 batches per epoch: 6 samples per epoch, due at 18 and 30
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "ckpt", monitor="val/loss", save_top_k=-1)
    callbacks = [checkpoint, curve_in_trainer(), SigmaCalibration(every_n_samples=15)]
    rows = _fit(tmp_path, "first", 6, callbacks)

    assert _epochs_with(rows, "val/e_log_sigma_integral") == [2, 4]
    assert _epochs_with(rows, "val/sigma_hat_log_ratio_rms") == [2, 4]
    # the perfect model has no error and a calibrated sigma_hat
    value = next(row for row in rows if row.get("val/e_log_sigma_integral"))
    assert float(value["val/e_log_sigma_integral"]) == pytest.approx(0.0, abs=1e-6)
    assert float(value["val/sigma_hat_ratio_0.3_1"]) == pytest.approx(1.0, abs=1e-5)

    # the checkpoint of a due validation already holds the interval as served
    states = {}
    for path in Path(checkpoint.dirpath).glob("*.ckpt"):
        state = torch.load(path, weights_only=False)["callbacks"]["DenoisingErrorCurve"]
        states[state["samples_seen"]] = state["served"]
    assert states == {6: 0, 12: 0, 18: 1, 24: 1, 30: 2, 36: 2}

    # resume after the second run: the next multiple, 45, is passed in epoch 7
    resumed = [curve_in_trainer(), SigmaCalibration(every_n_samples=15)]
    after_epoch_4 = next(
        path for path in Path(checkpoint.dirpath).glob("*.ckpt")
        if torch.load(path, weights_only=False)["callbacks"]["DenoisingErrorCurve"]["samples_seen"] == 30
    )
    rows = _fit(tmp_path, "second", 8, resumed, ckpt_path=str(after_epoch_4))

    assert resumed[0].samples_seen == 48
    assert _epochs_with(rows, "val/e_log_sigma_integral") == [7]
    assert _epochs_with(rows, "val/sigma_hat_log_ratio_rms") == [7]
