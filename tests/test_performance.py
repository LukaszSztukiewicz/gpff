from types import SimpleNamespace

import pytest
import torch
from schnetpack import properties

import gpff.performance as monitoring


def test_epoch_timing_loss_extrema_counts_and_resume(monkeypatch):
    clock = iter([10.0, 13.0, 15.0, 16.0])
    monkeypatch.setattr(monitoring.time, "perf_counter", lambda: next(clock))
    logged = {}
    module = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(output_modules=[SimpleNamespace(output_gain=torch.tensor(1.25))]),
        log_dict=lambda values, **kwargs: logged.update(values),
    )
    trainer = SimpleNamespace(current_epoch=4, sanity_checking=False)
    callback = monitoring.PerformanceMetrics()
    callback.on_train_epoch_start(trainer, module)
    for i, loss in enumerate([2.0, float("nan"), 8.0, float("inf")]):
        callback.on_train_batch_end(
            trainer, module, {"loss": torch.tensor(loss, requires_grad=True)},
            {properties.n_atoms: torch.tensor([3, 5])}, i,
        )
    callback.on_validation_start(trainer, module)
    callback.on_validation_end(trainer, module)
    callback.on_train_epoch_end(trainer, module)
    assert logged["perf/train_seconds"] == 4.0
    assert logged["perf/validation_seconds"] == 2.0
    assert logged["perf/epoch_seconds"] == 6.0
    assert logged["perf/train_molecules_per_second"] == 2.0
    assert logged["perf/examples_seen"] == 8
    assert logged["perf/completed_epochs"] == 5
    assert logged["train/finite_batch_fraction"] == 0.5
    assert logged["train/nonfinite_batches"] == 2
    assert logged["train/max_finite_loss"] == 8.0
    assert logged["model/output_gain_at_epoch_end"] == 1.25
    resumed = monitoring.PerformanceMetrics()
    resumed.load_state_dict(callback.state_dict())
    assert resumed.examples_seen == 8
    assert callback._max_finite_loss.grad_fn is None


def test_no_validation_or_gain_and_all_losses_nonfinite(monkeypatch):
    clock = iter([1.0, 3.0])
    monkeypatch.setattr(monitoring.time, "perf_counter", lambda: next(clock))
    logged = {}
    module = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(output_modules=[SimpleNamespace()]),
        log_dict=lambda values, **kwargs: logged.update(values),
    )
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False)
    callback = monitoring.PerformanceMetrics()
    callback.on_train_epoch_start(trainer, module)
    callback.on_train_batch_end(
        trainer, module, torch.tensor(float("inf")),
        {properties.n_atoms: torch.tensor([3])}, 0,
    )
    callback.on_train_epoch_end(trainer, module)
    assert logged["perf/train_seconds"] == 2.0
    assert logged["perf/validation_seconds"] == 0.0
    assert logged["train/finite_batch_fraction"] == 0.0
    assert "train/max_finite_loss" not in logged
    assert "model/output_gain_at_epoch_end" not in logged


@pytest.mark.parametrize("accelerator", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)])
def test_real_lightning_fit_logs_epoch_metrics(tmp_path, accelerator):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader

    class Task(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.output_modules = torch.nn.ModuleList([torch.nn.Linear(3, 1)])

        def training_step(self, batch, batch_idx):
            return self.model.output_modules[0](batch["x"]).square().mean()

        def validation_step(self, batch, batch_idx):
            self.log("val/loss", self.model.output_modules[0](batch["x"]).square().mean())

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    batches = [{properties.n_atoms: torch.tensor([3, 4]), "x": torch.ones(2, 3)}] * 2
    callback = monitoring.PerformanceMetrics()
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, max_epochs=2, num_sanity_val_steps=0,
        logger=CSVLogger(tmp_path), callbacks=[callback], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1,
    )
    trainer.fit(Task(), DataLoader(batches, batch_size=None), DataLoader(batches[:1], batch_size=None))
    assert trainer.callback_metrics["perf/examples_seen"] == 8
    assert trainer.callback_metrics["perf/completed_epochs"] == 2
    assert trainer.callback_metrics["train/finite_batch_fraction"] == 1.0
    assert trainer.callback_metrics["perf/train_seconds"] > 0
    assert trainer.callback_metrics["perf/validation_seconds"] > 0
    assert trainer.callback_metrics["perf/epoch_seconds"] >= trainer.callback_metrics["perf/train_seconds"]
    if accelerator == "cuda":
        assert trainer.callback_metrics["perf/peak_allocated_gib"] > 0
        assert trainer.callback_metrics["perf/peak_reserved_gib"] >= trainer.callback_metrics["perf/peak_allocated_gib"]


def test_multiple_validation_passes_are_subtracted(monkeypatch):
    clock = iter([10.0, 12.0, 13.0, 15.0, 17.0, 20.0])
    monkeypatch.setattr(monitoring.time, "perf_counter", lambda: next(clock))
    logged = {}
    module = SimpleNamespace(
        device=torch.device("cpu"), model=SimpleNamespace(),
        log_dict=lambda values, **kwargs: logged.update(values),
    )
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False)
    callback = monitoring.PerformanceMetrics()
    callback.on_train_epoch_start(trainer, module)
    for _ in range(2):
        callback.on_validation_start(trainer, module)
        callback.on_validation_end(trainer, module)
    callback.on_train_epoch_end(trainer, module)
    assert logged["perf/epoch_seconds"] == 10.0
    assert logged["perf/validation_seconds"] == 3.0
    assert logged["perf/train_seconds"] == 7.0


@pytest.mark.parametrize("experiment", ["gpff", "gpff_stable", "dm_t_aware"])
def test_performance_metrics_enabled_by_default_and_can_be_disabled(experiment):
    import gpff.cli.train
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    with initialize_config_module(config_module="gpff.configs", version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])
        assert isinstance(instantiate(cfg.callbacks.performance_metrics), monitoring.PerformanceMetrics)
        disabled = compose(config_name="train", overrides=[
            f"experiment={experiment}", "~callbacks.performance_metrics",
        ])
        assert "performance_metrics" not in disabled.callbacks
