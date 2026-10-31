from types import SimpleNamespace
from unittest.mock import Mock

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from schnetpack import properties
from torch.utils.data import DataLoader, Dataset

from gpff.callbacks import ExponentialMovingAverage, SampleCheckpoint


class Molecules(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {properties.n_atoms: torch.tensor(3), "x": torch.ones(1)}


class TinyTask(LightningModule):
    def __init__(self, size=10, batch_size=4):
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.model.weight)
        self.size = size
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        return self.model(batch["x"]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def train_dataloader(self):
        return DataLoader(Molecules(self.size), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(Molecules(2), batch_size=2)

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self.model(batch["x"]).square().mean())

    def save_model(self, path, do_postprocessing):
        assert do_postprocessing
        torch.save(self.model.state_dict(), path)


def make_trainer(tmp_path, callbacks, **kwargs):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=kwargs.pop("max_epochs", 1),
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        **kwargs,
    )


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("batch_size,step", [(512, 2), (128, 8)])
def test_target_rounds_up_to_completed_batch(tmp_path, batch_size, step):
    callback = SampleCheckpoint([1000], dirpath=str(tmp_path / "samples"))
    make_trainer(tmp_path, [callback]).fit(TinyTask(size=1024, batch_size=batch_size))

    checkpoint = load_checkpoint(tmp_path / "samples/samples_1000.ckpt")
    assert checkpoint["global_step"] == step
    assert checkpoint["callbacks"][callback.state_key]["samples_seen"] == 1024
    exported = load_checkpoint(tmp_path / "samples/samples_1000_model")
    assert torch.equal(exported["weight"], checkpoint["state_dict"]["model.weight"])


def test_short_batches_accumulation_and_multiple_targets(tmp_path):
    callback = SampleCheckpoint([5, 7, 9, 11], dirpath=str(tmp_path / "samples"))
    make_trainer(tmp_path, [callback], accumulate_grad_batches=2).fit(TinyTask())

    # Two full batches feed update 1; the short final batch feeds update 2.
    for target, step, count in [(5, 1, 8), (7, 1, 8), (9, 2, 10)]:
        checkpoint = load_checkpoint(tmp_path / f"samples/samples_{target}.ckpt")
        assert checkpoint["global_step"] == step
        state = checkpoint["callbacks"][callback.state_key]
        assert state["samples_seen"] == count
        assert target in state["saved_samples"]
    assert callback.samples_seen == 10  # excludes the two validation molecules
    assert not (tmp_path / "samples/samples_11.ckpt").exists()


def test_ema_snapshots_do_not_change_training(tmp_path):
    reference_ema = ExponentialMovingAverage(decay=0.9)
    reference = TinyTask()
    reference_trainer = make_trainer(tmp_path / "reference", [reference_ema], max_epochs=2)
    reference_trainer.fit(reference)

    ema = ExponentialMovingAverage(decay=0.9)
    callback = SampleCheckpoint([7, 9, 15, 17], dirpath=str(tmp_path / "samples"))
    task = TinyTask()
    # Deliberately put checkpointing first: Lightning must run it after EMA.
    trainer = make_trainer(tmp_path / "actual", [callback, ema], max_epochs=2)
    trainer.fit(task)

    assert callback.samples_seen == 20
    assert torch.equal(task.model.weight, reference.model.weight)
    assert torch.equal(ema.ema.shadow_params[0], reference_ema.ema.shadow_params[0])
    assert torch.equal(ema.ema.collected_params[0], reference_ema.ema.collected_params[0])
    for target, count in [(7, 8), (9, 10), (15, 18), (17, 18)]:
        checkpoint = load_checkpoint(tmp_path / f"samples/samples_{target}.ckpt")
        ema_state = checkpoint["callbacks"][ema.state_key]["ema"]
        weights = checkpoint["state_dict"]["model.weight"]
        exported = load_checkpoint(tmp_path / f"samples/samples_{target}_model")
        assert torch.equal(weights, ema_state["shadow_params"][0])
        assert torch.equal(weights, exported["weight"])
        assert not torch.equal(weights, ema_state["collected_params"][0])
        assert checkpoint["callbacks"][callback.state_key]["samples_seen"] == count


class ResumeWeights(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not hasattr(self, "first_weights"):
            self.first_weights = pl_module.model.weight.detach().clone()


def test_mid_epoch_resume_restores_counter_ema_and_raw_training_weights(tmp_path):
    callback = SampleCheckpoint([7, 15], dirpath=str(tmp_path / "original"))
    ema = ExponentialMovingAverage(decay=0.9)
    original = TinyTask(size=12)
    make_trainer(tmp_path / "first", [ema, callback], max_epochs=2).fit(original)
    checkpoint_path = tmp_path / "original/samples_7.ckpt"
    checkpoint = load_checkpoint(checkpoint_path)
    raw_weights = checkpoint["callbacks"][ema.state_key]["ema"]["collected_params"][0]

    resumed = SampleCheckpoint([7, 15], dirpath=str(tmp_path / "resumed"))
    resumed_ema, probe = ExponentialMovingAverage(decay=0.9), ResumeWeights()
    trainer = make_trainer(tmp_path / "second", [resumed_ema, probe, resumed], max_epochs=2)
    task = TinyTask(size=12)
    trainer.fit(task, ckpt_path=checkpoint_path, weights_only=False)

    assert torch.equal(probe.first_weights, raw_weights)
    assert torch.equal(task.model.weight, original.model.weight)
    assert torch.equal(resumed_ema.ema.collected_params[0], ema.ema.collected_params[0])
    assert resumed.samples_seen == 24
    assert not (tmp_path / "resumed/samples_7.ckpt").exists()
    later = load_checkpoint(tmp_path / "resumed/samples_15.ckpt")
    assert later["global_step"] == 4
    assert later["callbacks"][resumed.state_key]["samples_seen"] == 16
    assert later["callbacks"][resumed.state_key]["saved_samples"] == [7, 15]


def test_global_count_and_only_global_zero_writes_inference_model(tmp_path):
    callback = SampleCheckpoint([5], dirpath=str(tmp_path))
    trainer = SimpleNamespace(
        global_step=1,
        world_size=2,
        strategy=SimpleNamespace(reduce=Mock(return_value=torch.tensor(6))),
        callbacks=[],
        save_checkpoint=Mock(),
        is_global_zero=False,
    )
    task = SimpleNamespace(device=torch.device("cpu"), save_model=Mock())
    callback.on_train_batch_end(
        trainer, task, None, {properties.n_atoms: torch.tensor([2, 8, 4])}, 0
    )

    assert callback.samples_seen == 6
    trainer.strategy.reduce.assert_called_once()
    assert trainer.strategy.reduce.call_args.kwargs == {"reduce_op": "sum"}
    trainer.save_checkpoint.assert_called_once_with(str(tmp_path / "samples_5.ckpt"))
    task.save_model.assert_not_called()


def test_resuming_legacy_checkpoint_requires_explicit_sample_offset(tmp_path):
    trainer = SimpleNamespace(global_step=10, callbacks=[])
    with pytest.raises(ValueError, match="initial_samples_seen"):
        SampleCheckpoint([100]).on_train_start(trainer, None)

    callback = SampleCheckpoint([20, 100], initial_samples_seen=40)
    callback.on_train_start(trainer, None)
    assert callback.samples_seen == 40
    assert callback.state_dict()["saved_samples"] == [20]


@pytest.mark.parametrize("samples", [[], [0], [-1], [1.5], [True]])
def test_invalid_targets_are_rejected(samples):
    with pytest.raises(ValueError, match="positive integer"):
        SampleCheckpoint(samples)


def test_sample_milestones_hydra_opt_in():
    with initialize_config_module(config_module="gpff.configs", version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=gpff",
                "+callbacks=sample_milestones",
                "callbacks.sample_checkpoint.samples=[1000,1000000]",
            ],
        )
    callback = instantiate(config.callbacks.sample_checkpoint)
    assert isinstance(callback, SampleCheckpoint)
    assert callback.samples == [1000, 1000000]
    assert "model_checkpoint" in config.callbacks
