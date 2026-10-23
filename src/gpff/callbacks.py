import os
import random
import torch
import numpy as np
import time
from typing import Any, Dict, List, Optional
from pytorch_lightning import Callback
from gpff.sample.base import Sampler
from gpff.sample.priors import Prior
from gpff.sample.results import batches_to_relaxed
from gpff.analysis import get_rdkit_validity
from ase.db import connect
from ase import Atoms
from schnetpack.train.callbacks import ModelCheckpoint as SPKModelCheckpoint
import pytorch_lightning as pl
import schnetpack.transform as trn
from schnetpack import properties
from gpff.utils.wandb import upload_artifact


__all__ = [
    "SamplerCallback",
    "ModelCheckpoint",
    "MilestoneCheckpoint",
    "DeterministicDataSeed",
]


class DeterministicDataSeed(Callback):
    """Make each epoch's data randomness (shuffle + transform noise in the
    DataLoader workers) a pure function of ``(base_seed, epoch)``, in the spirit
    of ``DistributedSampler.set_epoch``. PL does not checkpoint RNG state, so
    without this a resumed run consumes a different RNG stream than a continuous
    run and silently diverges from it.

    By default each worker draws the noise of its batches from its own stream,
    so the noise also depends on the number of workers. With
    ``per_sample_train_noise`` it is a function of ``(base_seed, epoch,
    database index)`` alone, which persistent workers need: their RNG state
    would otherwise carry over from the epochs before a resume.
    """

    def __init__(
        self,
        base_seed: int,
        val_noise_seed: Optional[int] = 0,
        per_sample_train_noise: bool = False,
    ) -> None:
        super().__init__()
        # the run's config seed; identical on launch and resume
        self._base_seed = base_seed
        # independent of base_seed, so runs with different seeds share val noise
        self._val_noise_seed = val_noise_seed
        self._per_sample_train_noise = per_sample_train_noise
        # the epoch the training noise is seeded from; in shared memory so that
        # persistent DataLoader workers see the main process' updates
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._first_epoch_reiterated = False

    def setup(self, trainer, pl_module, stage) -> None:
        # Runs after datamodule.setup, which resets the dataset transforms, so
        # wrapping here never nests. Pins val/test noise to each molecule's
        # database index: val/loss becomes a deterministic function of the
        # weights. Training noise stays fresh every epoch; with
        # per_sample_train_noise it is pinned to (base_seed, epoch, index).
        dm = trainer.datamodule
        if dm is None:
            return
        if self._per_sample_train_noise:
            if dm.train_dataset is not None:
                dm.train_dataset.transforms = [
                    _PinnedNoise(self._base_seed, dm.train_transforms, epoch=self._epoch)
                ]
        elif getattr(dm, "persistent_workers", False):
            raise ValueError(
                "Persistent workers carry their RNG state across epochs, so a resumed "
                "run would draw different training noise. Set "
                "callbacks.deterministic_data_seed.per_sample_train_noise=true."
            )
        if self._val_noise_seed is None:
            return
        for dataset, transforms in (
            (dm.val_dataset, dm.val_transforms),
            (dm.test_dataset, dm.test_transforms),
        ):
            if dataset is not None:
                dataset.transforms = [_PinnedNoise(self._val_noise_seed, transforms)]

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # hashed, not base_seed + epoch, so that runs with nearby seeds do not
        # see each other's shuffles one epoch apart
        entropy = (self._base_seed, int(trainer.current_epoch))
        seed = int(np.random.SeedSequence(entropy).generate_state(1)[0])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # read by the training transforms of the iterator created below
        self._epoch.fill_(int(trainer.current_epoch))

        # PL (through 2.6.6) creates the first training epoch's dataloader iterator before
        # this hook fires (in _FitLoop.setup_data), so its shuffle permutation and
        # worker base_seed predate the reseed above — recreate it once. Later
        # epochs create their iterator after this hook and need no special-casing.
        if not self._first_epoch_reiterated:
            self._first_epoch_reiterated = True
            fetcher = trainer.fit_loop._data_fetcher
            if fetcher is not None:
                iter(fetcher)


class _PinnedNoise(trn.Transform):
    """Applies ``transforms`` with the CPU torch and python RNGs seeded from the
    sample's database index (and the current epoch, if ``epoch`` is given),
    leaving the caller's RNG state untouched."""

    def __init__(
        self,
        seed: int,
        transforms: List[trn.Transform],
        epoch: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.epoch = epoch
        self.transforms = torch.nn.Sequential(*transforms)

    def _sample_seed(self, idx: int) -> int:
        if self.epoch is None:
            # keep all entropy in the low 32 bits: torch's CPU MT19937 drops the rest
            return (self.seed * 1_000_003 + idx) & 0xFFFFFFFF
        # hashed, so that the streams of different seeds and epochs do not overlap
        entropy = (self.seed, int(self.epoch), idx)
        return int(np.random.SeedSequence(entropy).generate_state(1)[0])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seed = self._sample_seed(int(inputs[properties.idx]))
        py_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                # the CPU generator only: torch.manual_seed also seeds every
                # accelerator backend and costs ~0.75 ms per call
                torch.default_generator.manual_seed(seed)
                random.seed(seed)
                return self.transforms(inputs)
        finally:
            random.setstate(py_state)


class SamplerCallback(Callback):
    """
    Callback to sample or denoise molecules for monitoring during training.
    """

    def __init__(
        self,
        sampler: Sampler,
        prior_sampler: Prior,
        store_path: str,
        every_n_batchs: int = 1,
        every_n_epochs: int = 1,
        start_epoch: int = 1,
        log_validity: bool = True,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.prior_sampler = prior_sampler
        self.store_path = store_path
        self.every_n_batchs = every_n_batchs
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch
        self.log_validity = log_validity

        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

    def sample(
        self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> List[Atoms]:
        # update the model in the sampler if needed
        self.sampler.update_model(model)
        
        # sample prior
        batch = self.prior_sampler.sample_from_batch(batch)

        # denoise and process
        final_batch = self.sampler.denoise(batch)
        denoised_atoms = batches_to_relaxed([final_batch])

        return [atoms for atoms, _ in denoised_atoms]

    def save_samples(
        self, samples: List[Atoms], epoch: int, batch_idx: int, phase: str = "val"
    ) -> None:
        # Save to ASE database
        db_path = os.path.join(self.store_path, f"samples_{phase}_{epoch}.db")
        db = connect(db_path)
        for atoms in samples:
            db.write(atoms)

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        epoch = trainer.current_epoch
        if (
            epoch >= self.start_epoch
            and epoch % self.every_n_epochs == 0
            and batch_idx % self.every_n_batchs == 0
        ):
            # generate samples
            samples = self.sample(pl_module.model, batch)

            # save samples
            self.save_samples(samples, epoch, batch_idx, phase="val")

            # calculate and log validity
            if self.log_validity:
                validity = [get_rdkit_validity(atoms) for atoms in samples]

                # log
                pl_module.log(
                    f"val/validity",
                    np.mean(validity).item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        epoch = trainer.current_epoch

        # generate samples
        samples = self.sample(pl_module.model, batch)

        # save samples
        self.save_samples(samples, epoch, batch_idx, phase="test")

        # calculate and log validity
        if self.log_validity:
            validity = [get_rdkit_validity(atoms) for atoms in samples]

            # log
            pl_module.log(
                f"test/validity",
                np.mean(validity).item(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )


class ModelCheckpoint(SPKModelCheckpoint):
    """
    Like the PyTorch Lightning ModelCheckpoint callback,
    but also saves the best inference model with activated post-processing
    """

    def __init__(
        self, 
        model_path: str, 
        do_postprocessing: bool = True, 
        wandb_upload_id: Optional[str] = None,
        *args, **kwargs,
        ):
        super().__init__(
            model_path, 
            do_postprocessing=do_postprocessing, 
            *args, 
            **kwargs,
            )
        self.wandb_upload_id = wandb_upload_id or "wandb_upload_id"
    
    def on_fit_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        # upload the split artifact once per run; the marker in the run dir
        # keeps resumed jobs from re-uploading it
        if self.wandb_upload_id is None or trainer.strategy.local_rank != 0:
            return
        split_marker = "split_uploaded.flag"
        if os.path.exists(split_marker):
            return
        upload_artifact(
            self.wandb_upload_id,
            "split",
            "Split indices",
            trainer.datamodule.split_file,
            trainer,
        )
        with open(split_marker, "w") as f:
            f.write(str(trainer.datamodule.split_file))

    def _update_best_and_save(
        self, current: torch.Tensor, trainer, monitor_candidates: Dict[str, Any]
    ):
        # save model checkpoint
        super()._update_best_and_save(current, trainer, monitor_candidates)

        # save best inference model locally; the W&B upload is deferred to
        # on_fit_end so only one artifact version is pushed per run
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"))

        if current == self.best_model_score:
            if self.trainer.strategy.local_rank == 0:
                # remove references to trainer and data loaders to avoid pickle error in ddp
                self.task.save_model(self.model_path, do_postprocessing=True)

    def on_fit_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        # on_fit_end only fires when fit() returns normally, so interrupted
        # cycles upload nothing
        if (
            self.wandb_upload_id is not None
            and trainer.strategy.local_rank == 0
            and os.path.exists(self.model_path)
        ):
            upload_artifact(
                self.wandb_upload_id,
                "model",
                "Best model checkpoint with postprocessing",
                self.model_path,
                trainer,
            )


class MilestoneCheckpoint(Callback):
    """Keep the model after fixed numbers of training epochs, independent of
    which epoch is best; milestones a run stops before are skipped. ``N``
    counts completed epochs, so milestone ``N`` holds the weights a run with
    ``trainer.max_epochs=N`` ends with (Lightning's 0-based ``epoch=N-1``).

    Writes ``epoch_<N>.ckpt`` (Lightning checkpoint, to resume or evaluate
    from) and ``epoch_<N>_model`` (inference model, the ``model_path`` for
    ``gpff_generate``) to ``dirpath``. Both are saved after the epoch's
    validation, so like ``best_model`` they hold the validated weights (the
    EMA average when the EMA callback is on).
    """

    def __init__(self, epochs: List[int], dirpath: str = "milestones") -> None:
        super().__init__()
        self.epochs = set(epochs)
        self.dirpath = dirpath

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # current_epoch only advances after this hook
        n_epochs = trainer.current_epoch + 1
        if n_epochs not in self.epochs:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, f"epoch_{n_epochs}")
        # save_checkpoint must run on every rank; save_model writes on rank 0
        trainer.save_checkpoint(f"{path}.ckpt")
        pl_module.save_model(f"{path}_model", do_postprocessing=True)
