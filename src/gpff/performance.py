"""Epoch timing, throughput, CUDA memory, and loss statistics for training.

Loss statistics stay on the accelerator until epoch end. CUDA synchronization
is limited to epoch/validation boundaries; no model hooks or extra forwards.
"""

import time

import torch
from pytorch_lightning import Callback
from schnetpack import properties


class PerformanceMetrics(Callback):
    """Log per-device metrics through Lightning's configured loggers."""

    def __init__(self):
        super().__init__()
        self.examples_seen = 0
        self._epoch_start = None
        self._validation_start = None

    def state_dict(self):
        return {"examples_seen": self.examples_seen}

    def load_state_dict(self, state_dict):
        self.examples_seen = state_dict.get("examples_seen", 0)

    @staticmethod
    def _sync(pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync(pl_module)
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self._epoch_start = time.perf_counter()
        self._validation_start = None
        self._validation_seconds = 0.0
        self._examples = 0
        self._loss_batches = 0
        self._finite_batches = None
        self._max_finite_loss = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        examples = batch[properties.n_atoms].shape[0]
        self._examples += examples
        self.examples_seen += examples
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if not isinstance(loss, torch.Tensor):
            return
        loss = loss.detach().float().reshape(())
        finite = torch.isfinite(loss)
        finite_loss = torch.where(finite, loss, loss.new_full((), -float("inf")))
        if self._finite_batches is None:
            self._finite_batches = finite.to(torch.int64)
            self._max_finite_loss = finite_loss
        else:
            self._finite_batches.add_(finite)
            self._max_finite_loss = torch.maximum(self._max_finite_loss, finite_loss)
        self._loss_batches += 1

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking or self._epoch_start is None:
            return
        self._sync(pl_module)
        self._validation_start = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self._validation_start is None:
            return
        self._sync(pl_module)
        self._validation_seconds += time.perf_counter() - self._validation_start
        self._validation_start = None

    def on_train_epoch_end(self, trainer, pl_module):
        self._sync(pl_module)
        epoch_seconds = time.perf_counter() - self._epoch_start
        train_seconds = epoch_seconds - self._validation_seconds
        metrics = {
            "perf/epoch_seconds": epoch_seconds,
            "perf/train_seconds": train_seconds,
            "perf/validation_seconds": self._validation_seconds,
            "perf/train_molecules_per_second": self._examples / max(train_seconds, 1e-12),
            "perf/examples_seen": self.examples_seen,
            "perf/completed_epochs": trainer.current_epoch + 1,
        }
        if pl_module.device.type == "cuda":
            metrics["perf/peak_allocated_gib"] = torch.cuda.max_memory_allocated(pl_module.device) / 2**30
            metrics["perf/peak_reserved_gib"] = torch.cuda.max_memory_reserved(pl_module.device) / 2**30
        if self._loss_batches:
            n_finite = int(self._finite_batches.item())
            metrics["train/finite_batch_fraction"] = n_finite / self._loss_batches
            metrics["train/nonfinite_batches"] = self._loss_batches - n_finite
            if n_finite:
                metrics["train/max_finite_loss"] = self._max_finite_loss.item()
        heads = getattr(pl_module.model, "output_modules", ())
        gain = getattr(heads[0], "output_gain", None) if heads else None
        if gain is not None:
            metrics["model/output_gain_at_epoch_end"] = gain.detach().item()
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=False)
        self._epoch_start = None
