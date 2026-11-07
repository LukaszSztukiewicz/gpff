"""Validation-time diagnostics of the learned force field, run every N training
samples instead of every epoch.

``EveryNSamples`` counts the molecules the model has been trained on and lets a
subclass work on the validation set of the first validation after each multiple
of ``every_n_samples``. On every other epoch it costs one integer addition per
training batch and one boolean test per validation batch: no tensor operation,
no host synchronization. The diagnostics themselves queue their forward passes
on the device without waiting for them and read the result back once, at the
end of the validation epoch, where the logger flush synchronizes anyway.

``DenoisingErrorCurve`` and ``SigmaCalibration`` log to the configured logger
(including W&B) next to ``val/loss``. Both need a time-agnostic force model.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import Callback
from schnetpack import properties
from schnetpack.nn import scatter_add

from gpff import keys
from gpff.analysis import field

__all__ = ["EveryNSamples", "DenoisingErrorCurve", "SigmaCalibration"]


class EveryNSamples(Callback):
    """Base class of callbacks that run on the validation set after every
    ``every_n_samples`` training samples.

    A sample is a molecule of a training batch, counted over all devices. Once
    the count passes a multiple of ``every_n_samples`` the callback is *due*,
    and the next validation epoch (not the sanity check) is a *due* one: the
    ``on_due_validation_*`` hooks run in it, nowhere else. If several multiples
    pass between two validations, it runs once. The check is therefore as
    coarse as the validation schedule, e.g. an epoch.

    The counts are saved in checkpoints, so a resumed run continues from where
    the original stopped; a callback added to a run on resume counts from zero.
    An interval is marked served at the end of the validation epoch, before the
    checkpoint callbacks save, so a checkpoint saved in that validation does
    not make the resumed run repeat it.

    Subclasses implement the ``on_due_validation_*`` hooks. To stay out of the
    training's way they must not synchronize with the device in
    ``on_due_validation_batch_end``: accumulate on the device and read it back
    in ``on_due_validation_epoch_end``, the only hook that may call
    ``pl_module.log`` (Lightning allows it there and not in
    ``on_validation_end``).

    Args:
        every_n_samples: the interval in training samples. One epoch of QM9 is
            about 99k samples (80% of the 123,569 molecules of the paper's
            database), so 5,000,000 is about 50 epochs.
    """

    def __init__(self, every_n_samples: int) -> None:
        super().__init__()
        if every_n_samples < 1:
            raise ValueError(f"every_n_samples must be positive, got {every_n_samples}.")
        self.every_n_samples = int(every_n_samples)
        self.samples_seen = 0
        # multiples of every_n_samples that were served already
        self._served = 0
        self._running = False

    @property
    def due(self) -> bool:
        """Whether the samples seen passed a multiple of ``every_n_samples``
        that no validation has served yet."""
        return self.samples_seen // self.every_n_samples > self._served

    def state_dict(self) -> Dict[str, int]:
        return {"samples_seen": self.samples_seen, "served": self._served}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        self.samples_seen = state_dict.get("samples_seen", 0)
        self._served = state_dict.get("served", 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # the shape of a device tensor is known on the host: no synchronization.
        # The ranks of a distributed run see equally many samples.
        self.samples_seen += batch[properties.n_atoms].shape[0] * trainer.world_size

    def on_validation_start(self, trainer, pl_module) -> None:
        self._running = self.due and not trainer.sanity_checking
        if self._running:
            self.on_due_validation_start(trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if self._running:
            self.on_due_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._running:
            self.on_due_validation_epoch_end(trainer, pl_module)
            # marked before on_validation_end, where the checkpoints are saved
            self._served = self.samples_seen // self.every_n_samples
            self._running = False

    def on_due_validation_start(self, trainer, pl_module) -> None:
        """Start of a due validation epoch."""

    def on_due_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """End of a validation batch of a due validation epoch."""

    def on_due_validation_epoch_end(self, trainer, pl_module) -> None:
        """End of a due validation epoch: log here."""


def _require_time_agnostic_force_model(pl_module: Any, name: str) -> None:
    """The diagnostics read the force output as ``D(x) = x + F/2`` of a model
    that is told nothing but the noisy positions."""
    if not any(getattr(output, "name", None) == keys.forces for output in getattr(pl_module, "outputs", ())):
        raise ValueError(
            f"{name} needs a force model (an output named '{keys.forces}'), not a noise "
            f"prediction model."
        )
    heads = getattr(getattr(pl_module, "model", None), "output_modules", ())
    if any(getattr(head, "include_time", False) for head in heads):
        raise ValueError(f"{name} needs a time-agnostic model: it re-noises molecules without a time input.")


class DenoisingErrorCurve(EveryNSamples):
    """Log the validation denoising error of a force model at fixed noise
    levels, ``val/e_sigma_<sigma>``, and its integral over log(sigma),
    ``val/e_log_sigma_integral`` (see ``gpff.analysis.field.denoising_errors``).

    Unlike ``val/loss``, which mixes randomly drawn noise levels under the
    training loss weights, the curve shows which noise levels the model gets
    wrong, in units of the noise: 1 removes nothing, a perfect model approaches
    ``1/(N-1)`` at small sigma. It costs one forward pass per sigma and
    validation batch, only in due epochs.

    The noise comes from a generator seeded per validation batch, so the curve
    is a deterministic function of the weights and the global RNGs the training
    draws from stay untouched. The validation batches are the same in every
    epoch, so the noisy molecules are too: their alignment, the only CPU bound
    part, is solved once and kept in host memory (``12 * len(sigmas)`` bytes
    per validation atom in float32), and later runs only queue forward passes.
    ``cache_alignment=False`` solves it again every time instead.

    Args:
        sigmas: the noise levels, at least two.
        every_n_samples: see ``EveryNSamples``.
        align_permutation: permute the noisy atoms to their nearest same-element
            atoms of the clean molecule, as the training transform does. Match
            ``globals.align_permutation``.
        align_rotation: rotate them onto the clean molecule, as the training
            transform does. Match ``globals.align_rotation``.
        seed: seed of the noise.
        cache_alignment: keep the aligned noisy molecules between due epochs.
    """

    def __init__(
        self,
        sigmas: Sequence[float],
        every_n_samples: int = 5_000_000,
        align_permutation: bool = True,
        align_rotation: bool = False,
        seed: int = 0,
        cache_alignment: bool = True,
    ) -> None:
        super().__init__(every_n_samples)
        self.sigmas = sorted(float(sigma) for sigma in sigmas)
        if len(self.sigmas) < 2 or self.sigmas[0] <= 0:
            raise ValueError("DenoisingErrorCurve needs at least two positive noise levels.")
        self.align_permutation = align_permutation
        self.align_rotation = align_rotation
        self.seed = seed
        self.cache_alignment = cache_alignment
        self._cache: Dict[int, torch.Tensor] = {}
        self._sums: Optional[torch.Tensor] = None
        self._count = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        _require_time_agnostic_force_model(pl_module, type(self).__name__)

    def on_due_validation_start(self, trainer, pl_module) -> None:
        self._sums = None
        self._count = 0

    def _host_positions(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """The aligned noisy molecules of a validation batch, computed on the
        CPU in double precision like the training transform."""
        n_atoms = batch[properties.n_atoms].cpu()
        idx_m = batch[properties.idx_m].cpu()
        x_0 = field.center_per_molecule(batch[keys.x_0].cpu().double(), idx_m, n_atoms)
        seed = int(np.random.SeedSequence((self.seed, batch_idx)).generate_state(1)[0])
        noise = field.mean_free_noise(x_0, idx_m, n_atoms, torch.Generator().manual_seed(seed))
        positions = field.noisy_positions(
            x_0, batch[properties.Z].cpu(), n_atoms, self.sigmas, noise,
            self.align_permutation, self.align_rotation,
        )
        positions = positions.to(batch[properties.R].dtype)
        # pinned memory lets the copy back to the device run asynchronously
        return positions.pin_memory() if batch[properties.R].is_cuda else positions

    def _positions(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        positions = self._cache.get(batch_idx)
        if positions is None or positions.shape[1] != batch[properties.R].shape[0]:
            positions = self._host_positions(batch, batch_idx)
            if self.cache_alignment:
                self._cache[batch_idx] = positions
        return positions.to(batch[properties.R].device, non_blocking=True)

    @torch.no_grad()
    def on_due_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        errors = field.denoising_errors(
            pl_module.predict_without_postprocessing, batch, batch[keys.x_0], self.sigmas,
            self._positions(batch, batch_idx),
        )
        total = errors.sum(0, dtype=torch.float64)
        self._sums = total if self._sums is None else self._sums + total
        self._count += errors.shape[0]

    def on_due_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._count == 0:
            return
        errors = self._sums / self._count
        log_sigmas = torch.log(torch.tensor(self.sigmas, dtype=errors.dtype, device=errors.device))
        metrics = {f"val/e_sigma_{sigma:g}": error for sigma, error in zip(self.sigmas, errors.tolist())}
        metrics["val/e_log_sigma_integral"] = torch.trapezoid(errors, log_sigmas).item()
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=False)
        self._sums = None


class SigmaCalibration(EveryNSamples):
    """Log how well the noise level estimate ``sigma_hat = std(F_hat/2)`` of a
    force model matches the std of its target, ``sigma_target = std(x_0 - x_t)``
    (what a perfect model predicts), on the validation set. The adaptive
    samplers and direct denoising use sigma_hat as the noise level.

    For every validation molecule ``r = sigma_hat / sigma_target``; 1 is
    calibrated. Logged, in due epochs only:

    - ``val/sigma_hat_ratio_<lo>_<hi>``: the geometric mean of r over the
      molecules whose noise level sigma lies in the bin ``[lo, hi)`` (the first
      and last bin also take lower and higher sigma). Empty bins are not logged.
    - ``val/sigma_hat_log_ratio_mean``, ``val/sigma_hat_log_ratio_rms``: the
      bias and the root mean square of ``log r`` over all molecules.

    The bins use the noise level the molecule was drawn with, ``keys.sigma_curr``,
    while the target is the std of the displacement the model is shown. They
    agree without permutation alignment; with it the displacement is shorter at
    large sigma, and a calibrated model follows that.

    Costs one extra forward pass per validation batch, only in due epochs, and
    never waits for the device.

    Args:
        every_n_samples: see ``EveryNSamples``.
        sigma_edges: increasing bin edges of the noise level. The default spans
            the noise levels of training (``sigma_max`` 30) in half decades.
    """

    def __init__(
        self,
        every_n_samples: int = 5_000_000,
        sigma_edges: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    ) -> None:
        super().__init__(every_n_samples)
        self.sigma_edges = [float(edge) for edge in sigma_edges]
        if len(self.sigma_edges) < 2 or any(a >= b for a, b in zip(self.sigma_edges, self.sigma_edges[1:])):
            raise ValueError("sigma_edges must be at least two increasing values.")
        self._bins: Optional[Dict[str, torch.Tensor]] = None

    def on_fit_start(self, trainer, pl_module) -> None:
        _require_time_agnostic_force_model(pl_module, type(self).__name__)

    def on_due_validation_start(self, trainer, pl_module) -> None:
        self._bins = None

    @torch.no_grad()
    def on_due_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        idx_m = batch[properties.idx_m]
        n_atoms = batch[properties.n_atoms]
        # a fresh dict: schnetpack models write their outputs into the inputs
        forces = pl_module.predict_without_postprocessing({**batch})[keys.forces]
        sigma_hat = field.std_per_molecule(forces / 2, idx_m, n_atoms)
        sigma_target = field.std_per_molecule(batch[keys.x_0] - batch[properties.R], idx_m, n_atoms)
        tiny = torch.finfo(sigma_hat.dtype).tiny
        log_ratio = (sigma_hat.clamp_min(tiny).log() - sigma_target.clamp_min(tiny).log()).double()

        # the noise level of a molecule is the same on all of its atoms
        sigma = scatter_add(batch[keys.sigma_curr].reshape(-1), idx_m, n_atoms.shape[0])
        sigma = sigma / n_atoms.to(sigma.dtype)
        inner = torch.tensor(self.sigma_edges[1:-1], dtype=sigma.dtype, device=sigma.device)
        # right=True: the bins are [lo, hi), the first and last open ended
        bins = torch.bucketize(sigma, inner, right=True) if len(inner) else torch.zeros_like(sigma, dtype=torch.long)

        if self._bins is None:
            n_bins = len(self.sigma_edges) - 1
            zeros = torch.zeros(n_bins, dtype=torch.float64, device=sigma.device)
            self._bins = {
                "log_sum": zeros.clone(),
                "count": zeros.clone(),
                "square_sum": torch.zeros((), dtype=torch.float64, device=sigma.device),
            }
        self._bins["log_sum"].index_add_(0, bins, log_ratio)
        self._bins["count"].index_add_(0, bins, torch.ones_like(log_ratio))
        self._bins["square_sum"] += log_ratio.square().sum()

    def on_due_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._bins is None:
            return
        log_sum = self._bins["log_sum"].tolist()
        count = self._bins["count"].tolist()
        n = sum(count)
        if n == 0:
            return
        metrics = {
            "val/sigma_hat_log_ratio_mean": sum(log_sum) / n,
            "val/sigma_hat_log_ratio_rms": (self._bins["square_sum"].item() / n) ** 0.5,
        }
        for low, high, bin_log_sum, bin_count in zip(self.sigma_edges, self.sigma_edges[1:], log_sum, count):
            if bin_count > 0:
                metrics[f"val/sigma_hat_ratio_{low:g}_{high:g}"] = float(np.exp(bin_log_sum / bin_count))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=False)
        self._bins = None
