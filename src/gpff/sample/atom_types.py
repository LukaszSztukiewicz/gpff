"""
Direct denoising that also generates the atom types, for models with a
:class:`~gpff.model.atom_types.TypeLogits` head (experiment=gpff_ctmc_types and
experiment=gpff_continuous_types).

The positions denoise exactly as in DirectDenoisingSampler. The types start from the
prior of the model's type process, whatever atomic numbers the batch had, so no
composition of a data molecule enters the sample. Every model call predicts their clean
classes along with the forces, and after the positions' update the types take the
reverse step of their process. properties.Z follows the types: the atomic number of the
most likely class, MASK_Z while a slot is MASK. A molecule with a MASK slot does not
converge, and the last step decides every slot.

For a model that generates the number of atoms, :class:`CanvasPrior` replaces the
molecules by canvases of slots as in training; the slots that end ABSENT are dropped
from the molecules (gpff.utils.batching.batch_to_atoms_list).
"""

import math
from typing import Dict, Optional

import torch
from schnetpack import properties

from gpff import keys
from gpff.atom_types import MASK_Z, AbsorbingTypes, ContinuousTypes
from gpff.model.atom_types import TypeLogits
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.sample.priors import Prior


__all__ = ["CanvasPrior", "AbsorbingTypeSampler", "ContinuousTypeSampler"]


class CanvasPrior(Prior):
    """
    Positions from an isotropic normal on canvases of slots, for the models that generate
    the atom types; the sampler sets the types. Every molecule of the batch becomes a
    canvas of ``size`` slots, or of its own number of atoms and U{0, ..., max_extra} more:
    the canvases of training (gpff.data.transform.Canvas). With neither, a canvas has the
    size of its molecule. Only the number of molecules and, without a size, their sizes
    are taken from the batch; its elements and positions are dropped.

    Args:
        sigma: standard deviation of the positions, the run's sigma_max.
        size: number of slots of every canvas.
        max_extra: largest number of slots added to a molecule, instead of a size.
    """

    def __init__(self, sigma: float = 30.0, size: Optional[int] = None, max_extra: int = 0):
        if size is not None and max_extra:
            raise ValueError("a canvas has a size or extra slots, not both")
        if (size is not None and size < 1) or max_extra < 0:
            raise ValueError(f"invalid canvas: size {size}, max_extra {max_extra}")
        self.sigma = sigma
        self.size = size
        self.max_extra = max_extra

    def sample_from_batch(self, batch, conditions=None):
        n_atoms = batch[properties.n_atoms]
        if self.size is not None:
            sizes = torch.full_like(n_atoms, self.size)
        else:
            sizes = n_atoms + torch.randint(self.max_extra + 1, n_atoms.shape, device=n_atoms.device)
        n_slots = int(sizes.sum())
        positions = batch[properties.R]
        return {
            properties.idx: batch[properties.idx],
            properties.n_atoms: sizes,
            properties.idx_m: torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), sizes),
            properties.Z: torch.full((n_slots,), MASK_Z, dtype=batch[properties.Z].dtype, device=sizes.device),
            properties.R: self.sigma * torch.randn(n_slots, 3, dtype=positions.dtype, device=positions.device),
            properties.cell: batch[properties.cell],
            properties.pbc: batch[properties.pbc],
        }


class _TypeDenoisingSampler(DirectDenoisingSampler):
    """
    The part of the type samplers that does not depend on the process, with the
    unmasking of the absorbing process, MDLM's ancestral sampler (Sahoo et al. 2024). The
    masking rate falls from 1 before the first step to 0 after the last one. From rate r
    to r' every MASK slot is unmasked with probability (r - r') / r, drawing its class from
    the prediction, so that a slot is MASK with probability r' afterwards. As the rate is
    drawn independently of the positions' noise level in training, any schedule is one the
    model was trained for.

    Args:
        schedule: the masking rate after a share u of the steps, "cosine",
            cos(pi u / 2), which unmasks few slots early and many late (MaskGIT), or
            "linear", 1 - u.
        temperature: temperature of the predicted class distribution.
    """

    process_type: type = object

    def __init__(self, schedule: str = "cosine", temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"schedule must be cosine or linear, got {schedule}")
        if not temperature > 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.schedule = schedule
        self.temperature = temperature
        self.process = None

    def masking_rate(self, step: int) -> float:
        """The masking rate after a step (counting down from n_steps to 1); 1 before the first."""
        if step > self.n_steps:
            return 1.0
        if step <= 1:
            return 0.0
        u = self.get_denoising_ratio(step)
        return math.cos(math.pi * u / 2) if self.schedule == "cosine" else 1.0 - u

    def _unmasked(self, h: torch.Tensor, step: int) -> torch.Tensor:
        """(n_atoms,) True for the MASK slots this step unmasks."""
        before, after = self.masking_rate(step + 1), self.masking_rate(step)
        draw = torch.rand(len(h), device=h.device)
        return self.process.vocabulary.masked(h) & (draw < (before - after) / before)

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Take the type process from the model's head and start the types from its prior.
        """
        batch = super()._init_state(batch)
        heads = [m for m in self.calculator.model.modules() if isinstance(m, TypeLogits)]
        if len(heads) != 1 or not isinstance(heads[0].process, self.process_type):
            raise ValueError(
                f"{type(self).__name__} needs a model with one TypeLogits head of {self.process_type.__name__}"
            )
        self.process = heads[0].process
        batch[keys.h] = self.process.prior(batch[properties.R])
        return self._read_types(batch)

    def _read_types(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Set properties.Z from the types."""
        vocabulary = self.process.vocabulary
        h = batch[keys.h]
        numbers = vocabulary.atomic_numbers(h[:, : vocabulary.n_classes].argmax(-1))
        batch[properties.Z] = torch.where(vocabulary.masked(h), MASK_Z, numbers)
        return batch

    def _draw(self, logits: torch.Tensor) -> torch.Tensor:
        """One class per slot from softmax(logits), by the Gumbel-max trick."""
        u = torch.rand_like(logits).clamp(min=torch.finfo(logits.dtype).tiny)
        return (logits - torch.log(-torch.log(u))).argmax(-1)

    def _undecided(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(n_molecules,) True for the molecules with a MASK slot."""
        masked = self.process.vocabulary.masked(batch[keys.h]).to(torch.int32)
        undecided = torch.zeros(len(batch[properties.n_atoms]), dtype=torch.int32, device=masked.device)
        return undecided.scatter_reduce(0, batch[properties.idx_m], masked, reduce="amax", include_self=False).bool()

    def update_convergence(self, batch, value, threshold, reduce: str = "mean"):
        """As the base class, except that a molecule with a MASK slot does not converge."""
        converged = batch[keys.converged]
        batch = super().update_convergence(batch, value, threshold, reduce=reduce)
        batch[keys.converged] = converged | (batch[keys.converged] & ~self._undecided(batch))
        return batch

    def perturb_types(self, batch: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """The types the model is called with at this step; unchanged by default."""
        return batch[keys.h]

    def next_types(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        logits: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """The types after this step, from the logits at temperature."""
        raise NotImplementedError

    def preprocess_step(self, batch, step: int, conditions: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        batch = super().preprocess_step(batch, step, conditions=conditions, **kwargs)
        converged = batch[keys.converged][batch[properties.idx_m]].unsqueeze(-1)
        batch[keys.h] = torch.where(converged, batch[keys.h], self.perturb_types(batch, step))
        return batch

    def postprocess_step(
        self,
        batch,
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        predictions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        # no logits only if every molecule had converged, which ends the loop before
        if predictions is not None and keys.type_logits in predictions:
            logits = predictions[keys.type_logits] / self.temperature
            batch = self.update_batch(batch, {keys.h: self.next_types(batch, step, logits, predictions)})
            batch = self._read_types(batch)
        return super().postprocess_step(batch, step, conditions=conditions, **kwargs)


class AbsorbingTypeSampler(_TypeDenoisingSampler):
    """
    The reverse process of AbsorbingTypes over the steps of direct denoising: MASK slots
    are unmasked on the schedule, and a decided slot keeps its class, as an absorbing
    process has no path back.

    Args:
        schedule: the masking rate over the steps, "cosine" or "linear".
        temperature: temperature of the predicted class distribution.
    """

    process_type = AbsorbingTypes

    def next_types(self, batch, step, logits, predictions):
        h = batch[keys.h]
        drawn = self.process.vocabulary.one_hot(self._draw(logits), h.dtype)
        return torch.where(self._unmasked(h, step).unsqueeze(-1), drawn, h)


class ContinuousTypeSampler(_TypeDenoisingSampler):
    """
    The types of ContinuousTypes denoised like the positions: before every model call
    they are perturbed by the positions' noise, and after it they jump to the clean
    estimate s p, as the positions jump to x_0. The jumps are plain: the types take no
    transport steps.

    With a process that masks, every slot starts as MASK and is unmasked on the schedule,
    as for absorbing types, to the drawn class's one-hot s onehot(c); from then on it is
    continuous like the others, so it may still change its class.

    Args:
        schedule: with a process that masks, the masking rate over the steps, "cosine"
            or "linear".
        temperature: temperature of the predicted class distribution.
    """

    process_type = ContinuousTypes

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transport_sigma is not None:
            raise ValueError("ContinuousTypeSampler takes jumps only: transport_sigma must be None")

    def perturb_types(self, batch, step):
        h = batch[keys.h]
        vocabulary = self.process.vocabulary
        noise = self.add_noise(torch.zeros_like(h), step)
        noise[:, vocabulary.n_classes :] = 0
        return torch.where(vocabulary.masked(h).unsqueeze(-1), h, h + noise)

    def next_types(self, batch, step, logits, predictions):
        h = batch[keys.h]
        h_next = torch.where(self.process.vocabulary.masked(h).unsqueeze(-1), h, self.process.clean_estimate(logits))
        drawn = self.process.scale * self.process.vocabulary.one_hot(self._draw(logits), h.dtype)
        return torch.where(self._unmasked(h, step).unsqueeze(-1), drawn, h_next)
