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
"""

import math
from typing import Dict, Optional

import torch
from schnetpack import properties

from gpff import keys
from gpff.atom_types import MASK_Z, AbsorbingTypes, ContinuousTypes
from gpff.model.atom_types import TypeLogits
from gpff.sample.direct_denoising import DirectDenoisingSampler


__all__ = ["AbsorbingTypeSampler", "ContinuousTypeSampler"]


class _TypeDenoisingSampler(DirectDenoisingSampler):
    """
    The part of the type samplers that does not depend on the process.

    Args:
        temperature: temperature of the predicted class distribution.
    """

    process_type: type = object

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if not temperature > 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.process = None

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
    The reverse process of AbsorbingTypes, MDLM's ancestral sampler (Sahoo et al. 2024),
    over the steps of direct denoising. The masking rate falls from 1 before the first
    step to 0 after the last one. From rate r to r' every MASK slot is unmasked with
    probability (r - r') / r, drawing its class from the prediction, so that a slot is
    MASK with probability r' afterwards. A decided slot keeps its class: an absorbing
    process has no path back.

    Args:
        schedule: the masking rate after a share u of the steps, "cosine",
            cos(pi u / 2), which unmasks few slots early and many late (MaskGIT), or
            "linear", 1 - u.
        temperature: temperature of the predicted class distribution.
    """

    process_type = AbsorbingTypes

    def __init__(self, schedule: str = "cosine", **kwargs):
        super().__init__(**kwargs)
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"schedule must be cosine or linear, got {schedule}")
        self.schedule = schedule

    def masking_rate(self, step: int) -> float:
        """The masking rate after a step (counting down from n_steps to 1); 1 before the first."""
        if step > self.n_steps:
            return 1.0
        if step <= 1:
            return 0.0
        u = self.get_denoising_ratio(step)
        return math.cos(math.pi * u / 2) if self.schedule == "cosine" else 1.0 - u

    def next_types(self, batch, step, logits, predictions):
        h = batch[keys.h]
        before, after = self.masking_rate(step + 1), self.masking_rate(step)
        unmask = self.process.vocabulary.masked(h) & (torch.rand(len(h), device=h.device) < (before - after) / before)
        drawn = self.process.vocabulary.one_hot(self._draw(logits), h.dtype)
        return torch.where(unmask.unsqueeze(-1), drawn, h)


class ContinuousTypeSampler(_TypeDenoisingSampler):
    """
    The types of ContinuousTypes denoised like the positions: before every model call
    they are perturbed by the positions' noise, and after it they jump to the clean
    estimate s p, as the positions jump to x_0. The jumps are plain: the types take no
    transport steps.

    Args:
        temperature: temperature of the predicted class distribution.
    """

    process_type = ContinuousTypes

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transport_sigma is not None:
            raise ValueError("ContinuousTypeSampler takes jumps only: transport_sigma must be None")

    def perturb_types(self, batch, step):
        h = batch[keys.h]
        noise = self.add_noise(torch.zeros_like(h), step)
        noise[:, self.process.vocabulary.n_classes :] = 0
        return h + noise

    def next_types(self, batch, step, logits, predictions):
        return self.process.clean_estimate(logits)
