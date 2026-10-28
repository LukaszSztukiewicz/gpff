"""
Sampling for absorbing-state discrete diffusion over the atom types.

The reverse process of masked diffusion is the one that gives the forward
process of :mod:`gpff.data.transform.absorbing_types` back. Going from masking
rate ``r_before`` to ``r_after``, every atom that is still MASK is unmasked
independently with probability ``(r_before - r_after) / r_before``, and an atom
that already has an element keeps it (Sahoo et al. 2024). Both halves matter:

    carry-over    an unmasked type is never revisited. The absorbing process has
                  no path back, so rewriting one would sample off-distribution.
                  It is also what makes the type channel stop drifting, which is
                  the failure the continuous one-hot port shows.
    independence  unmasking per atom needs no per-molecule bookkeeping, and the
                  last step uses probability 1, so every atom ends with an
                  element whatever the schedule did before.

The positions keep denoising exactly as in
:class:`~gpff.sample.direct_denoising.DirectDenoisingSampler`, so the geometry
and the types resolve together over the same steps.

Unmasking in a random order is the correct ancestral sampler. Picking the most
confident atoms instead (MaskGIT) is the obvious variant, but in 3D it tends to
select spatially clustered atoms -- it would decode one end of the molecule
first -- so it is left out here rather than guessed at.
"""

import math
from typing import Dict, Optional

import torch

from schnetpack import properties

from gpff import keys, vocab
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.sample.priors import Prior


__all__ = ["AbsorbingPrior", "AbsorbingDirectDenoisingSampler"]

PROBABILITIES_KEY = "type_probabilities"


class AbsorbingPrior(Prior):
    """
    The t = 1 state: positions from an isotropic normal, every type MASK.

    The batch's own elements are overwritten, so nothing about the composition
    of the molecule it came from survives -- unlike the plain priors, which keep
    ``properties.Z`` and therefore generate with a real molecule's formula.

    Args:
        sigma: standard deviation of the positions, normally the run's sigma_max.
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch[properties.R] = torch.randn_like(batch[properties.R]) * self.sigma
        batch[properties.Z] = torch.full_like(batch[properties.Z], vocab.MASK_Z)
        return batch


class AbsorbingDirectDenoisingSampler(DirectDenoisingSampler):
    """
    Direct denoising of the positions with absorbing-state unmasking of the types.

    Args:
        type_schedule: how the masking rate falls, "cosine" (concave, as in
            MaskGIT, which unmasks few atoms early and many late) or "linear".
        type_temperature: temperature of the draw; < 1 sharpens towards the
            most likely element, > 1 flattens.
        type_seed: seed of the generator the type draws come from. It is a
            generator of its own, so the positions see the same noise whatever
            the type settings are.
        require_types_for_convergence: do not let a molecule count as converged
            while any of its atoms is still MASK.
        remask_prob: strength of the remasking corrector. 0 keeps the pure
            carry-over reverse process. Above 0, already decided atoms are put
            back to MASK with probability remask_prob * masking_rate(step), so
            they are decided again later with more context. The factor
            masking_rate(step) is 0 on the last step, so sampling still ends with
            every type decided.
        remask_mode: which decided atoms to put back. "confidence" remasks in
            proportion to 1 - p(the type it currently holds), i.e. the ones the
            model has come to doubt, which is the targeted-remasking idea; the
            correction is towards MASK rather than towards another guess, so no
            wrong type is ever swapped straight for another. "random" remasks
            uniformly, as the ablation.
    """

    def __init__(
        self,
        type_schedule: str = "cosine",
        type_temperature: float = 1.0,
        type_seed: int = 0,
        require_types_for_convergence: bool = True,
        remask_prob: float = 0.0,
        remask_mode: str = "confidence",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if remask_prob < 0:
            raise ValueError("remask_prob must be non-negative.")
        if remask_mode not in ("confidence", "random"):
            raise ValueError(
                f"Unknown remask_mode {remask_mode!r}, expected confidence or random."
            )
        self.remask_prob = remask_prob
        self.remask_mode = remask_mode
        if type_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"Unknown type_schedule {type_schedule!r}, expected cosine or linear."
            )
        if type_temperature <= 0:
            raise ValueError("type_temperature must be positive.")
        self.type_schedule = type_schedule
        self.type_temperature = type_temperature
        self.type_seed = type_seed
        self.require_types_for_convergence = require_types_for_convergence
        self._type_generators: Dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._type_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.type_seed)
            self._type_generators[device] = generator
        return self._type_generators[device]

    def masking_rate(self, step: int) -> float:
        """
        The masking rate the types should be left at after ``step``.

        ``step`` counts down from n_steps to 1, so this falls from nearly 1 to
        exactly 0.
        """
        # exactly zero on the last step, so that unmask_probability is exactly
        # one there and no atom can be left masked: cos(pi/2) is 6e-17, not 0
        if step <= 1:
            return 0.0
        if step > self.n_steps:
            return 1.0
        progress = self.get_denoising_ratio(step)
        if self.type_schedule == "cosine":
            return math.cos(math.pi * progress / 2)
        return 1.0 - progress

    def unmask_probability(self, step: int) -> float:
        """
        The probability a still-masked atom is unmasked during ``step``.
        """
        before = self.masking_rate(step + 1)
        after = self.masking_rate(step)
        if before <= 0.0:
            return 1.0
        return min(max((before - after) / before, 0.0), 1.0)

    def _draw_types(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Draw a vocabulary index per atom from the predicted clean-type
        distribution. Rows that are not finite fall back to their argmax, since
        multinomial raises a device-side assert on them.
        """
        probabilities = torch.nan_to_num(
            probabilities, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)
        if self.type_temperature != 1.0:
            probabilities = probabilities ** (1.0 / self.type_temperature)
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0)
        total = probabilities.sum(dim=-1, keepdim=True)
        degenerate = ~(total > 1e-12)
        fallback = torch.zeros_like(probabilities)
        fallback.scatter_(-1, probabilities.argmax(dim=-1, keepdim=True), 1.0)
        safe = torch.where(degenerate, torch.ones_like(total), total)
        probabilities = torch.where(
            degenerate.expand_as(probabilities), fallback, probabilities / safe
        )
        return torch.multinomial(
            probabilities, 1, generator=self._generator(probabilities.device)
        ).squeeze(-1)

    def unmask_types(
        self,
        batch: Dict[str, torch.Tensor],
        probabilities: torch.Tensor,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Unmask a share of the still-masked atoms, drawing their element from the
        predicted distribution. Atoms that already have an element are untouched,
        and so are the atoms of converged molecules.
        """
        z = batch[properties.Z]
        masked = z == vocab.MASK_Z
        if not bool(masked.any()):
            return batch

        probability = self.unmask_probability(step)
        if probability <= 0.0:
            return batch
        draw = torch.rand(
            z.shape[0], device=z.device, generator=self._generator(z.device)
        )
        selected = masked & (draw < probability)

        converged_atoms = batch[keys.converged][batch[properties.idx_m]]
        selected = selected & ~converged_atoms
        if not bool(selected.any()):
            return batch

        indices = self._draw_types(probabilities[selected])
        new_z = z.clone()
        new_z[selected] = vocab.index_to_z(indices).to(z.dtype)
        batch[properties.Z] = new_z
        return self.remask_types(batch, probabilities, step)

    def remask_types(
        self,
        batch: Dict[str, torch.Tensor],
        probabilities: torch.Tensor,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Put some already decided atoms back to MASK, so they are decided again
        later with more context. See ``remask_prob`` and ``remask_mode``.

        A pure carry-over decode cannot revise an early mistake, and the wrong
        type stays in the context biasing everything decided after it. Resetting
        to MASK rather than to another element is deliberate: it removes the
        wrong evidence instead of replacing it with a fresh guess made from the
        same bad context.
        """
        if self.remask_prob <= 0:
            return batch
        rate = self.masking_rate(step)
        if rate <= 0:
            return batch

        z = batch[properties.Z]
        decided = z != vocab.MASK_Z
        converged_atoms = batch[keys.converged][batch[properties.idx_m]]
        decided = decided & ~converged_atoms
        if not bool(decided.any()):
            return batch

        weight = torch.ones_like(z, dtype=probabilities.dtype)
        if self.remask_mode == "confidence":
            held = vocab.z_to_index(z.clamp(min=0))
            p_held = probabilities.gather(-1, held.unsqueeze(-1)).squeeze(-1)
            weight = (1.0 - torch.nan_to_num(p_held, nan=1.0)).clamp(0.0, 1.0)

        draw = torch.rand(z.shape[0], device=z.device, generator=self._generator(z.device))
        selected = decided & (draw < self.remask_prob * rate * weight)
        if not bool(selected.any()):
            return batch
        new_z = z.clone()
        new_z[selected] = vocab.MASK_Z
        batch[properties.Z] = new_z
        return batch

    def denoise_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch = self.preprocess_step(batch, step, conditions=conditions, **kwargs)

        predictions = self.calculator.calculate(
            batch, forces=True, x0=True, estimate_sigma=True
        )
        x_next = predictions[keys.x_0]
        forces = predictions[keys.forces]
        sigma_curr = predictions[keys.sigma_curr]

        if self.f_max > 0:
            force_norms = torch.linalg.norm(forces, dim=1)
            batch = self.update_convergence(
                batch, force_norms, self.f_max, reduce="amax"
            )

        self._log(
            timestep=step,
            forces=torch.mean(torch.linalg.norm(forces, dim=1)),
            masked=float((batch[properties.Z] == vocab.MASK_Z).float().mean()),
        )
        batch = self.update_batch(
            batch,
            updates={properties.R: x_next, keys.sigma_curr: sigma_curr},
        )
        if PROBABILITIES_KEY in predictions:
            batch = self.unmask_types(batch, predictions[PROBABILITIES_KEY], step)
        batch = self.postprocess_step(batch, step, conditions=conditions, **kwargs)
        return batch

    def check_convergence(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        As the base class, except that a molecule with an undecided type is not
        converged however small its forces are. The NaN rule still applies, so a
        diverged molecule stops accumulating NFE either way.
        """
        converged_before = batch[keys.converged].clone()
        batch = super().check_convergence(batch, step=step, **kwargs)
        if not self.require_types_for_convergence:
            return batch

        masked_atoms = (batch[properties.Z] == vocab.MASK_Z).to(torch.int32)
        masked_molecules = torch.zeros_like(converged_before, dtype=torch.int32)
        masked_molecules.scatter_reduce_(
            0,
            batch[properties.idx_m],
            masked_atoms,
            reduce="amax",
            include_self=False,
        )
        # never un-converge a molecule that was already done, and keep the NaN
        # freezes the base class made
        nan_positions = torch.isnan(batch[properties.R]).any(dim=-1).to(torch.int32)
        nan_molecules = torch.zeros_like(converged_before, dtype=torch.int32)
        nan_molecules.scatter_reduce_(
            0,
            batch[properties.idx_m],
            nan_positions,
            reduce="amax",
            include_self=False,
        )
        hold = masked_molecules.bool() & ~nan_molecules.bool()
        batch[keys.converged] = torch.where(
            hold, converged_before, batch[keys.converged]
        )
        return batch
