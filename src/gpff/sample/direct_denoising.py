"""
This module contains the molecule samplers for pseudo-PES based models.
"""

# python
import torch
from typing import Dict, Optional
import warnings

# gpff
from gpff.sample.base import Sampler
from gpff import keys
from gpff.utils.alignment import (
    batch_apply_covariances,
    batch_match_principal_variances,
)

# schnetpack
from schnetpack import properties


__all__ = [
    "DirectDenoisingSampler",
    "ShapeDirectDenoisingSampler",
    "ScaffoldDirectDenoisingSampler",
    "ShapeScaffoldDirectDenoisingSampler",
    "DesignFreedomDirectDenoisingSampler",
]


class DirectDenoisingSampler(Sampler):
    """
    Sampler for X0 like denoising of atomic positions in batch mode using a trained model.

    Each denoising step runs ``preprocess_step`` (stochastic perturbation of the
    positions), one model inference, and ``postprocess_step`` (constraint
    enforcement, a no-op here). Subclasses customize behavior by overriding the
    two hooks.
    """

    def __init__(
        self,
        time_aware_model: bool = False,
        stochastic_lambda: float = 1.0,
        f_max: float = 0.0,
        **kwargs,
    ):
        if time_aware_model:
            warnings.warn(
                f"time_aware_model is not supported for {type(self).__name__} and will be ignored.",
                UserWarning,
                stacklevel=2
            )
        super().__init__(time_aware_model=False, **kwargs)
        self.stochastic_lambda = stochastic_lambda
        self.f_max = f_max

    def add_noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Add schedule-scaled stochastic noise to positions.
        """
        noise_scale = self.stochastic_lambda * self.get_noise_ratio(step)
        return x + noise_scale * torch.randn_like(x)

    def preprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb the positions stochastically before the model call.
        """
        batch[properties.R] = self.add_noise(batch[properties.R], step)
        return batch

    def postprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Enforce constraints on the updated positions. No-op by default.
        """
        return batch

    def denoise_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Run consistency sampling in batch mode using a trained model.

        Args:
            batch: A schnetpack batch dict containing batched molecular data.
            conditions: Optional sampler conditioning, forwarded to the
                preprocess and postprocess hooks.

        Returns:
            batch: A dictionary with updated atomic positions after consistency sampling.
        """
        batch = self.preprocess_step(batch, step, conditions=conditions, **kwargs)

        # get predictions
        predictions = self.calculator.calculate(batch, forces=True, x0=True, estimate_sigma=True)
        x_next = predictions[keys.x_0]
        forces = predictions[keys.forces]
        sigma_curr = predictions[keys.sigma_curr]

        # mid-step on purpose: molecules converging now must not take this
        # step's position update
        if self.f_max > 0:
            force_norms = torch.linalg.norm(forces, dim=1)
            batch = self.update_convergence(
                batch,
                force_norms,
                self.f_max,
                reduce="amax",
            )

        # log
        self._log(
                timestep=step,
                forces=torch.mean(torch.linalg.norm(forces, dim=1)),
            )
        batch = self.update_batch(
            batch,
            updates={
                properties.R: x_next,
                keys.sigma_curr: sigma_curr,
            },
        )
        batch = self.postprocess_step(batch, step, conditions=conditions, **kwargs)

        return batch


class ShapeDirectDenoisingSampler(DirectDenoisingSampler):
    """
    Direct denoising sampler that enforces a target shape in the preprocess
    step, on top of the stochastic perturbation. Only the principal variance
    ratios of the target are enforced; each molecule keeps its own orientation,
    centroid, and total spread.

    Conditions:
        - keys.target_variances: (n_molecules, 3) principal variances,
          descending, ordered like idx_m.unique().
    """

    def __init__(
        self,
        prior_shape_denoising_power: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prior_shape_denoising_power = prior_shape_denoising_power

    def preprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb the positions stochastically, then enforce the target shape.
        """
        batch = super().preprocess_step(batch, step, conditions=conditions, **kwargs)
        batch[properties.R] = self.apply_shape(
            batch[properties.R],
            conditions[keys.target_variances],
            batch[properties.idx_m],
            step,
        )
        return batch

    def apply_shape(self, x_curr, target_variances, idx_m, step: int) -> torch.Tensor:
        """
        Rescale the current positions along their principal axes so their
        variance ratios match the target, softly blended over the schedule.
        """
        reshaped = batch_match_principal_variances(x_curr, target_variances, idx_m, preserve_spread=True)
        denoising_ratio = self.get_denoising_ratio(step) ** self.prior_shape_denoising_power
        return (1 - denoising_ratio) * reshaped + denoising_ratio * x_curr


class ScaffoldDirectDenoisingSampler(DirectDenoisingSampler):
    """
    Sampler for scaffold-based denoising. The preprocess step applies stochastic
    perturbation to the free (non-scaffold) atoms only; the postprocess step
    restores scaffold atoms to their prior positions, so the scaffold stays
    exactly where the prior put it while the free atoms denoise around it.

    Nothing constrains the shape of the free atoms; see
    :class:`ShapeScaffoldDirectDenoisingSampler` for that.

    Conditions:
        - keys.scaffold_mask: (n_atoms,) boolean mask, True for scaffold atoms.
    """

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Snapshot the prior positions so scaffold atoms can be restored each step.
        """
        batch = super()._init_state(batch)
        batch[keys.x_prior] = batch[properties.R].clone()
        return batch

    def preprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb the free atoms; scaffold atoms are untouched.
        """
        free_mask = ~conditions[keys.scaffold_mask]
        batch[properties.R][free_mask] = self.add_noise(batch[properties.R][free_mask], step)
        return batch

    def postprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Restore scaffold atom positions to their prior values.
        """
        scaffold_mask = conditions[keys.scaffold_mask]
        batch[properties.R][scaffold_mask] = batch[keys.x_prior][scaffold_mask]
        return batch


class ShapeScaffoldDirectDenoisingSampler(ScaffoldDirectDenoisingSampler):
    """
    Shape-enforcing sampler for scaffold-based denoising. The preprocess step
    applies stochastic perturbation and shape enforcement to the free
    (non-scaffold) atoms only, using a full target covariance and center since
    the free atoms' shape can be oriented relative to the fixed scaffold; the
    postprocess step restores scaffold atoms to their prior positions.

    Conditions:
        - keys.scaffold_mask: (n_atoms,) boolean mask, True for scaffold atoms.
        - keys.free_covariance: (n_molecules, 3, 3) target covariance of the
          free atoms, ordered like idx_m.unique().
        - keys.free_center: (n_molecules, 3) target centroid of the free atoms,
          ordered like idx_m.unique().
    """

    def __init__(
        self,
        prior_shape_denoising_power: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prior_shape_denoising_power = prior_shape_denoising_power

    def preprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb and shape-enforce the free atoms; scaffold atoms are untouched.
        """
        free_mask = ~conditions[keys.scaffold_mask]
        idx_m = batch[properties.idx_m]
        x_free = self.add_noise(batch[properties.R][free_mask], step)
        batch[properties.R][free_mask] = self.apply_shape(
            x_free,
            idx_m[free_mask],
            conditions[keys.free_covariance],
            conditions[keys.free_center],
            idx_m.unique(),
            step,
        )
        return batch

    def apply_shape(self, x_free, idx_m_free, covariances, centers, molecule_ids, step: int) -> torch.Tensor:
        """
        Transform the free atoms of each molecule towards the target covariance
        and center, softly blended over the schedule.
        """
        reshaped = batch_apply_covariances(
            x_free,
            idx_m_free,
            covariances,
            centers=centers,
            molecule_ids=molecule_ids,
        )
        denoising_ratio = self.get_denoising_ratio(step) ** self.prior_shape_denoising_power
        return (1 - denoising_ratio) * reshaped + denoising_ratio * x_free


class DesignFreedomDirectDenoisingSampler(DirectDenoisingSampler):
    """
    Direct denoising under a per-atom design-freedom field.

    Each atom's stochastic perturbation is capped by its own sigma: with the
    schedule at level s(step), atom i is perturbed by min(s(step), sigma_i), so
    a low-freedom atom settles while the free ones are still being kicked, and
    atoms of zero freedom are restored to the prior's geometry every step. A
    field taking only zero and a value above the schedule's first level
    reproduces :class:`ScaffoldDirectDenoisingSampler`; what the continuous
    field adds is every setting between a frozen atom and a free one.

    Capping instead of drawing each atom's level freely is what keeps the noise
    configuration near the ones training covered, since every level stays tied
    to one global schedule (Equivariant Asynchronous Diffusion,
    arXiv:2603.10093). Train with
    ``+pseudo_pes.geometry_sampler.sigma_spread`` above zero so the
    model has seen a field of levels at all.

    Conditions:
        - keys.sigma_field: (n_atoms, 1) starting sigma of each atom.
    """

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Snapshot the prior geometry, which the frozen atoms are held at.
        """
        batch = super()._init_state(batch)
        batch[keys.x_prior] = batch[properties.R].clone()
        return batch

    def preprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb every atom by the lesser of the schedule's level and its own.
        """
        noise_scale = self.stochastic_lambda * self.get_noise_ratio(step)
        scale = conditions[keys.sigma_field].clamp(max=noise_scale)
        positions = batch[properties.R]
        batch[properties.R] = positions + scale * torch.randn_like(positions)
        return batch

    def postprocess_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Restore the atoms of zero freedom to the geometry the prior left.
        """
        frozen = (conditions[keys.sigma_field] <= 0).squeeze(-1)
        batch[properties.R][frozen] = batch[keys.x_prior][frozen]
        return batch
