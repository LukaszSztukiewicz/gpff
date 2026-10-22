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
]


class DirectDenoisingSampler(Sampler):
    """
    Sampler for X0 like denoising of atomic positions in batch mode using a trained model.

    Each denoising step runs ``preprocess_step`` (stochastic perturbation of the
    positions), one model inference, and ``postprocess_step`` (constraint
    enforcement, a no-op here). Subclasses customize behavior by overriding the
    two hooks.

    The update x + F/2 jumps to the predicted x_0, which is the DDIM step from the
    current noise level all the way to sigma = 0. That is only accurate where the
    posterior over x_0 is narrow: from the sigma = 30 prior it lands on a posterior
    mean that is no molecule (atoms too close, the prior draw's shape). With
    ``transport_sigma`` set, molecules whose sigma estimate (std of F/2) is above it
    take the DDIM step to ``transport_ratio`` * sigma instead, x + (1 - transport_ratio)
    F/2 (sigma cancels), and are not perturbed, so the model's inputs stay noisy
    molecules until the jumps take over below ``transport_sigma``.
    """

    def __init__(
        self,
        time_aware_model: bool = False,
        stochastic_lambda: float = 1.0,
        f_max: float = 0.0,
        transport_sigma: Optional[float] = None,
        transport_ratio: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            transport_sigma: noise level above which molecules take DDIM steps
                instead of jumps; None jumps from the start.
            transport_ratio: noise level after a DDIM step over the one before.
        """
        if time_aware_model:
            warnings.warn(
                f"time_aware_model is not supported for {type(self).__name__} and will be ignored.",
                UserWarning,
                stacklevel=2
            )
        super().__init__(time_aware_model=False, **kwargs)
        self.stochastic_lambda = stochastic_lambda
        self.f_max = f_max
        self.transport_sigma = transport_sigma
        self.transport_ratio = transport_ratio

    def add_noise(self, x: torch.Tensor, step: int, in_transport: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add schedule-scaled stochastic noise to positions, except where in_transport.
        """
        noise_scale = self.stochastic_lambda * self.get_noise_ratio(step)
        noise = noise_scale * torch.randn_like(x)
        if in_transport is not None:
            noise = noise * ~in_transport
        return x + noise

    def transport_mask(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Per-atom (n_atoms, 1) mask of the molecules still in transport: their last
        sigma estimate is above transport_sigma, or there is none yet. None if
        transport is off.
        """
        if self.transport_sigma is None:
            return None
        sigma = batch.get(keys.sigma_curr)
        if sigma is None:
            return torch.ones_like(batch[properties.R][:, :1], dtype=torch.bool)
        return sigma > self.transport_sigma

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
        batch[properties.R] = self.add_noise(batch[properties.R], step, self.transport_mask(batch))
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

        # above transport_sigma, the DDIM step to transport_ratio * sigma instead
        # of the jump to x_0
        if self.transport_sigma is not None:
            step_size = torch.where(sigma_curr > self.transport_sigma, 1 - self.transport_ratio, 1.0)
            x_next = batch[properties.R] + step_size * forces / 2

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
        in_transport = self.transport_mask(batch)
        batch[properties.R][free_mask] = self.add_noise(
            batch[properties.R][free_mask], step, None if in_transport is None else in_transport[free_mask]
        )
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
        in_transport = self.transport_mask(batch)
        x_free = self.add_noise(
            batch[properties.R][free_mask], step, None if in_transport is None else in_transport[free_mask]
        )
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
