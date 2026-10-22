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
    "remove_rigid_motion",
    "DirectDenoisingSampler",
    "ShapeDirectDenoisingSampler",
    "ScaffoldDirectDenoisingSampler",
    "ShapeScaffoldDirectDenoisingSampler",
]


def remove_rigid_motion(
    vectors: torch.Tensor,
    positions: torch.Tensor,
    idx_m: torch.Tensor,
) -> torch.Tensor:
    """
    Remove each molecule's rigid-body part from per-atom vectors (e.g. forces): the
    mean, and the best-fitting rotation omega x (r - centroid) in the least-squares
    sense, I omega = sum (r - centroid) x v.
    """
    n_molecules = int(idx_m.max()) + 1
    zeros = vectors.new_zeros(n_molecules, 3)
    counts = torch.bincount(idx_m, minlength=n_molecules).to(vectors.dtype).unsqueeze(-1)
    v = vectors - (zeros.index_add(0, idx_m, vectors) / counts)[idx_m]
    r = positions - (zeros.index_add(0, idx_m, positions) / counts)[idx_m]
    torque = zeros.index_add(0, idx_m, torch.cross(r, v, dim=1))
    eye = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    inertia = vectors.new_zeros(n_molecules, 3, 3).index_add(
        0, idx_m, (r * r).sum(1)[:, None, None] * eye - r.unsqueeze(-1) * r.unsqueeze(-2)
    )
    # pinv: a linear molecule has no rotation about its axis
    omega = (torch.linalg.pinv(inertia) @ torque.unsqueeze(-1)).squeeze(-1)
    return v - torch.cross(omega[idx_m], r, dim=1)


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

    Adaptive DD: with ``transport_steps`` the DDIM steps go down an EDM grid of noise
    levels instead of a fixed ratio, each molecule from its own sigma estimate, so
    the relative steps grow as the noise falls; ``transport_gamma`` makes them
    stochastic (DDPM-like, noise gamma * sigma). Below ``transport_sigma`` the update
    is x + ``refine_step`` * F/2, and ``refine_rigid_free`` removes each molecule's net
    force and torque first: the model leaves some in F, which would move and turn the
    molecules forever and keep ``f_max`` from being reached.
    """

    def __init__(
        self,
        time_aware_model: bool = False,
        stochastic_lambda: float = 1.0,
        f_max: float = 0.0,
        transport_sigma: Optional[float] = None,
        transport_ratio: float = 0.5,
        transport_steps: Optional[int] = None,
        transport_sigma_max: float = 30.0,
        transport_rho: float = 5.0,
        transport_gamma: float = 0.0,
        refine_step: float = 1.0,
        refine_rigid_free: bool = False,
        **kwargs,
    ):
        """
        Args:
            transport_sigma: noise level above which molecules take DDIM steps
                instead of jumps; None jumps from the start.
            transport_ratio: noise level after a DDIM step over the one before.
            transport_steps: if set, the DDIM steps go down transport_steps levels
                of an EDM grid from transport_sigma_max to transport_sigma (exponent
                transport_rho) instead of using transport_ratio: each molecule steps
                to the first level below its sigma estimate, at least one level per
                step, and its transport ends after the last level.
            transport_gamma: share of the DDPM noise in a DDIM step, from 0 (none)
                to 1: with r = sigma_next / sigma the step is
                x + (1 - sqrt(r^2 - g^2)) F/2 + g * sigma * z, g = gamma * r * sqrt(1 - r^2).
            refine_step: below transport_sigma the update is x + refine_step * F/2;
                1 jumps to x_0, more than 1 over-relaxes.
            refine_rigid_free: below transport_sigma, remove each molecule's net
                force and torque from F before the update and the f_max test.
        """
        if transport_steps is not None and transport_sigma is None:
            raise ValueError("transport_steps needs transport_sigma, the level transport ends at.")
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
        self.transport_steps = transport_steps
        self.transport_gamma = transport_gamma
        self.refine_step = refine_step
        self.refine_rigid_free = refine_rigid_free
        self.transport_levels = None
        if transport_steps is not None:
            # EDM grid, from transport_sigma_max (level 0) to transport_sigma (level transport_steps)
            u = torch.linspace(0, 1, transport_steps + 1, dtype=torch.float64)
            low, high = transport_sigma ** (1 / transport_rho), transport_sigma_max ** (1 / transport_rho)
            self.transport_levels = (high + u * (low - high)) ** transport_rho

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
        return self.in_transport(batch, sigma)

    def in_transport(self, batch: Dict[str, torch.Tensor], sigma: torch.Tensor) -> torch.Tensor:
        """
        Per-atom (n_atoms, 1) mask of the molecules whose sigma estimate is above
        transport_sigma and, on the grid, that have levels left.
        """
        mask = sigma > self.transport_sigma
        if self.transport_levels is not None:
            levels_left = batch[keys.transport_level] < self.transport_steps
            mask = mask & levels_left[batch[properties.idx_m]].unsqueeze(-1)
        return mask

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Start every molecule at the top level of the transport grid.
        """
        batch = super()._init_state(batch)
        if self.transport_levels is not None:
            batch[keys.transport_level] = torch.zeros_like(batch[properties.n_atoms], dtype=torch.long)
        return batch

    def transport_ratios(
        self,
        batch: Dict[str, torch.Tensor],
        sigma: torch.Tensor,
        in_transport: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-atom sigma_next / sigma of this step's DDIM steps, and on the grid the
        molecules in transport move to their next level.
        """
        if self.transport_levels is None:
            return torch.full_like(sigma, self.transport_ratio)
        idx_m = batch[properties.idx_m]
        n_molecules = batch[properties.n_atoms].shape[0]
        sigma_mol = torch.zeros(n_molecules, dtype=sigma.dtype, device=sigma.device).scatter_reduce(
            0, idx_m, sigma.squeeze(-1), reduce="amax", include_self=False
        )
        levels = self.transport_levels.to(device=sigma.device, dtype=sigma.dtype)
        first_below = (levels.unsqueeze(0) >= sigma_mol.unsqueeze(1)).sum(1)
        level = torch.maximum(batch[keys.transport_level] + 1, first_below).clamp(max=self.transport_steps)
        moving = torch.zeros(n_molecules, dtype=torch.bool, device=sigma.device)
        moving[idx_m[in_transport.squeeze(-1)]] = True
        batch[keys.transport_level] = torch.where(moving, level, batch[keys.transport_level])
        ratio = (levels[level] / sigma_mol).clamp(0, 1)
        return ratio[idx_m].unsqueeze(-1)

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

        # above transport_sigma, the DDIM step to a lower noise level instead of
        # the jump to x_0; below it the jump, scaled by refine_step
        if self.transport_sigma is not None:
            positions = batch[properties.R]
            in_transport = self.in_transport(batch, sigma_curr)
            if self.refine_rigid_free:
                rigid_free = remove_rigid_motion(forces, positions, batch[properties.idx_m])
                forces = torch.where(in_transport, forces, rigid_free)
            ratio = self.transport_ratios(batch, sigma_curr, in_transport)
            if self.transport_gamma > 0:
                gamma = self.transport_gamma * ratio * torch.sqrt(1 - ratio ** 2)
                transport_step = 1 - torch.sqrt(ratio ** 2 - gamma ** 2)
            else:
                transport_step = 1 - ratio
            step_size = torch.where(in_transport, transport_step, self.refine_step)
            x_next = positions + step_size * forces / 2
            if self.transport_gamma > 0:
                x_next = x_next + torch.where(in_transport, gamma * sigma_curr, 0) * torch.randn_like(x_next)

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
