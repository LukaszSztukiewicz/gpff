"""
This module contains the molecule samplers for pseudo-PES based models.
"""

# python
import torch
from typing import Dict, Optional, Tuple
import warnings

# gpff
from gpff.sample.base import Sampler
from gpff.sample.jump_optimizers import JumpOptimizer
from gpff import keys
from gpff.utils.alignment import (
    batch_apply_covariances,
    batch_match_principal_variances,
    remove_rigid_motion,
)

# schnetpack
from schnetpack import properties


__all__ = [
    "DirectDenoisingSampler",
    "ShapeDirectDenoisingSampler",
    "ScaffoldDirectDenoisingSampler",
    "ShapeScaffoldDirectDenoisingSampler",
]

# sampler state between steps; keys starting with "_" stay out of the results
_LEVEL = "_transport_level"  # (n_molecules,) levels of the transport grid reached
_X0_PREV = "_x0_prev"  # (n_atoms, 3) clean estimate of the last call
_H_PREV = "_transport_h_prev"  # (n_atoms, 1) log(sigma / sigma_next) of the last call's transport step, 0 if none
_AFTER_CLEAN_JUMP = "_after_clean_jump"  # (n_atoms, 1) the last call was a jump without noise
_JUMP_STATE = "_jump_"  # prefix of the jump optimizer's state


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
    mean that is no molecule (atoms too close, the prior draw's shape). The sampler
    therefore has two phases, per molecule and by its sigma estimate (std of F/2):

    Transport, above ``transport_sigma``: DDIM steps to sigma_next = ratio * sigma,
    x + (1 - ratio) F/2 (sigma cancels, so the time-free model needs no time input).
    The molecules are not perturbed, so the model's inputs stay noisy molecules. The
    ratio is ``transport_ratio``, or with ``transport_steps`` that of the next level
    of an EDM grid: each molecule steps to the first level below its own sigma
    estimate, at least one level per step, so the relative steps are small while the
    shape forms and grow as the noise falls. ``transport_gamma`` makes the steps
    stochastic (DDPM-like) and ``transport_order=2`` second-order (multistep
    DPM-Solver++ 2M, which reuses the clean estimate of the last call; it needs
    transport to stop near the data's own spread, far below it the clean estimate
    saturates and the extrapolation overshoots).

    Refinement, below it (and everywhere without ``transport_sigma``): jumps
    x + ``refine_step`` * F/2, perturbed by the noise schedule. ``refine_rigid_free``
    removes each molecule's net force and torque from F first: the model leaves some
    in F, which would move and turn the molecules forever and keep ``f_max`` from
    being reached. The jumps without noise are the fixed-point iteration x <- x_0(x),
    which converges linearly and slows down near its end; a ``jump_optimizer`` (see
    :mod:`gpff.sample.jump_optimizers`, e.g. MDMin) takes them over, and
    ``relax_steps`` takes the noise off the last steps, so stochastic sampling can end
    with such jumps.
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
        transport_order: int = 1,
        refine_step: float = 1.0,
        refine_rigid_free: bool = False,
        jump_optimizer: Optional[JumpOptimizer] = None,
        relax_steps: int = 1,
        **kwargs,
    ):
        """
        Args:
            transport_sigma: noise level above which molecules take DDIM steps
                instead of jumps; None jumps from the start.
            transport_ratio: noise level after a DDIM step over the one before.
            transport_steps: if set, the DDIM steps go down transport_steps levels
                of an EDM grid from transport_sigma_max to transport_sigma (exponent
                transport_rho) instead of using transport_ratio; a molecule's
                transport ends after the last level. Fewer than n_steps.
            transport_gamma: share of the DDPM noise in a DDIM step, from 0 (none)
                to 1: with r = sigma_next / sigma the step is
                x + (1 - sqrt(r^2 - g^2)) F/2 + g * sigma * z, g = gamma * r * sqrt(1 - r^2).
            transport_order: 1 for DDIM steps, 2 for second-order multistep ones
                (deterministic: not with transport_gamma).
            refine_step: below transport_sigma the plain jump is x + refine_step * F/2;
                1 jumps to x_0, more than 1 over-relaxes.
            refine_rigid_free: below transport_sigma, remove each molecule's net
                force and torque from F before the update and the f_max test.
            jump_optimizer: moves the jumps without noise instead of the plain
                jump; None takes plain jumps.
            relax_steps: number of final steps without noise; 1 is the last step
                only, as in the plain schedule.
        """
        if time_aware_model:
            warnings.warn(
                f"time_aware_model is not supported for {type(self).__name__} and will be ignored.",
                UserWarning,
                stacklevel=2
            )
        super().__init__(time_aware_model=False, **kwargs)
        if transport_sigma is None and (transport_steps is not None or transport_gamma > 0 or transport_order != 1):
            raise ValueError("transport_steps, transport_gamma and transport_order need transport_sigma.")
        if transport_steps is not None and not 1 <= transport_steps < self.n_steps:
            raise ValueError(f"transport_steps must be between 1 and n_steps {self.n_steps} - 1, got {transport_steps}")
        if not 0 <= transport_gamma <= 1:
            raise ValueError(f"transport_gamma must be between 0 and 1, got {transport_gamma}")
        if transport_order not in (1, 2):
            raise ValueError(f"transport_order must be 1 or 2, got {transport_order}")
        if transport_order == 2 and transport_gamma > 0:
            raise ValueError("transport_order 2 steps are deterministic, transport_gamma must be 0.")
        if not 1 <= relax_steps <= self.n_steps:
            raise ValueError(f"relax_steps must be between 1 and n_steps {self.n_steps}, got {relax_steps}")
        self.stochastic_lambda = stochastic_lambda
        self.f_max = f_max
        self.transport_sigma = transport_sigma
        self.transport_ratio = transport_ratio
        self.transport_steps = transport_steps
        self.transport_gamma = transport_gamma
        self.transport_order = transport_order
        self.refine_step = refine_step
        self.refine_rigid_free = refine_rigid_free
        self.jump_optimizer = jump_optimizer
        self.relax_steps = relax_steps
        self.transport_levels = None
        if transport_steps is not None:
            # EDM grid, from transport_sigma_max (level 0) to transport_sigma (level transport_steps)
            u = torch.linspace(0, 1, transport_steps + 1, dtype=torch.float64)
            low, high = transport_sigma ** (1 / transport_rho), transport_sigma_max ** (1 / transport_rho)
            self.transport_levels = (high + u * (low - high)) ** transport_rho

    def get_noise_ratio(self, step: int) -> float:
        """
        The noise schedule of a sampler with relax_steps - 1 fewer steps, then
        relax_steps - 1 more steps without noise.
        """
        n_noisy = self.n_steps - self.relax_steps + 1
        step_noisy = step - self.relax_steps + 1
        if step_noisy < 1:
            return 0.0
        return 1 - (n_noisy - step_noisy + 1) / n_noisy

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
            levels_left = batch[_LEVEL] < self.transport_steps
            mask = mask & levels_left[batch[properties.idx_m]].unsqueeze(-1)
        return mask

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Start every molecule at the top level of the transport grid, with an empty
        history of earlier calls if the steps reuse them.
        """
        batch = super()._init_state(batch)
        positions = batch[properties.R]
        if self.transport_levels is not None:
            batch[_LEVEL] = torch.zeros_like(batch[properties.n_atoms], dtype=torch.long)
        if self.transport_order == 2:
            batch[_X0_PREV] = torch.zeros_like(positions)
            batch[_H_PREV] = positions.new_zeros(positions.shape[0], 1)
        if self.jump_optimizer is not None:
            batch[_AFTER_CLEAN_JUMP] = torch.zeros(positions.shape[0], 1, dtype=torch.bool, device=positions.device)
            for name, value in self.jump_optimizer.init_state(positions).items():
                batch[_JUMP_STATE + name] = value
        return batch

    def transport_ratios(
        self,
        batch: Dict[str, torch.Tensor],
        sigma: torch.Tensor,
        in_transport: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-atom (n_atoms, 1) sigma_next / sigma of this step's DDIM steps. On the
        grid the molecules in transport move to the level their step ends at.
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
        level = torch.maximum(batch[_LEVEL] + 1, first_below).clamp(max=self.transport_steps)
        moving = torch.zeros(n_molecules, dtype=torch.int32, device=sigma.device).scatter_reduce(
            0, idx_m, in_transport.squeeze(-1).to(torch.int32), reduce="amax", include_self=False
        )
        batch[_LEVEL] = torch.where(moving.bool(), level, batch[_LEVEL])
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

        denoise_step passes the calculator's results of this step as ``predictions``.
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
        x_curr = batch[properties.R].clone()
        batch = self.preprocess_step(batch, step, conditions=conditions, **kwargs)
        # molecules converged in an earlier step stay where they are; the
        # perturbation would otherwise be kept as their frozen position
        converged_atoms = batch[keys.converged][batch[properties.idx_m]].unsqueeze(-1)
        batch[properties.R] = torch.where(converged_atoms, x_curr, batch[properties.R])

        # get predictions
        predictions = self.calculator.calculate(batch, forces=True, x0=True, estimate_sigma=True)
        sigma_curr = predictions[keys.sigma_curr]
        x_next, forces, history = self.next_positions(batch, step, x_curr, predictions)

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
                **history,
            },
        )
        batch = self.postprocess_step(batch, step, conditions=conditions, predictions=predictions, **kwargs)

        return batch

    def next_positions(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        x_prev: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        The positions after this step, from the model input batch[R]: the DDIM step
        for the molecules in transport, the jump for the others.

        Args:
            x_prev: the positions before preprocess_step perturbed them.
            predictions: the calculator's x_0, forces and sigma estimate.

        Returns:
            x_next: (n_atoms, 3) the next positions.
            forces: (n_atoms, 3) the forces the f_max test reads, those of the
                molecules past transport without their net force and torque if
                refine_rigid_free.
            history: the atom-wise state this call hands to the next one.
        """
        x = batch[properties.R]
        idx_m = batch[properties.idx_m]
        n_molecules = batch[properties.n_atoms].shape[0]
        x_0, forces, sigma = predictions[keys.x_0], predictions[keys.forces], predictions[keys.sigma_curr]
        history = {}

        in_transport = torch.zeros_like(sigma, dtype=torch.bool)
        if self.transport_sigma is not None:
            in_transport = self.in_transport(batch, sigma)

        # refinement: the (over-relaxed) jump to x_0
        if self.refine_rigid_free:
            forces = torch.where(in_transport, forces, remove_rigid_motion(forces, x, idx_m))
        residual = forces / 2
        x_next = x + self.refine_step * residual

        if self.transport_sigma is not None:
            ratio = self.transport_ratios(batch, sigma, in_transport)
            step_size = 1 - ratio
            if self.transport_gamma > 0:
                # g = gamma r sqrt(1 - r^2), and sqrt(r^2 - g^2) written without the cancellation
                gamma = self.transport_gamma * ratio * torch.sqrt(1 - ratio**2)
                step_size = 1 - ratio * torch.sqrt(1 - self.transport_gamma**2 * (1 - ratio**2))
            direction = residual
            if self.transport_order == 2:
                # DPM-Solver++ 2M: the clean estimate extrapolated along its change over the
                # last transport step, x_0 + (x_0 - x_0_prev) / (2 r) with r = h_prev / h the
                # ratio of the steps' log noise ratios (1/2 for steps of a constant ratio)
                h = -torch.log(ratio)
                h_prev = batch[_H_PREV]
                weight = torch.where(h_prev > 0, h / (2 * h_prev.clamp_min(torch.finfo(h.dtype).tiny)), 0.0)
                direction = residual + weight * (x_0 - batch[_X0_PREV])
                history = {_X0_PREV: x_0, _H_PREV: torch.where(in_transport, h, 0.0)}
            x_transport = x + step_size * direction
            if self.transport_gamma > 0:
                x_transport = x_transport + gamma * sigma * torch.randn_like(x)
            x_next = torch.where(in_transport, x_transport, x_next)

        if self.jump_optimizer is not None:
            # the optimizers need successive jumps: not after transport, and the input
            # must be the last step's output, which the noise (or a shape) changes
            changed = (x != x_prev).any(dim=-1).to(torch.int32)
            perturbed = torch.zeros(n_molecules, dtype=torch.int32, device=x.device).scatter_reduce(
                0, idx_m, changed, reduce="amax", include_self=False
            )
            clean = ~in_transport & ~perturbed.bool()[idx_m].unsqueeze(-1)
            state = {key[len(_JUMP_STATE):]: value for key, value in batch.items() if key.startswith(_JUMP_STATE)}
            x_jump, state = self.jump_optimizer.step(
                x,
                residual,
                state,
                fresh=~batch[_AFTER_CLEAN_JUMP],
                idx_m=idx_m,
                n_molecules=n_molecules,
                last=step == 1,
            )
            x_next = torch.where(clean, x_jump, x_next)
            history[_AFTER_CLEAN_JUMP] = clean
            history.update({_JUMP_STATE + name: value for name, value in state.items()})
        return x_next, forces, history


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
