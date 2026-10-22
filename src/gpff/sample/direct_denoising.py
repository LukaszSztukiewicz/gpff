"""
This module contains the molecule samplers for pseudo-PES based models.
"""

# python
import torch
from typing import Dict, Optional, Tuple
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

# sampler state between steps; keys starting with "_" stay out of the results
_X0_HISTORY = "_x0_history"  # (n_atoms, depth, 3) clean estimates of the last calls, newest first
_RESIDUAL_HISTORY = "_residual_history"  # (n_atoms, depth, 3) their residuals x_0 - x
_HISTORY_LENGTH = "_history_length"  # (n_atoms, 1) consecutive clean jumps in the history
_AFTER_TRANSPORT = "_after_transport"  # (n_atoms, 1) the last call was a transport step


def _molecule_sum(values: torch.Tensor, idx_m: torch.Tensor, n_molecules: int) -> torch.Tensor:
    return torch.zeros(
        (n_molecules, *values.shape[1:]), device=values.device, dtype=values.dtype
    ).index_add_(0, idx_m, values)


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

    Both phases can reuse the clean estimates of earlier calls. With
    ``transport_order=2`` a transport step after another one extrapolates the
    clean estimate along its change, x_0 + (x_0 - x_0_prev) / 2: the second-order
    multistep solver of the probability-flow ODE (DPM-Solver++ 2M) for steps of a
    constant ratio. It needs transport to stop near the data's own spread: far
    below it the clean estimate saturates and the extrapolation overshoots, while
    DDIM steps contract. The jumps are the fixed-point iteration x <- x_0(x), which
    converges linearly and slows down near its end; with ``anderson_depth`` > 0 a
    jump without noise after other such jumps combines them by Anderson
    acceleration (DIIS). It restarts when the residual grows and moves at most
    ``anderson_max`` times the plain jump beyond x_0. ``relax_steps`` takes the
    noise off the last steps, so stochastic sampling can end with such jumps.
    """

    def __init__(
        self,
        time_aware_model: bool = False,
        stochastic_lambda: float = 1.0,
        f_max: float = 0.0,
        transport_sigma: Optional[float] = None,
        transport_ratio: float = 0.5,
        transport_order: int = 1,
        anderson_depth: int = 0,
        anderson_max: float = 10.0,
        relax_steps: int = 1,
        **kwargs,
    ):
        """
        Args:
            transport_sigma: noise level above which molecules take DDIM steps
                instead of jumps; None jumps from the start.
            transport_ratio: noise level after a DDIM step over the one before.
            transport_order: 1 for DDIM transport steps, 2 for second-order
                multistep ones.
            anderson_depth: number of earlier jumps Anderson acceleration combines;
                0 takes plain jumps.
            anderson_max: largest move beyond x_0, relative to the plain jump.
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
        if transport_order not in (1, 2):
            raise ValueError(f"transport_order must be 1 or 2, got {transport_order}")
        if anderson_depth < 0:
            raise ValueError(f"anderson_depth must be non-negative, got {anderson_depth}")
        if not 1 <= relax_steps <= self.n_steps:
            raise ValueError(f"relax_steps must be between 1 and n_steps {self.n_steps}, got {relax_steps}")
        self.stochastic_lambda = stochastic_lambda
        self.f_max = f_max
        self.transport_sigma = transport_sigma
        self.transport_ratio = transport_ratio
        self.transport_order = transport_order
        self.anderson_depth = anderson_depth
        self.anderson_max = anderson_max
        self.relax_steps = relax_steps

    @property
    def _uses_history(self) -> bool:
        return self.transport_order == 2 or self.anderson_depth > 0

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Start with an empty history of earlier calls if the steps reuse them.
        """
        batch = super()._init_state(batch)
        if self._uses_history:
            positions = batch[properties.R]
            depth = max(self.anderson_depth, 1)
            batch[_X0_HISTORY] = positions.new_zeros(positions.shape[0], depth, 3)
            batch[_RESIDUAL_HISTORY] = positions.new_zeros(positions.shape[0], depth, 3)
            batch[_HISTORY_LENGTH] = torch.zeros(positions.shape[0], 1, dtype=torch.long, device=positions.device)
            batch[_AFTER_TRANSPORT] = torch.zeros(positions.shape[0], 1, dtype=torch.bool, device=positions.device)
        return batch

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
        history = {}
        if self._uses_history:
            x_next, history = self.extrapolate(batch, step, predictions, x_next)

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
        batch = self.postprocess_step(batch, step, conditions=conditions, **kwargs)

        return batch

    def extrapolate(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        predictions: Dict[str, torch.Tensor],
        x_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Replace x_next by the second-order transport step or the Anderson jump
        where the history allows it, and push this call into the history.
        """
        x_0 = predictions[keys.x_0]
        residual = predictions[keys.forces] / 2
        sigma_curr = predictions[keys.sigma_curr]
        x0_history = batch[_X0_HISTORY]
        residual_history = batch[_RESIDUAL_HISTORY]
        length = batch[_HISTORY_LENGTH]

        in_transport = (
            sigma_curr > self.transport_sigma
            if self.transport_sigma is not None
            else torch.zeros_like(batch[_AFTER_TRANSPORT])
        )
        # this call's input was perturbed by preprocess_step
        perturbed = torch.zeros_like(in_transport)
        if self.stochastic_lambda * self.get_noise_ratio(step) > 0:
            transport_before = self.transport_mask(batch)
            perturbed = ~transport_before if transport_before is not None else ~perturbed
        clean = ~in_transport & ~perturbed

        if self.transport_order == 2:
            # h_now / (2 h_prev) = 1/2: both steps shrink sigma by transport_ratio
            second_order = batch[properties.R] + (1 - self.transport_ratio) * (
                residual + (x_0 - x0_history[:, 0]) / 2
            )
            x_next = torch.where(in_transport & batch[_AFTER_TRANSPORT], second_order, x_next)

        if self.anderson_depth > 0:
            idx_m = batch[properties.idx_m]
            n_molecules = batch[properties.n_atoms].shape[0]
            grew = _molecule_sum(residual.square().sum(-1), idx_m, n_molecules) > _molecule_sum(
                residual_history[:, 0].square().sum(-1), idx_m, n_molecules
            )
            length = torch.where(grew[idx_m].unsqueeze(-1), 0, length)
            accelerated = self._anderson(x_0, residual, x0_history, residual_history, length, idx_m, n_molecules)
            x_next = torch.where(clean & (length > 0), accelerated, x_next)

        depth = x0_history.shape[1]
        history = {
            _X0_HISTORY: torch.cat([x_0.unsqueeze(1), x0_history[:, :-1]], dim=1),
            _RESIDUAL_HISTORY: torch.cat([residual.unsqueeze(1), residual_history[:, :-1]], dim=1),
            _HISTORY_LENGTH: torch.where(clean, (length + 1).clamp(max=depth), 0),
            _AFTER_TRANSPORT: in_transport,
        }
        return x_next, history

    def _anderson(
        self,
        x_0: torch.Tensor,
        residual: torch.Tensor,
        x0_history: torch.Tensor,
        residual_history: torch.Tensor,
        length: torch.Tensor,
        idx_m: torch.Tensor,
        n_molecules: int,
    ) -> torch.Tensor:
        """
        Anderson step per molecule: x_0 - sum_j gamma_j (x_0 - x_0_j), with gamma
        minimizing |residual - sum_j gamma_j (residual - residual_j)| over the
        first ``length`` history entries.
        """
        valid = (torch.arange(x0_history.shape[1], device=length.device) < length).unsqueeze(-1)
        d_residual = (residual.unsqueeze(1) - residual_history) * valid
        d_x0 = (x_0.unsqueeze(1) - x0_history) * valid

        gram = _molecule_sum(torch.einsum("aic,ajc->aij", d_residual, d_residual), idx_m, n_molecules).double()
        rhs = _molecule_sum(torch.einsum("aic,ac->ai", d_residual, residual), idx_m, n_molecules).double()
        # empty entries get gamma = 0, a small relative ridge keeps the rest well posed
        diagonal = gram.diagonal(dim1=1, dim2=2)
        ridge = 1e-6 * diagonal.amax(-1, keepdim=True) + (diagonal == 0).double()
        gamma = torch.linalg.solve(gram + torch.diag_embed(ridge), rhs).to(x_0.dtype)
        correction = -torch.einsum("ai,aic->ac", gamma[idx_m], d_x0)

        # at most anderson_max times the plain jump
        jump_norm = _molecule_sum(residual.square().sum(-1), idx_m, n_molecules).sqrt()
        correction_norm = _molecule_sum(correction.square().sum(-1), idx_m, n_molecules).sqrt()
        scale = (self.anderson_max * jump_norm / correction_norm.clamp_min(torch.finfo(x_0.dtype).tiny)).clamp(max=1)
        return x_0 + scale[idx_m].unsqueeze(-1) * correction


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
