"""Walk-jump sampling on the pseudo-PES at a single noise level."""

# python
import math
import warnings
from typing import Dict, Optional

import torch

# gpff
from gpff import keys
from gpff.sample.base import Sampler

# schnetpack
from schnetpack import properties


__all__ = ["WalkJumpSampler"]


class WalkJumpSampler(Sampler):
    """
    Walk-jump sampling (Saremi and Hyvarinen, JMLR 2019; Frey et al., ICLR
    2024) on a GPFF pseudo-PES.

    One noise level throughout, instead of a schedule: the walk is an
    underdamped Langevin chain on the smoothed density p_sigma, whose score the
    pseudo-force gives as F / (2 sigma^2), and the final step is the jump, the
    denoiser's own estimate x_0 = x + F / 2. Sampling is then a single MCMC run
    at one scale rather than a transport across scales, which is what lets a
    bias potential or a Metropolis correction act on a stationary target.

    The walk integrates

        v <- v + dt u g(y) - dt gamma v + sqrt(2 gamma u dt) xi
        y <- y + dt v

    with g the score, u the inverse mass, gamma the friction and dt the step
    size. Since the curvature of log p_sigma is of order 1 / sigma^2, the
    integrator is stable for roughly dt < 2 sigma / sqrt(u); the constructor
    warns past that.

    Pair with a prior at the same scale (prior=gaussian prior.sigma=<sigma>):
    the chain mixes within p_sigma, so a start far off that typical set is
    paid for in burn-in steps.

    The model enters through its score alone, so a direct force head runs too.
    It is the conservative surface (experiment=gpff_conservative) that makes
    this a Langevin chain on an energy rather than on a vector field that need
    not have a stationary density at all.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        step_size: float = 0.2,
        friction: float = 1.0,
        inverse_mass: float = 1.0,
        time_aware_model: bool = False,
        **kwargs,
    ):
        """
        Args:
            sigma: the single noise level the chain lives at.
            step_size: the Langevin step dt.
            friction: the friction gamma; the velocity relaxes over 1 / gamma.
            inverse_mass: the inverse mass u, which scales the score's pull and
                the thermal velocity alike.
            time_aware_model: unsupported, there is no time to be aware of.
            **kwargs: forwarded to :class:`gpff.sample.base.Sampler`.
        """
        if time_aware_model:
            warnings.warn(
                f"time_aware_model is not supported for {type(self).__name__} and "
                f"will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(time_aware_model=False, **kwargs)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if friction < 0 or inverse_mass <= 0:
            raise ValueError(
                f"friction must be non-negative and inverse_mass positive, got "
                f"{friction} and {inverse_mass}."
            )
        stability_limit = 2 * sigma / math.sqrt(inverse_mass)
        if step_size >= stability_limit:
            warnings.warn(
                f"step_size {step_size} is at or past the stability limit "
                f"{stability_limit:.3g} = 2 * sigma / sqrt(inverse_mass) of the "
                f"Langevin integrator; the walk is likely to diverge.",
                UserWarning,
                stacklevel=2,
            )
        self.sigma = sigma
        self.step_size = step_size
        self.friction = friction
        self.inverse_mass = inverse_mass

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Draw the initial velocities from their stationary distribution.
        """
        batch = super()._init_state(batch)
        batch[keys.velocity] = math.sqrt(self.inverse_mass) * torch.randn_like(
            batch[properties.R]
        )
        return batch

    def walk_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        One underdamped Langevin step on p_sigma.
        """
        predictions = self.calculator.calculate(batch, score=True, sigma_t=self.sigma)
        score = predictions[keys.score]

        velocity = batch[keys.velocity]
        velocity = (
            velocity
            + self.step_size * self.inverse_mass * score
            - self.step_size * self.friction * velocity
            + math.sqrt(2 * self.friction * self.inverse_mass * self.step_size)
            * torch.randn_like(velocity)
        )

        self._log(score=torch.mean(torch.linalg.norm(score, dim=1)))
        return self.update_batch(
            batch,
            updates={
                properties.R: batch[properties.R] + self.step_size * velocity,
                keys.velocity: velocity,
            },
        )

    def jump_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        The jump: replace the noisy chain state by the denoiser's estimate of
        the clean geometry.
        """
        predictions = self.calculator.calculate(batch, x0=True, sigma_t=self.sigma)
        return self.update_batch(batch, updates={properties.R: predictions[keys.x_0]})

    def denoise_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Walk while steps remain, and jump on the last one.

        The loop counts down, so ``step == 1`` is the jump and the n_steps - 1
        steps before it are the chain.
        """
        if step > 1:
            return self.walk_step(batch)
        return self.jump_step(batch)
