"""
Noise schedules and sigma-transition estimators used by samplers.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


__all__ = [
    "NoiseSchedule",
    "EDMNoiseSchedule",
]


class NoiseSchedule(ABC):
    """
    Base class for sigma schedules used by samplers.
    """

    def __init__(
        self,
        n_steps: int,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ):
        if n_steps <= 1:
            raise ValueError(f"n_steps must be > 1, got {n_steps}.")
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.step_size = 1.0 / (self.n_steps - 1)

    @abstractmethod
    def sigma_next(
        self,
        sigma_target: torch.Tensor,
        step_size: torch.Tensor | float,
        is_last_step: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_time_tensor(
        self,
        idx_m: torch.Tensor,
        sigma: torch.Tensor | float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sigma_tensor = torch.as_tensor(sigma, device=device, dtype=dtype)
        if sigma_tensor.ndim == 0:
            return torch.ones(
                idx_m.shape[0],
                device=device,
                dtype=dtype,
            ).unsqueeze(-1) * (sigma_tensor / self.sigma_max)

        sigma_tensor = sigma_tensor.to(device=device, dtype=dtype)
        if sigma_tensor.ndim == 1:
            sigma_tensor = sigma_tensor.unsqueeze(-1)
        return sigma_tensor / self.sigma_max


class EDMNoiseSchedule(NoiseSchedule):
    """
    EDM (Elucidated Diffusion Models) noise schedule.
    Uses a power-law distribution to concentrate steps in the critical
    denoising regions.
    """

    def __init__(
        self,
        n_steps: int,
        sigma_min: float = 0.01,
        sigma_max: float = 30.0,
        rho: float = 5.0,
    ):
        super().__init__(
            n_steps=n_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        steps = torch.linspace(0, 1, self.n_steps)
        self.rho = rho

        self.inv_rho = 1.0 / rho
        sigmas_raw = (
            sigma_min**self.inv_rho
            + steps * (sigma_max**self.inv_rho - sigma_min**self.inv_rho)
        ) ** rho

        self.sigmas = torch.cat([torch.tensor([0.0]), sigmas_raw])

    def get_sigma(self, step: int) -> torch.Tensor:
        """Get the sigma for the given time."""
        return self.sigmas[step].item()

    def get_timestep(self, step: int) -> torch.Tensor:
        """Get the timestep for the given sigma."""
        return self.get_sigma(step) / self.sigmas[-1].item()

    def sigma_next(
        self,
        sigma_curr: torch.Tensor,
        step_size: torch.Tensor | float,
        is_last_step: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the next sigma value using the EDM recurrence rule.

        Uses the caller-provided ``step_size`` in EDM transformed space.
        """
        if is_last_step:
            return torch.zeros_like(sigma_curr)

        inv_rho = 1.0 / self.rho
        delta = step_size * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)

        sigma_next = (torch.clamp(sigma_curr, min=0.0) ** inv_rho + delta) ** self.rho
        sigma_next = torch.where(
            sigma_next < self.sigma_min*0.95,
            torch.zeros_like(sigma_next),
            sigma_next,
        )
        return sigma_next
    
    def get_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Get the timestep for the given sigma."""
        relative_t = (
            (sigma**self.inv_rho - self.sigma_min**self.inv_rho)
            / (self.sigma_max**self.inv_rho - self.sigma_min**self.inv_rho)
        )
        return relative_t

# ErrorEstimateSchedule was used for corrective x0 experiments and is no longer
# part of the active workflow. Keep the old implementation in git history.
