"""
This module contains ancestral samplers for the VE diffusion model.
"""

# python
import torch
from typing import Dict

# gpff
from gpff.sample.base import DiffusionSampler
from gpff import keys

# schnetpack
from schnetpack import properties


__all__ = [
    "AncestralSampler",
    "AdaptiveAncestralSampler",
]


class AncestralSampler(DiffusionSampler):
    """
    Ancestral sampler for denoising atomic positions in batch mode.
    """

    def denoise_step(self, batch: Dict[str, torch.Tensor], step: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        One step of the denoising process for the ancestral sampler.

        Args:
            batch: the input data.
            step: the current time step.

        Returns:
            the updated batch data.
        """
        # get properties

        sigma_curr = self.noise_schedule.get_sigma(step)
        sigma_next = self.noise_schedule.get_sigma(step - 1)
        x_curr = batch[properties.R]

        # get predictions
        if self.time_aware_model:
            batch[keys.t] = self._ones_like_batch_dim(batch[properties.idx_m]) * self.noise_schedule.get_timestep(step)
        predictions = self.calculator.calculate(batch, score=True, sigma_t=sigma_curr)
        score = predictions[keys.score]

        self._log(
                step=step,
                sigma=sigma_curr,
                sigma_next=sigma_next,
                t=self.noise_schedule.get_timestep(step),
            )
        
        # update atomic positions
        x_next = x_curr + score * (sigma_curr**2 - sigma_next**2)
        
        # add noise
        x_next = x_next + torch.randn_like(x_next) * (sigma_next**2 * (sigma_curr**2 - sigma_next**2) / sigma_curr**2) ** 0.5
        
        batch[properties.R] = x_next

        return batch


class AdaptiveAncestralSampler(DiffusionSampler):
    """
    Adaptive ancestral sampler for denoising atomic positions in batch mode.
    """
    def __init__(
        self,
        n_steps_target: int = 10,
        estimate_sigma: bool = True,
        **kwargs,
    ):
        super().__init__(n_steps_target=n_steps_target, **kwargs)
        self.estimate_sigma = estimate_sigma

    def denoise_step(self, batch: Dict[str, torch.Tensor], step: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        One step of the denoising process for the adaptive ancestral sampler.

        Args:
            batch: the input data.
            step: the current time step.

        Returns:
            the updated batch data.
        """
        # get properties
        x_curr = batch[properties.R]
        sigma_curr = batch[keys.sigma_curr]

        # get predictions
        predictions = self.calculator.calculate(batch, x0=True, estimate_sigma=True)
        x_0 = predictions[keys.x_0]
        if self.estimate_sigma:
            sigma_curr = predictions[keys.sigma_curr]
        
        # get sigma_next
        step_size = self.adjust_step_size(
            batch=batch,
            sigma_curr=sigma_curr,
            step=step,
        )
        sigma_next = self.noise_schedule.sigma_next(
            sigma_curr,
            step_size=step_size,
            is_last_step=step == 1,
        )
        sigma_next = self.clamp_sigma_next(sigma_next=sigma_next, step=step)
        
        # update atomic positions
        x_next = (sigma_next**2 / sigma_curr**2) * x_curr + (sigma_curr**2 - sigma_next**2) / sigma_curr**2 * x_0
        
        # add noise
        x_next = x_next + torch.randn_like(x_next) * (sigma_next**2 * (sigma_curr**2 - sigma_next**2) / sigma_curr**2) ** 0.5
        
        # update batch
        batch = self.update_batch(
            batch,
            updates={
                properties.R: x_next,
                keys.sigma_curr: sigma_next,
                keys.sigma_last: sigma_curr,
            },
        )
        # log
        self._log(
                step=step,
                sigma=sigma_curr.mean(),
                sigma_next=sigma_next.mean(),
                step_size=step_size.mean(),
            )

        return batch

    def check_convergence(self, batch: Dict[str, torch.Tensor], step: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Converge molecules whose sigma dropped to (near) zero, on top of the
        base criteria.
        """
        batch = self.update_convergence(batch, batch[keys.sigma_curr], 1e-5)
        return super().check_convergence(batch, step, **kwargs)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gpff.sample.schedules import EDMNoiseSchedule

    # Configuration
    rho = 5.0
    n_steps = 20
    sigma_max = 30.0
    sigma_min = 0.1

    # Starting point
    edm_noise_schedule = EDMNoiseSchedule(n_steps, sigma_min, sigma_max, rho=rho)
    # iterative sigma_sampling
    sigmas = [sigma_max]
    sigma_curr = sigma_max
    for i in range(n_steps):
        # 2. The recurrence rule:
        # Move to linear space, add delta, move back to power space
        sigma_curr = edm_noise_schedule.sigma_next(
            torch.tensor([sigma_curr]),
            step_size=edm_noise_schedule.step_size,
        )
        sigmas.append(sigma_curr.item())
    print(sigmas)
    plt.plot(edm_noise_schedule.sigmas.flip(0), label="sigmas")
    plt.plot(sigmas, label="sigmas iterative")
    plt.legend()
    plt.show()
