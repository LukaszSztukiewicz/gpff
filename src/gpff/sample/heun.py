"""
This module contains the sampler for Heun's method for denoising of atomic positions in batch mode using a trained model.
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
    "HeunSampler",
    "AdaptiveHeunSampler",
    "AdaptiveStochasticHeunSampler",
    "StochasticHeunSampler",
]


class HeunSampler(DiffusionSampler):
    """
    Sampler for Heun's method for denoising of atomic positions in batch mode using a trained model.
    """

    def denoise_step(self, batch: Dict[str, torch.Tensor], step: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Run consistency sampling in batch mode using a trained model.

        Args:
            batch: A schnetpack batch dict containing batched molecular data.

        Returns:
            batch: A dictionary with updated atomic positions after consistency sampling.
        """
        # prepare calculatoins
        sigma_curr = self.noise_schedule.get_sigma(step)
        sigma_next = self.noise_schedule.get_sigma(step - 1)
        x_curr = batch[properties.R]
        h_curr = batch.get(keys.h)
        
        # step 0: calculate x_0_predicted (=x_0)
        if self.time_aware_model:
            batch[keys.t] = self.noise_schedule.get_time_tensor(
                batch[properties.idx_m],
                sigma_curr,
                device=self.device,
                dtype=self.dtype,
            )
        predictions = self.calculator.calculate(batch, x0=True, sigma_t=sigma_curr)
        x_pred0 = predictions[keys.x_0]

        # step 1:calculate d_1 and x_next
        slope_1 = 1 / sigma_curr * (x_curr - x_pred0)
        x_next = x_curr + (sigma_next - sigma_curr) * slope_1
        types = keys.h_0 in predictions
        if types:
            h_slope_1 = 1 / sigma_curr * (h_curr - predictions[keys.h_0])
            h_next = h_curr + (sigma_next - sigma_curr) * h_slope_1

        # step_2: second order step
        if sigma_next > 0:
            batch[properties.R] = x_next
            if types:
                batch[keys.h] = h_next
            if self.time_aware_model:
                batch[keys.t] = self.noise_schedule.get_time_tensor(
                    batch[properties.idx_m],
                    sigma_next,
                    device=self.device,
                    dtype=self.dtype,
                )
            predictions = self.calculator.calculate(batch, x0=True, sigma_t=sigma_next)
            x_pred0 = predictions[keys.x_0]
            slope_2 = 1 / sigma_next * (x_next - x_pred0)
            x_next = x_curr + (sigma_next - sigma_curr) * (slope_2 + slope_1) / 2
            if types:
                h_slope_2 = 1 / sigma_next * (h_next - predictions[keys.h_0])
                h_next = h_curr + (sigma_next - sigma_curr) * (h_slope_2 + h_slope_1) / 2

        # print log
        self._log(
                step=step,
                sigma=sigma_curr,
                t=self.noise_schedule.get_timestep(step),
            )

        # update batch
        batch[properties.R] = x_next
        if types:
            batch = self.update_types(batch, h_next)

        return batch


class AdaptiveHeunSampler(DiffusionSampler):
    """
    Sampler for Heun's method for denoising of atomic positions in batch mode using a trained model.
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
        Run consistency sampling in batch mode using a trained model.

        Args:
            batch: A schnetpack batch dict containing batched molecular data.

        Returns:
            batch: A dictionary with updated atomic positions after consistency sampling.
        """
        # prepare calculations and initialize batch values
        x_curr = batch[properties.R]
        sigma_curr = batch[keys.sigma_curr]

        # === Step 1 ===
        if self.time_aware_model:
            batch[keys.t] = self.noise_schedule.get_time_tensor(
                batch[properties.idx_m],
                sigma_curr,
                device=self.device,
                dtype=self.dtype,
            )
        predictions = self.calculator.calculate(batch, x0=True, estimate_sigma=True)
        x_pred0 = predictions[keys.x_0]
        sigma_predicted = predictions[keys.sigma_curr]

        # get sigma
        if self.estimate_sigma:
            sigma_curr = sigma_predicted
        else:
            sigma_curr = batch[keys.sigma_curr]

        # get sigma_next
        step_size = self.adjust_step_size(
            batch=batch,
            sigma_curr=sigma_curr,
            step=step,
        )

        sigma_next = self.noise_schedule.sigma_next(
            sigma_curr=sigma_curr,
            step_size=step_size,
            is_last_step=step == 1,
        )
        sigma_next = self.clamp_sigma_next(sigma_next=sigma_next, step=step)

        batch[keys.sigma_next] = sigma_next

        # calculate x_next
        slope_1 = 1 / sigma_curr * (x_curr - x_pred0)
        x_next = x_curr + (sigma_next - sigma_curr) * slope_1
        self.update_batch(
            batch,
            updates={
                keys.sigma_curr: sigma_next,
                keys.sigma_last: sigma_curr,
                properties.R: x_next,
            },
        )

        # mid-step on purpose: molecules converging now skip the second Heun
        # stage (no extra NFE, no correction)
        batch = self.update_convergence(batch, sigma_next, 0.0)

        # === Step 2 ===
        # get predictions
        if self.time_aware_model:
            batch[keys.t] = self.noise_schedule.get_time_tensor(
                batch[properties.idx_m],
                sigma_next,
                device=self.device,
                dtype=self.dtype,
            )
        predictions = self.calculator.calculate(batch, x0=True, sigma_t=sigma_next)
        x_pred0 = predictions[keys.x_0]
        slope_2 = 1 / sigma_next * (x_next - x_pred0)
        x_next = x_curr + (sigma_next - sigma_curr) * (slope_2 + slope_1) / 2

        # === Logging and updating batch ===
        # print log
        self._log(
                step=step,
                sigma=sigma_curr.mean(),
                sigma_next=sigma_next.mean(),
                step_size=step_size.mean(),
            )

        # update batch (frozen molecules keep their previous values)
        batch = self.update_batch(
            batch,
            updates={properties.R: x_next, keys.step_size: step_size, keys.sigma_predicted: sigma_predicted},
        )

        return batch


class AdaptiveStochasticHeunSampler(DiffusionSampler):
    """
    Stochastic Heun sampler with sigma estimation and adaptive step sizes.
    """

    def __init__(
        self,
        n_steps_target: int = 10,
        sigma_tmin: float = 0.01,
        sigma_tmax = 15.,
        sigma_churn = 60.,
        sigma_noise = 1.,
        gamma_scale: float = 1.0,
        estimate_sigma: bool = True,
        **kwargs,
    ):
        super().__init__(n_steps_target=n_steps_target, **kwargs)
        self.sigma_tmin = sigma_tmin
        self.sigma_tmax = sigma_tmax
        self.sigma_churn = sigma_churn
        self.sigma_noise = sigma_noise
        self.gamma_scale = gamma_scale
        self.estimate_sigma = estimate_sigma
        
    def denoise_step(self, batch: Dict[str, torch.Tensor], step: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Run stochastic Heun denoising with adaptive sigma prediction.
        """
        x_curr = batch[properties.R]
        if step == self.n_steps:
            if self.time_aware_model:
                batch[keys.t] = self.noise_schedule.get_time_tensor(
                    batch[properties.idx_m],
                    batch[keys.sigma_curr],
                    device=self.device,
                    dtype=self.dtype,
                )
            sigma_prediction = self.calculator.calculate(
                batch,
                estimate_sigma=True,
            )
            batch[keys.sigma_curr] = sigma_prediction[keys.sigma_curr]
        sigma_curr = batch[keys.sigma_curr]

        # add noise to sample
        noise = torch.randn_like(x_curr) * self.sigma_noise
        gamma = torch.zeros_like(sigma_curr)
        churn_mask = (sigma_curr >= self.sigma_tmin) & (sigma_curr <= self.sigma_tmax)
        gamma[churn_mask] = min(self.sigma_churn / self.n_steps, 2**0.5 - 1)
        sigma_hat = sigma_curr + gamma * sigma_curr
        x_hat = x_curr + noise * (sigma_hat**2 - sigma_curr**2)**0.5

        # === Step 1 ===
        # get predictions
        batch[properties.R] = x_hat
        if self.time_aware_model:
            batch[keys.t] = self.noise_schedule.get_time_tensor(
                batch[properties.idx_m],
                sigma_hat,
                device=self.device,
                dtype=self.dtype,
            )
        predictions = self.calculator.calculate(batch, x0=True, estimate_sigma=True)
        x_pred0 = predictions[keys.x_0]
        sigma_hat_predicted = predictions[keys.sigma_curr].clamp_min(torch.finfo(x_hat.dtype).eps)
        sigma_predicted = sigma_hat_predicted / (1 + self.gamma_scale * gamma)
        sigma_hat = sigma_hat_predicted

        # calculate sigma_next
        step_size = self.adjust_step_size(
            batch=batch,
            sigma_curr=sigma_predicted,
            step=step,
        )
        sigma_next = self.noise_schedule.sigma_next(
            sigma_curr=sigma_predicted,
            step_size=step_size,
            is_last_step=step == 1,
        )
        sigma_next = self.clamp_sigma_next(sigma_next=sigma_next, step=step).clamp_min(torch.finfo(x_pred0.dtype).eps)
        batch[keys.sigma_next] = sigma_next

        # calculate x_next
        slope_1 = (x_hat - x_pred0) / sigma_hat_predicted
        x_next = x_hat + (sigma_next - sigma_hat_predicted) * slope_1
        self.update_batch(
            batch,
            updates={
                keys.sigma_curr: sigma_next,
                keys.sigma_last: sigma_predicted,
                properties.R: x_next,
            },
        )
        # mid-step on purpose: molecules converging now skip the second Heun
        # stage (no extra NFE, no correction)
        batch = self.update_convergence(batch, sigma_next, 0.01)

        # === Step 2 ===
        predictions = self.calculator.calculate(batch, x0=True, sigma_t=sigma_next)
        x_pred0 = predictions[keys.x_0]
        slope_2 = (x_next - x_pred0) / sigma_next
        x_next_corrected = x_hat + (sigma_next - sigma_hat_predicted) * (slope_2 + slope_1) / 2
        x_next = torch.where(sigma_next > 0, x_next_corrected, x_next)

        self._log(
                step=step,
                sigma=sigma_predicted.mean(),
                sigma_hat=sigma_hat_predicted.mean(),
                sigma_next=sigma_next.mean(),
                step_size=step_size.mean(),
            )

        batch = self.update_batch(
            batch,
            updates={
                properties.R: x_next,
                keys.step_size: step_size,
                keys.sigma_predicted: sigma_predicted,
            },
        )

        return batch


class StochasticHeunSampler(DiffusionSampler):
    """
    Sampler for Stochastic Heun's method for denoising of atomic positions in batch mode using a trained model.
    """

    def __init__(
        self,
        sigma_tmin: float = 0.01,
        sigma_tmax = 15.,
        sigma_churn = 60.,
        sigma_noise = 1.,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_tmin = sigma_tmin
        self.sigma_tmax = sigma_tmax
        self.sigma_churn = sigma_churn
        self.sigma_noise = sigma_noise

    def denoise_step(self, batch: Dict[str, torch.Tensor], step: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Run consistency sampling in batch mode using a trained model.

        Args:
            batch: A schnetpack batch dict containing batched molecular data.

        Returns:
            batch: A dictionary with updated atomic positions after consistency sampling.
        """
        # prepare calculatoins
        sigma_curr = self.noise_schedule.get_sigma(step)
        sigma_next = self.noise_schedule.get_sigma(step - 1)
        x_curr = batch[properties.R]

        # add noise to sample
        noise = torch.randn_like(x_curr) * self.sigma_noise
        gamma = 0
        if (sigma_curr >= self.sigma_tmin) and (sigma_curr <= self.sigma_tmax):
            gamma = min(self.sigma_churn / self.n_steps, 2**0.5 - 1)
        sigma_hat = sigma_curr + gamma * sigma_curr
        x_hat = x_curr + noise * (sigma_hat**2 - sigma_curr**2)**0.5
        h_curr = batch.get(keys.h)
        if h_curr is not None:
            h_noise = torch.randn_like(h_curr) * self.sigma_noise
            batch[keys.h] = h_hat = h_curr + h_noise * (sigma_hat**2 - sigma_curr**2)**0.5

        # === Step 1 ===
        # calculate d_1 and x_next
        batch[properties.R] = x_hat
        if self.time_aware_model:
            batch[keys.t] = self.noise_schedule.get_time_tensor(
                batch[properties.idx_m],
                sigma_hat,
                device=self.device,
                dtype=self.dtype,
            )
        predictions = self.calculator.calculate(batch, x0=True, sigma_t=sigma_hat)
        x_pred0 = predictions[keys.x_0]
        slope_1 = (x_hat - x_pred0) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * slope_1
        types = keys.h_0 in predictions
        if types:
            h_slope_1 = (h_hat - predictions[keys.h_0]) / sigma_hat
            h_next = h_hat + (sigma_next - sigma_hat) * h_slope_1

        # === Step 2 ===
        # second order step
        if sigma_next > 0:
            batch[properties.R] = x_next
            if types:
                batch[keys.h] = h_next
            if self.time_aware_model:
                batch[keys.t] = self.noise_schedule.get_time_tensor(
                    batch[properties.idx_m],
                    sigma_next,
                    device=self.device,
                    dtype=self.dtype,
                )
            predictions = self.calculator.calculate(batch, x0=True, sigma_t=sigma_next)
            x_pred0 = predictions[keys.x_0]
            slope_2 = (x_next - x_pred0) / sigma_next
            x_next = x_hat + (sigma_next - sigma_hat) * (slope_2 + slope_1) / 2
            if types:
                h_slope_2 = (h_next - predictions[keys.h_0]) / sigma_next
                h_next = h_hat + (sigma_next - sigma_hat) * (h_slope_2 + h_slope_1) / 2

        # print log
        self._log(
                step=step,
                sigma=sigma_curr,
            )

        # update batch
        batch[properties.R] = x_next
        if types:
            batch = self.update_types(batch, h_next)

        return batch
