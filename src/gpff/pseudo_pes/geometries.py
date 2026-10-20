from typing import Optional, Union
import abc
from random import randint, random, choice

import numpy as np
import torch
from scipy.special import ndtr, ndtri

from gpff.utils.alignment import sample_covariance_noise, align_permutation


__all__ = ["VarianceExplodingGeometries", "VariancePreservingGeometries", "Geometries", "MixedGeometries"]


def linear_decay(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """
    Linear beta schedule from https://arxiv.org/pdf/2006.11239.pdf, returned as
    alphas_bar. Ported from morered.noise_schedules.linear_decay.
    """
    # rescale beta_start and beta_end as the paper uses 1000 timesteps
    scale = 1000 / timesteps
    betas = np.linspace(scale * beta_start, scale * beta_end, timesteps, dtype=np.float64)
    return np.cumprod(1 - betas)


class Geometries(abc.ABC):
    """
    Base class for sampling noisy geometries from x_0 samples.
    """

    def __init__(self, mean_free_noise: bool = False):
        self.mean_free_noise = mean_free_noise

    def __call__(self, x_0: torch.Tensor, t: Optional[int] = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample noisy geometry
        x_t, sigma_t, t, noise = self.get_noisy_geometry(x_0, t, **kwargs)

        return x_t, sigma_t, t, noise

    @abc.abstractmethod
    def get_noisy_geometry(self, x_0: torch.Tensor, t: Optional[int] = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class VarianceExplodingGeometries(Geometries):

    def __init__(
        self,
        sigma_max: Optional[float] = None,
        sigma_mean: float = -0.7,
        sigma_std: float = 1.2,
        destroy_signal: bool | str = False,
        mean_free_noise: bool = False,
        sample_uniformly: bool = False,
    ):  

        super().__init__(mean_free_noise)
        self.sigma_max = sigma_max
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.destroy_signal = destroy_signal
        self.sample_uniformly = sample_uniformly

    def sigma_quantile(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF of the noise levels get_noisy_geometry draws: the level at quantile u in (0, 1)."""
        if self.sample_uniformly:
            return u * self.sigma_max + 1e-4
        # ln(sigma) ~ N(sigma_mean, sigma_std), drawn again while sigma > sigma_max,
        # is the normal truncated at ln(sigma_max): invert its CDF
        top = ndtr((np.log(self.sigma_max) - self.sigma_mean) / self.sigma_std)
        return np.exp(self.sigma_mean + self.sigma_std * ndtri(u * top))

    def get_noisy_geometry(
        self, 
        x_0: torch.Tensor,
        t: Optional[float] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ):  
        """
        Args:
            x_0: (N, 3) tensor of initial positions
            t: (N, 1) tensor of timesteps
            noise: (N, 3) tensor of noise
            **kwargs: additional arguments
        Returns:
            x_t: (N, 3) tensor of noisy positions
            sigma_t: (N, 1) tensor of noise levels
            t: (N, 1) tensor of timesteps
            noise: (N, 3) tensor of noise
        """
        # sample noise level; sigma_quantile must describe these draws, since
        # the pinned validation noise (gpff.data.eval_noise) is drawn through it
        if t is not None:
            sigma_t = t * self.sigma_max
        elif self.sample_uniformly:
            sigma_t = torch.rand(size=(1,), device=x_0.device, dtype=x_0.dtype) * self.sigma_max + 1e-4
        else:
            sigma_t = torch.exp(torch.normal(mean=self.sigma_mean, std=self.sigma_std, size=(1,), device=x_0.device, dtype=x_0.dtype))
            while sigma_t > self.sigma_max:
                sigma_t = torch.exp(torch.normal(mean=self.sigma_mean, std=self.sigma_std, size=(1,), device=x_0.device, dtype=x_0.dtype))
        
        # calculate t
        if t is None:
            t = sigma_t/self.sigma_max
        
        # repeat t and sigma_t for each atom (shape: (N, 1))
        t = torch.ones(size=(x_0.shape[0], 1), device=x_0.device, dtype=x_0.dtype) * t
        sigma_t = torch.ones(size=(x_0.shape[0], 1), device=x_0.device, dtype=x_0.dtype) * sigma_t

        # sample noise
        if noise is None:
            noise = torch.randn_like(x_0)
        else:
            if noise.shape != x_0.shape:
                raise ValueError("noise must have the same shape as x_0")
        
        # calculate x_t
        offset = noise * sigma_t
        x_t = x_0 + offset

        return x_t, sigma_t, t, noise



class VarianceExplodingGeometriesOld(Geometries):

    def __init__(
        self,
        sigma_max: Optional[float] = None,
        use_multivariate_normal: bool = False,
        steepness: float = 1,
        destroy_signal: bool | str = False,
        per_atom_noise: bool = False,
        mean_free_noise: bool = False,
        dropout_ratio: float = 0.0,
    ):  
        # if sigma_max is None and not use_multivariate_normal:
        #     raise ValueError("sigma_max must be provided if use_multivariate_normal is False")
        # if sigma_max is not None and use_multivariate_normal:
        #     raise ValueError("sigma_max must be None if use_multivariate_normal is True")

        super().__init__(mean_free_noise)
        self.sigma_max = sigma_max or -1
        self.use_multivariate_normal = use_multivariate_normal
        self.steepness = steepness
        self.destroy_signal = destroy_signal
        self.per_atom_noise = per_atom_noise
        self.dropout_ratio = dropout_ratio

    def get_noisy_geometry(
        self, 
        x_0: torch.Tensor,
        t: Optional[float] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 1. get timesteps for each atom
        # 1.1 sample base timestep
        if t is None:
            t = random()
        t_i = torch.ones((x_0.shape[0], 1), device=x_0.device, dtype=x_0.dtype) * t

        # 1.2 add additional noise to some atoms
        if self.per_atom_noise:
            t_range = (1 - t) * random()
            t_i += t_range * torch.rand((x_0.shape[0], 1), device=x_0.device, dtype=x_0.dtype)
        if self.dropout_ratio > 0:
            mask = torch.rand((x_0.shape[0], 1), device=x_0.device, dtype=x_0.dtype) < self.dropout_ratio
            t_i[mask] = 1
        
        # 1.3 scale timestep
        t_i = self._scale_noise_level(t_i)

        # 2. add noise to the geometry
        # 2.1 sample noise
        if noise is None and self.use_multivariate_normal:
            noise = sample_covariance_noise(x_0)
        elif noise is None:
            noise = torch.randn_like(x_0) * self.sigma_max
        else:
            if noise.shape != x_0.shape:
                raise ValueError("noise must have the same shape as x_0")
        
        # 2.2 center noise
        if self.mean_free_noise:
            noise = noise - noise.mean(0, keepdim=True)

        # 2.3 sample x_t
        if self.destroy_signal == False:
            x_t = x_0 + t_i * noise
        elif self.destroy_signal == True:
            x_t = (1 - t_i) * x_0 + t_i * noise
        elif self.destroy_signal == "sqrt":
            x_t = (1 - t_i)**0.5 * x_0 + t_i**0.5 * noise
        else:
            raise ValueError(f"Invalid destroy_signal: {self.destroy_signal}")

        return x_t, t_i*self.sigma_max, t_i

    def _scale_noise_level(self, t):
        if self.steepness > 0:
            a = 1 / (torch.e**self.steepness - 1)
            t = a * torch.e ** (self.steepness * t) - a
        return t


class VariancePreservingGeometries(Geometries):
    def __init__(self, beta_min, beta_max, timesteps, center_xt: bool = True):
        super().__init__(center_xt)
        self.timesteps = timesteps
        # todo: fix later
        self.alphas = torch.tensor(linear_decay(self.timesteps, beta_min, beta_max))
        #self.alphas = torch.linspace(1, 0, self.timesteps)

    def get_noisy_geometry(
        self,
        x_0: torch.Tensor,
        t: Optional[float] = None,
        **kwargs,
    ):
        
        # process t to an integer between 0 and self.timesteps
        if t is None:
            t = randint(0, self.timesteps-1)
        else:
            if t<0 or t>1:
                raise ValueError("t must be between 0 and 1")
            t = int(t*self.timesteps)
        
        # sample x_t
        x_t = torch.sqrt(self.alphas[t]) * x_0 + torch.randn_like(x_0) * torch.sqrt(
            1 - self.alphas[t]
        )

        # convert t to a float between 0 and 1
        t = torch.tensor([t/self.timesteps], device=x_0.device, dtype=x_0.dtype)
        sigma_t = -1 * torch.ones_like(t)

        return x_t, sigma_t, t


class MixedGeometries(Geometries):
    def __init__(self, sigma_max: float=1.0, t_sigma: Optional[float] = None):
        super().__init__()
        self.supported_schedules = ["vp", "ve", "ve_ds", "multivariate"]
        self.sigma_max = sigma_max
        self.t_sigma = t_sigma

    def get_noisy_geometry(self, x_0: torch.Tensor, t: Optional[float] = None, **kwargs):
        atom_types = kwargs.pop("atom_types", None)
        # sample random schedule
        #schedule = choice(self.supported_schedules)
        schedule = "multivariate"

        # get per atom timestep
        if t is None:
            t = random()

        if self.t_sigma is not None and choice([True, False]):
            t_i = torch.normal(mean=t, std=self.t_sigma, size=(x_0.shape[0], 1))
            t_i.clamp_(min=0.0, max=1.0)
        else:
            t_i = torch.ones((x_0.shape[0], 1), device=x_0.device, dtype=x_0.dtype) * t


        # perturb geometry
        if schedule == "vp":
            noise = torch.randn_like(x_0)
            x_t = (1 - t_i)**0.5 * x_0 + t_i**0.5 * noise
        elif schedule == "ve":
            sigma_max = torch.rand(()) * (self.sigma_max - 1) + 1
            noise = torch.randn_like(x_0) * sigma_max
            x_t = x_0 + t_i * noise
        elif schedule == "ve_ds":
            sigma_max = torch.rand(()) * (self.sigma_max - 1) + 1
            noise = torch.randn_like(x_0) * sigma_max
            x_t = (1 - t_i) * x_0 + t_i * noise
        elif schedule == "multivariate":
            if atom_types is None:
                raise ValueError("atom_types must be provided for multivariate schedule")
            # TODO: how do we handle destroy signal?!
            noise = sample_covariance_noise(x_0)
            # align permutation of noise and x_0
            # noise, atom_types = align_permutation(
            #     positions=noise,
            #     target_positions=x_0,
            #     numbers=atom_types,
            #     target_numbers=atom_types,
            # )
            x_t = (1 - t_i) * x_0 + t_i * noise
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        return x_t, t_i, t_i