from typing import Optional
import abc

import torch


__all__ = ["LossWeights", "SigmaWeights", "NormWeights", "FlooredSigmaWeights"]


class LossWeights(abc.ABC):
    """
    Base class for calculating energy and force weights for the pseudo-PES loss.
    """

    def __init__(
        self,
        min_clipping: Optional[float] = None,
        max_clipping: Optional[float] = None,
    ):
        """
        Args:
            min_clipping: minimum value for the weights
            max_clipping: maximum value for the weights
        """
        self.min_clipping = min_clipping
        self.max_clipping = max_clipping

    @abc.abstractmethod
    def _get_weights(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get energy and force weights.
        """
        raise NotImplementedError

    def __call__(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate weights
        energy_weight, force_weight = self._get_weights(**kwargs)

        # apply clipping of weights
        if self.max_clipping is not None or self.min_clipping is not None:
            energy_weight = torch.clamp(
                energy_weight, min=self.min_clipping, max=self.max_clipping
            )
            force_weight = torch.clamp(
                force_weight, min=self.min_clipping, max=self.max_clipping
            )

        return energy_weight, force_weight


class SigmaWeights(LossWeights):

    def _get_weights(
        self, sigma: torch.Tensor, x_0: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy_weight = 1 / sigma**4
        force_weight = 1 / sigma**2 * torch.ones_like(x_0[:, 0]).unsqueeze(-1)

        return energy_weight, force_weight


class NormWeights(LossWeights):

    def _get_weights(
        self, energy: torch.Tensor, forces: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy_weight = 1 / energy ** 2
        force_weight = 1 / torch.linalg.norm(forces, dim=1, keepdim=True) ** 2

        return energy_weight, force_weight


class FlooredSigmaWeights(LossWeights):
    """
    Force weight 1/(sigma^2 + sigma_floor^2): 1/sigma^2 well above sigma_floor, 1/sigma_floor^2
    at sigma = 0, and no corner in between. In noise-prediction terms the loss stays flat down
    to sigma_floor and fades like sigma^2 below it, where the force labels become smaller than
    the model's own error. SigmaWeights with max_clipping=1 instead flattens the weight below
    1 Å, so the loss there already fades like sigma^2.
    """

    def __init__(self, sigma_floor: float = 0.03, **kwargs):
        """
        Args:
            sigma_floor: noise level below which the weight levels off
        """
        super().__init__(**kwargs)
        self.sigma_floor = sigma_floor

    def _get_weights(
        self, sigma: torch.Tensor, x_0: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = 1 / (sigma**2 + self.sigma_floor**2)
        energy_weight = weight**2
        force_weight = weight * torch.ones_like(x_0[:, 0]).unsqueeze(-1)

        return energy_weight, force_weight
