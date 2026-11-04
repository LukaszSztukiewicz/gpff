import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from schnetpack.nn.radial import gaussian_rbf


__all__ = ["ExpGaussianRBF", "GridGaussianRBF", "UniformGaussianRBF", "HybridGaussianRBF", "LogGaussianRBF"]


def exponential_mapping(
    inputs: torch.Tensor,
    gamma: float | torch.Tensor = 1.0,
    cuspless: bool = False,
) -> torch.Tensor:
    """Exponential mapping function."""
    if cuspless:
        inputs = inputs + torch.exp(-inputs) - 1
    return torch.exp(-gamma * inputs)


def exponential_gaussian(
    inputs: torch.Tensor,
    offsets: torch.Tensor,
    widths: torch.Tensor,
    gamma: float | torch.Tensor = 1.0,
    cuspless: bool = False,
    use_exponential_weighting: bool = False,
) -> torch.Tensor:
    """Exponential Gaussian basis functions."""
    mapping = exponential_mapping(inputs, gamma, cuspless=cuspless)
    gaussian = gaussian_rbf(1 - mapping, offsets=offsets, widths=widths)
    if use_exponential_weighting:
        gaussian = gaussian * mapping.unsqueeze(-1)
    return gaussian


class ExpGaussianRBF(nn.Module):
    """
    Exponential Gaussian radial basis functions.
    
    # TODO: Add references and credit E3X and UltraRed.
    """

    def __init__(
        self,
        n_rbf: int,
        gamma: float = 1.0,
        cuspless: bool = False,
        use_exponential_weighting: bool = True,
        trainable: bool = False,
    ):
        """
        Args:
            n_rbf: total n_rbfber of Gaussian functions, :math:`N_g`.
            gamma: exponential decay factor.
            cuspless: If True, the mapping is cuspless.
            use_exponential_weighting: If True, the Gaussian functions are weighted
                by the exponential mapping.
            trainable: If True, the widths, offsets, and gamma are trainable.
        """
        super(ExpGaussianRBF, self).__init__()
        self.n_rbf = n_rbf
        self.cuspless = cuspless
        self.use_exponential_weighting = use_exponential_weighting

        offsets = torch.linspace(0, 1, n_rbf)
        widths = torch.full_like(offsets, 1.0 / (n_rbf - 1))

        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offsets)
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offsets)
            self.register_buffer("gamma", torch.tensor(float(gamma)))
            

    def forward(self, inputs: torch.Tensor):
        return exponential_gaussian(
            inputs,
            offsets=self.offsets,
            widths=self.widths,
            gamma=self.gamma,
            cuspless=self.cuspless,
            use_exponential_weighting=self.use_exponential_weighting,
        )


def _local_widths(offsets: torch.Tensor) -> torch.Tensor:
    """
    The mean of the gaps to the neighboring offsets (the gap itself at the two
    ends), so a Gaussian covers the space to its neighbors. On a uniform grid
    this is the grid spacing, schnetpack's GaussianRBF width.
    """
    gaps = offsets[1:] - offsets[:-1]
    left = torch.cat([gaps[:1], gaps])
    right = torch.cat([gaps, gaps[-1:]])
    return 0.5 * (left + right)


class GridGaussianRBF(nn.Module):
    """
    Gaussians exp(-(d - mu_k)^2 / (2 w_k^2)) at given offsets mu_k, with a width
    w_k for every one of them: schnetpack's GaussianRBF on a grid that need not
    be uniform. Without widths, every Gaussian is as wide as the mean gap to its
    neighbors (see _local_widths).

    The point of a non-uniform grid is to resolve the short distances of a
    molecule finely and still reach the distances between the atoms of a noisy
    or large one, with a few functions instead of one per 0.25 A.
    """

    def __init__(
        self,
        offsets: Sequence[float] | torch.Tensor,
        widths: Optional[Sequence[float] | torch.Tensor] = None,
        trainable: bool = False,
    ):
        """
        Args:
            offsets: increasing centers of the Gaussians.
            widths: their widths, by default the mean gap to the neighbors.
            trainable: if True, offsets and widths are trainable.
        """
        super().__init__()
        offsets = torch.as_tensor(offsets, dtype=torch.float32)
        if offsets.ndim != 1 or len(offsets) < 2 or not bool((offsets[1:] > offsets[:-1]).all()):
            raise ValueError("offsets must be at least two increasing values")
        widths = _local_widths(offsets) if widths is None else torch.as_tensor(widths, dtype=torch.float32)
        if trainable:
            self.offsets = nn.Parameter(offsets)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("widths", widths)

    @property
    def n_rbf(self) -> int:
        return len(self.offsets)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return gaussian_rbf(inputs, self.offsets, self.widths)


class UniformGaussianRBF(GridGaussianRBF):
    """
    n_rbf Gaussians on a uniform grid from start to cutoff, each as wide as the
    spacing: schnetpack's GaussianRBF, with the same values.
    """

    def __init__(self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False):
        offsets = torch.linspace(start, cutoff, n_rbf)
        # computed like schnetpack does, so that the values are the same to the last bit
        widths = torch.abs(offsets[1] - offsets[0]) * torch.ones_like(offsets)
        super().__init__(offsets, widths, trainable=trainable)


class HybridGaussianRBF(GridGaussianRBF):
    """
    A uniform grid up to ``switch`` and a geometric one from there: offsets
    spaced by ``spacing`` in [0, switch], then switch * ratio^k for k = 1, 2, ...
    up to the last one that does not pass ``cutoff``.

    The defaults (0.25 A up to 6 A, ratio 1.25 up to 300 A) take 25 + 17 = 42
    functions instead of the 4 * cutoff of a uniform grid (600 at 150 A). The
    bonded and non-bonded distances of a clean molecule keep their resolution;
    the distances of a molecule at a large noise level, 10 to 300 A, are
    resolved to within 25 % of their value.
    """

    def __init__(self, spacing: float = 0.25, switch: float = 6.0, ratio: float = 1.25, cutoff: float = 300.0):
        """
        Args:
            spacing: gap between the offsets up to ``switch``.
            switch: where the grid turns geometric.
            ratio: ratio between consecutive offsets beyond ``switch``.
            cutoff: no offset lies beyond it.
        """
        if not (0 < spacing < switch < cutoff and ratio > 1):
            raise ValueError("needs 0 < spacing < switch < cutoff and ratio > 1")
        n_uniform = round(switch / spacing)
        uniform = torch.arange(n_uniform + 1, dtype=torch.float64) * (switch / n_uniform)
        n_geometric = int(math.floor(math.log(cutoff / switch) / math.log(ratio) + 1e-9))
        geometric = switch * ratio ** torch.arange(1, n_geometric + 1, dtype=torch.float64)
        super().__init__(torch.cat([uniform, geometric]))


class LogGaussianRBF(GridGaussianRBF):
    """
    n_rbf Gaussians with log-spaced offsets from start to cutoff (geometric
    spacing, a constant ratio between neighbors), each as wide as the gaps to its
    neighbors, so its relative width is the same.
    """

    def __init__(self, n_rbf: int = 32, start: float = 0.5, cutoff: float = 300.0):
        if not (0 < start < cutoff and n_rbf >= 2):
            raise ValueError("needs 0 < start < cutoff and n_rbf >= 2")
        offsets = torch.logspace(math.log10(start), math.log10(cutoff), n_rbf, dtype=torch.float64)
        super().__init__(offsets)
