import torch
import torch.nn as nn
from schnetpack.nn.radial import gaussian_rbf


__all__ = ["ExpGaussianRBF"]


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