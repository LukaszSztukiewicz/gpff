import torch
import abc


__all__ = ["PseudoCalculator", "VEPseudoCalculator"]


class PseudoCalculator(abc.ABC):

    @abc.abstractmethod
    def get_energy(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_forces_analytic(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_forces(self, x_0: torch.Tensor, x_t: torch.Tensor, **kwargs) -> torch.Tensor:
        x_t.requires_grad = True
        energy = self.get_energy(x_0=x_0, x_t=x_t, **kwargs)
        energy.backward()
        forces = -x_t.grad
        forces.detach()
        energy.detach()
        x_t.requires_grad = False
        return forces


class VEPseudoCalculator(PseudoCalculator):
    """
    Calculate pseudo energy and forces for geometries based on variance exploding diffusion.
    """

    def get_energy(self, x_0: torch.Tensor, x_t: torch.Tensor, **kwargs):
        return ((x_t - x_0) ** 2).sum().unsqueeze(-1)

    def get_forces_analytic(self, x_0: torch.Tensor, x_t: torch.Tensor, **kwargs):
        return 2 * (x_0 - x_t)
