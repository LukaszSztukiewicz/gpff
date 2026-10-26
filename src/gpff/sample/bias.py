"""Bias potentials added to the pseudo-PES at sampling time.

A bias potential is an energy of the positions alone that the sampler adds to
the model's. On a conservative pseudo-PES this is exact composition: the total
force is the gradient of the summed energy, so the stationary target is the
product of the two distributions, and several restraints compose the same way.
Scores do not compose like that, which is why the composition literature needs
an energy parameterization and a Metropolis-corrected sampler to get the
product right (Du et al., ICML 2023, arXiv:2302.11552).

Both potentials here are measured against one reference point cloud shared by
the whole batch, which is the usual structure-based case: one pocket or one
reference ligand, many candidates. Nothing stops a per-molecule reference, but
it would need the cloud padded per molecule and is left out until something
asks for it.
"""
# python
import abc
from pathlib import Path
from typing import Union

import numpy as np
import torch
from ase.io import read as ase_read

# schnetpack
import schnetpack.nn as snn


__all__ = [
    "BiasPotential",
    "ReferenceCloudBias",
    "GaussianShapeBias",
    "StericBias",
]


def load_reference_cloud(path: Union[str, Path]) -> torch.Tensor:
    """
    Read a reference point cloud of shape (n_points, 3).

    ``.npy`` is read as an array of positions; anything else goes through ASE,
    so an .xyz, .pdb or .sdf of a pocket or a reference ligand works.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No reference cloud at {path}.")
    if path.suffix == ".npy":
        cloud = np.load(path)
    else:
        cloud = ase_read(str(path)).get_positions()
    cloud = torch.as_tensor(np.asarray(cloud), dtype=torch.float64)
    if cloud.ndim != 2 or cloud.shape[-1] != 3:
        raise ValueError(
            f"{path} holds an array of shape {tuple(cloud.shape)}, not (n_points, 3)."
        )
    return cloud


class BiasPotential(abc.ABC):
    """
    An energy of the positions that a sampler adds to the model's.

    Subclasses implement :meth:`energy`; the forces come from autograd on it,
    so there is one definition of the potential and no gradient to keep in
    step with it by hand.
    """

    @abc.abstractmethod
    def energy(self, positions: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
        """
        The bias energy per molecule, of shape (n_molecules,).
        """
        raise NotImplementedError

    def forces(self, positions: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
        """
        -dE/dx of this potential, of shape (n_atoms, 3).

        Grad is enabled and the positions are copied, since the samplers run
        under no_grad and go on updating their own tensor in place.
        """
        with torch.enable_grad():
            positions = positions.detach().clone().requires_grad_(True)
            total = self.energy(positions, idx_m).sum()
            (gradient,) = torch.autograd.grad(total, positions)
        return -gradient.detach()

    @staticmethod
    def _per_molecule(per_atom: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
        """
        Sum a per-atom quantity over each molecule.
        """
        return snn.scatter_add(per_atom, idx_m, dim_size=int(idx_m[-1]) + 1)


class ReferenceCloudBias(BiasPotential):
    """
    Base for the potentials measured against a fixed reference point cloud.
    """

    def __init__(self, reference_path: Union[str, Path], strength: float = 1.0):
        """
        Args:
            reference_path: the cloud's file, see :func:`load_reference_cloud`.
            strength: the prefactor the energy is scaled by.
        """
        if strength < 0:
            raise ValueError(f"strength must be non-negative, got {strength}.")
        self.reference = load_reference_cloud(reference_path)
        self.strength = strength

    def reference_for(self, positions: torch.Tensor) -> torch.Tensor:
        """
        The cloud on the positions' device and dtype.
        """
        return self.reference.to(device=positions.device, dtype=positions.dtype)

    def _squared_distances(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Squared distances from every atom to every reference point, of shape
        (n_atoms, n_points).
        """
        reference = self.reference_for(positions)
        return (positions[:, None, :] - reference[None, :, :]).square().sum(-1)


class GaussianShapeBias(ReferenceCloudBias):
    """
    Soft shape overlap with a reference cloud, in the spirit of a Gaussian
    shape score:

        E = -strength * sum_i sum_r exp(-|x_i - p_r|^2 / (2 width^2))

    Overlap lowers the energy, so this pulls the molecule into the reference's
    volume without saying which atom goes where, which is what distinguishes a
    shape restraint from the position restraint a scaffold mask imposes. Keep
    ``width`` near the spacing of the reference points: much smaller and the
    gradient vanishes away from them, much larger and the term stops telling
    shapes apart.

    The reference is used in its own frame, so it has to be placed where the
    molecule is meant to go.
    """

    def __init__(
        self,
        reference_path: Union[str, Path],
        strength: float = 1.0,
        width: float = 1.0,
    ):
        super().__init__(reference_path=reference_path, strength=strength)
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}.")
        self.width = width

    def energy(self, positions: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
        overlap = torch.exp(
            -self._squared_distances(positions) / (2 * self.width**2)
        ).sum(-1)
        return -self.strength * self._per_molecule(overlap, idx_m)


class StericBias(ReferenceCloudBias):
    """
    Soft repulsion from a fixed obstacle cloud, e.g. a protein pocket:

        E = strength * sum_i sum_r relu(min_distance - d_ir)^2

    Flat wherever the molecule clears the obstacle, so it costs nothing until a
    clash appears, and quadratic beyond that, so the force stays finite and
    the surface stays differentiable at contact.
    """

    def __init__(
        self,
        reference_path: Union[str, Path],
        strength: float = 1.0,
        min_distance: float = 3.0,
    ):
        super().__init__(reference_path=reference_path, strength=strength)
        if min_distance <= 0:
            raise ValueError(f"min_distance must be positive, got {min_distance}.")
        self.min_distance = min_distance

    def energy(self, positions: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
        # the epsilon keeps the gradient of sqrt finite on a coincident pair
        distances = (self._squared_distances(positions) + 1e-12).sqrt()
        clash = torch.clamp(self.min_distance - distances, min=0.0)
        return self.strength * self._per_molecule(clash.square().sum(-1), idx_m)
