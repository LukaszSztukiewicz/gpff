"""Checks on the pseudo-PES a trained GPFF model defines.

The pseudo-PES labels put a clean data geometry at the bottom of the surface:
the force labels 2(x_0 - x_t) vanish at x_t = x_0. So a data geometry should be
a stationary point of a trained model, and a minimum rather than a saddle. The
four checks here test that, straight on the model, without a sampler:

    residual force   |F(x_0)|, which should vanish
    conservativity   the relative asymmetry of the force Jacobian J = dF/dx,
                     which vanishes exactly when F is the gradient of a
                     potential
    curvature        the spectrum of the implied Hessian H = -(J + J^T)/2,
                     which should hold the six zero modes of the translations
                     and rotations the learned energy is invariant under, and
                     no negative one
    relaxation       whether following the forces carries a perturbed geometry
                     back to the data geometry

Only a model whose forces come from an energy (experiment=gpff_conservative)
can pass the conservativity check. A direct force head has an asymmetric
Jacobian by construction, hence no valid Hessian, and the curvature numbers
below then describe its symmetric part rather than a potential (Bigi et al.,
ICML 2025, arXiv:2412.11569).
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from schnetpack import properties

from gpff import keys
from gpff.utils.alignment import align_rotation

__all__ = [
    "aligned_rmsd",
    "count_modes",
    "diagnose",
    "force_jacobian",
    "hessian_eigenvalues",
    "jacobian_asymmetry",
    "relax",
    "residual_force",
]


def _model_forces(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    The model's forces at ``positions``, with the rest of ``batch`` unchanged.

    Grad is enabled because a model deriving its forces from an energy needs
    it. Only the batch dict is copied: the model sets requires_grad on the
    positions it is handed, so a caller that keeps its own passes a copy.
    """
    with torch.enable_grad():
        return model({**batch, properties.R: positions})[keys.forces]


def residual_force(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Per-atom force norms at the batch's geometry, of shape (n_atoms,).

    The force labels vanish at a data geometry, so on one of those these norms
    measure how far the model puts it from a stationary point of the surface.
    """
    forces = _model_forces(model, batch, batch[properties.R].detach().clone())
    return forces.detach().norm(dim=-1)


def force_jacobian(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    The Jacobian dF/dx of the batch's forces, of shape (3 n_atoms, 3 n_atoms).

    One molecule at a time: the Jacobian is dense and costs 3 n_atoms backward
    passes. The model is put in train mode for the call because schnetpack's
    Forces response keeps the graph for a second derivative only while
    training; PaiNN has no train-mode-dependent layer, so the forces
    themselves are unchanged.
    """
    positions = batch[properties.R].detach().clone()
    n_coordinates = positions.numel()
    was_training = model.training
    model.train()
    try:
        jacobian = torch.autograd.functional.jacobian(
            lambda x: _model_forces(model, batch, x), positions
        )
    finally:
        model.train(was_training)
    return jacobian.reshape(n_coordinates, n_coordinates).detach()


def jacobian_asymmetry(jacobian: torch.Tensor) -> float:
    """
    The relative asymmetry ||J - J^T|| / ||J + J^T|| of a force Jacobian.

    Zero exactly when the forces are the gradient of a potential, and of order
    one for a vector field that is not.
    """
    symmetric_norm = (jacobian + jacobian.T).norm()
    return ((jacobian - jacobian.T).norm() / symmetric_norm.clamp(min=1e-30)).item()


def hessian_eigenvalues(jacobian: torch.Tensor) -> torch.Tensor:
    """
    Ascending eigenvalues of the Hessian H = -(J + J^T)/2 a force Jacobian
    implies. The symmetrization is a no-op for a conservative model.
    """
    hessian = -0.5 * (jacobian + jacobian.T)
    return torch.linalg.eigvalsh(hessian.double())


def count_modes(eigenvalues: torch.Tensor, zero_tol: float = 1e-3) -> Dict[str, int]:
    """
    Split a Hessian spectrum into zero and negative modes.

    A minimum of an energy invariant under translation and rotation has six
    zero modes (five for a linear molecule) and no negative one. The tolerance
    is relative to the largest eigenvalue, so it follows the scale of the
    surface instead of the units of the labels.
    """
    threshold = zero_tol * eigenvalues.abs().max().clamp(min=1e-30)
    return {
        "n_zero_modes": int((eigenvalues.abs() <= threshold).sum()),
        "n_negative_modes": int((eigenvalues < -threshold).sum()),
    }


def relax(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    n_steps: int = 200,
    step_size: float = 0.2,
    force_tol: float = 1e-3,
) -> Tuple[torch.Tensor, int, float]:
    """
    Follow the forces from the batch's geometry.

    One step is x <- x + step_size * F(x). The labels make F = 2(x_0 - x), so
    on the exact surface every step_size < 0.5 converges, contracting the
    distance to x_0 by 1 - 2 step_size; the default leaves room for the model
    to be wrong about the direction.

    Returns the relaxed positions, the number of steps taken, and the largest
    per-atom force norm left at the end.
    """
    positions = batch[properties.R].detach().clone()
    max_force = float("nan")
    for step in range(1, n_steps + 1):
        # the model sets requires_grad on the positions it is handed, so it gets
        # its own leaf each step and the graph does not pile up over the walk
        forces = _model_forces(model, batch, positions.detach().clone()).detach()
        positions = positions + step_size * forces
        max_force = forces.norm(dim=-1).max().item()
        if max_force < force_tol:
            return positions, step, max_force
    return positions, n_steps, max_force


def aligned_rmsd(positions: torch.Tensor, target_positions: torch.Tensor) -> float:
    """
    RMSD between two geometries, after the translation and rotation the
    learned energy is invariant under.
    """
    target_positions = target_positions.detach().cpu()
    aligned = align_rotation(positions.detach().cpu(), target_positions)
    return (aligned - target_positions).square().sum(-1).mean().sqrt().item()


def diagnose(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    perturbation: float = 0.1,
    zero_tol: float = 1e-3,
    generator: Optional[torch.Generator] = None,
    **relax_kwargs,
) -> Dict[str, float]:
    """
    Run every check on one molecule's clean geometry.

    Args:
        model: the trained model.
        batch: a single-molecule batch holding a clean data geometry and a
            neighbor list.
        perturbation: standard deviation, in the geometry's units, of the
            Gaussian displacement the relaxation starts from. Zero relaxes
            from the data geometry itself.
        zero_tol: zero-mode tolerance, relative to the largest eigenvalue.
        generator: random generator for the displacement.
        **relax_kwargs: forwarded to :func:`relax`.

    Returns:
        One flat dict of numbers, ready for a table row.
    """
    data_positions = batch[properties.R].detach().clone()

    forces = residual_force(model, batch)
    jacobian = force_jacobian(model, batch)
    eigenvalues = hessian_eigenvalues(jacobian)

    # drawn on the CPU so one seed gives the same displacement on any device
    displacement = perturbation * torch.randn(
        data_positions.shape, generator=generator, dtype=data_positions.dtype
    ).to(data_positions.device)
    perturbed = {**batch, properties.R: data_positions + displacement}
    relaxed, n_relax_steps, relax_max_force = relax(model, perturbed, **relax_kwargs)

    return {
        "n_atoms": int(data_positions.shape[0]),
        "residual_force_max": forces.max().item(),
        "residual_force_rms": forces.square().mean().sqrt().item(),
        "jacobian_asymmetry": jacobian_asymmetry(jacobian),
        "eigenvalue_min": eigenvalues[0].item(),
        "eigenvalue_max": eigenvalues[-1].item(),
        **count_modes(eigenvalues, zero_tol=zero_tol),
        "perturbed_rmsd": aligned_rmsd(perturbed[properties.R], data_positions),
        "relaxed_rmsd": aligned_rmsd(relaxed, data_positions),
        "relax_steps": n_relax_steps,
        "relax_max_force": relax_max_force,
    }
