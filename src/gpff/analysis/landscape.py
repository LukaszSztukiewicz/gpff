"""Sampler-free diagnostics of a learned pseudo-force field.

The model is any callable mapping a schnetpack batch dict to a dict holding
``keys.forces``, read as the denoiser ``D(x) = x + F(x)/2`` of a time-agnostic
GPFF force model. A perfect landscape gives known values, so these measure the
field itself rather than a sampler built on it:

- the denoising error ``e(sigma)`` (:func:`denoising_errors`) falls to
  ``1/(N-1)`` as sigma -> 0: of the 3N-3 mean-free noise directions, only the
  3 rotations cannot be told apart from the clean molecule;
- clean molecules are fixed points, ``D(x0) = x0``;
- the Jacobian ``dD/dx`` at ``x0`` is symmetric, keeps rigid motions
  (eigenvalue 1) and removes every internal direction (eigenvalue 0)
  (:func:`jacobian_statistics`).
"""
from typing import Callable, Dict, List, Optional, Sequence

import torch
from schnetpack import properties
from schnetpack.nn import scatter_add

from gpff import keys
from gpff.utils.alignment import align_permutation

__all__ = [
    "denoise",
    "center_per_molecule",
    "mean_free_noise",
    "denoising_errors",
    "denoiser_jacobians",
    "jacobian_statistics",
]

Model = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


def denoise(model: Model, batch: Dict[str, torch.Tensor], positions: torch.Tensor) -> torch.Tensor:
    """``D(x) = x + F(x)/2`` at ``positions``, with the rest of ``batch``."""
    # a fresh dict: schnetpack models write their outputs into the inputs
    inputs = {**batch, properties.R: positions}
    return positions + model(inputs)[keys.forces] / 2


def center_per_molecule(x: torch.Tensor, idx_m: torch.Tensor) -> torch.Tensor:
    """Subtract each molecule's mean from its atoms' rows of ``x``."""
    n_molecules = int(idx_m.max()) + 1
    counts = scatter_add(torch.ones_like(x[:, :1]), idx_m, n_molecules)
    return x - (scatter_add(x, idx_m, n_molecules) / counts)[idx_m]


def mean_free_noise(
    x_0: torch.Tensor,
    idx_m: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Standard normal noise with zero mean per molecule, like the training noise.

    Drawn on the CPU so that a CPU ``generator`` can seed it on any device.
    """
    noise = torch.randn(x_0.shape, generator=generator, dtype=x_0.dtype)
    return center_per_molecule(noise.to(x_0.device), idx_m)


def _align_to_reference(
    x: torch.Tensor, x_0: torch.Tensor, numbers: torch.Tensor, sizes: List[int]
) -> torch.Tensor:
    """Permute each molecule's atoms in ``x`` to their nearest same-element
    atoms of ``x_0``, as the training transform does with align_permutation.
    Solved on the CPU, like in the training transform's data workers."""
    device = x.device
    x, x_0, numbers = x.cpu(), x_0.cpu(), numbers.cpu()
    return torch.cat([
        align_permutation(x_m, x_0_m, z_m, z_m)[0]
        for x_m, x_0_m, z_m in zip(x.split(sizes), x_0.split(sizes), numbers.split(sizes))
    ]).to(device)


@torch.no_grad()
def denoising_errors(
    model: Model,
    batch: Dict[str, torch.Tensor],
    x_0: torch.Tensor,
    sigmas: Sequence[float],
    noise: torch.Tensor,
    aligned: bool = False,
) -> torch.Tensor:
    """Per-molecule denoising error at each noise level, in units of the noise.

    ``e[m, k] = |D(x0 + sigma_k * noise) - x0|^2 / ((3N_m - 3) sigma_k^2)``:
    1 means no noise was removed, a perfect denoiser approaches ``1/(N_m-1)``
    as sigma -> 0. The same noise is used at every sigma, so the curve's shape
    is not masked by sampling noise between noise levels.

    Args:
        model: the force model.
        batch: the molecules (atom types, indices and neighbor list).
        x_0: (n_atoms, 3) clean positions; centered per molecule here.
        sigmas: noise levels.
        noise: (n_atoms, 3) mean-free noise, e.g. from :func:`mean_free_noise`.
        aligned: permute the noisy atoms to their nearest same-element atoms of
            ``x_0`` first: the target GPFF trains on with align_permutation.
            Otherwise the raw pairing, the target of a score model.

    Returns:
        (n_molecules, len(sigmas)) tensor of errors.
    """
    idx_m = batch[properties.idx_m]
    n_atoms = batch[properties.n_atoms]
    sizes = n_atoms.tolist()
    x_0 = center_per_molecule(x_0, idx_m)
    dims = (3 * n_atoms - 3).to(x_0.dtype)

    errors = []
    for sigma in sigmas:
        x = x_0 + sigma * noise
        if aligned:
            x = _align_to_reference(x, x_0, batch[properties.Z], sizes)
        squared = ((denoise(model, batch, x) - x_0) ** 2).sum(-1)
        errors.append(scatter_add(squared, idx_m, len(sizes)) / (dims * sigma**2))
    return torch.stack(errors, dim=1)


def denoiser_jacobians(
    model: Model, batch: Dict[str, torch.Tensor], positions: torch.Tensor
) -> List[torch.Tensor]:
    """``dD/dx`` of every molecule at ``positions``, one (3N, 3N) matrix each,
    rows and columns ordered atom by atom (x, y, z).

    Molecules do not interact, so the batch Jacobian is block diagonal: each
    backward pass selects the same output coordinate of every molecule and
    yields one row of every block, 3 * max(N) passes in total.
    """
    sizes = batch[properties.n_atoms].tolist()
    idx_m = batch[properties.idx_m]
    starts = torch.cumsum(torch.tensor([0] + sizes[:-1]), 0).to(idx_m.device)
    local = torch.arange(len(idx_m), device=idx_m.device) - starts[idx_m]

    with torch.enable_grad():
        x = positions.detach().requires_grad_(True)
        d = denoise(model, batch, x)
        rows = []
        for atom in range(max(sizes)):
            for axis in range(3):
                grad_outputs = torch.zeros_like(d)
                grad_outputs[local == atom, axis] = 1
                (row,) = torch.autograd.grad(d, x, grad_outputs, retain_graph=True)
                rows.append(row)
    # rows[3a + c][atoms of m] is row (a, c) of molecule m's block
    rows = torch.stack(rows)
    return [
        rows[: 3 * n, start : start + n].reshape(3 * n, 3 * n)
        for n, start in zip(sizes, starts.tolist())
    ]


def _rigid_motion_basis(positions: torch.Tensor) -> torch.Tensor:
    """Orthonormal (3N, 6) basis of the translations and infinitesimal
    rotations of ``positions``; (3N, 5) for a linear molecule."""
    n = positions.shape[0]
    centered = positions - positions.mean(0)
    translations = torch.eye(3, dtype=positions.dtype).repeat(n, 1)
    rotations = torch.stack(
        [torch.cross(axis.expand(n, 3), centered, dim=-1).reshape(-1)
         for axis in torch.eye(3, dtype=positions.dtype)],
        dim=1,
    )
    u, s, _ = torch.linalg.svd(torch.cat([translations, rotations], dim=1), full_matrices=False)
    return u[:, : int((s > 1e-6 * s[0]).sum())]


def jacobian_statistics(jacobian: torch.Tensor, positions: torch.Tensor) -> Dict[str, float]:
    """Compare a molecule's denoiser Jacobian at ``positions`` with the perfect
    one, which keeps rigid motions and removes every internal direction.

    Returns:
        asymmetry: ``|J - J^T| / |J|``; 0 for a conservative field.
        rigid_residual: ``|J Q - Q| / |Q|`` over the rigid-motion basis Q;
            0 when translations and rotations are kept.
        internal_mean, internal_max: eigenvalues of the symmetric part of J on
            the internal directions (complement of Q): the fraction of the
            noise kept along each internal mode. Perfect: 0. Above 1 the
            molecule is a saddle of the pseudo-PES, relaxation leaves it.
    """
    jacobian = jacobian.detach().double().cpu()
    rigid = _rigid_motion_basis(positions.detach().double().cpu())
    # the internal directions: the complement of the rigid motions
    internal = torch.linalg.svd(rigid, full_matrices=True)[0][:, rigid.shape[1] :]
    symmetric = (jacobian + jacobian.T) / 2
    eigenvalues = torch.linalg.eigvalsh(internal.T @ symmetric @ internal)
    no_internal = eigenvalues.numel() == 0
    return {
        "asymmetry": (torch.linalg.norm(jacobian - jacobian.T) / torch.linalg.norm(jacobian)).item(),
        "rigid_residual": (torch.linalg.norm(jacobian @ rigid - rigid) / torch.linalg.norm(rigid)).item(),
        "internal_mean": float("nan") if no_internal else eigenvalues.mean().item(),
        "internal_max": float("nan") if no_internal else eigenvalues.max().item(),
    }
