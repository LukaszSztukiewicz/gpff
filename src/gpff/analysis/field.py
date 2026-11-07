"""Sampler-free diagnostics of a learned force field.

The model is any callable mapping a schnetpack batch dict to a dict holding
``keys.forces``, read as the denoiser ``D(x) = x + F(x)/2`` of a time-agnostic
GPFF force model (the force label is ``2 (x_0 - x_t)``). A perfect field gives
known values, so these measure the field itself rather than a sampler built on
it:

- the denoising error ``e(sigma)`` (:func:`denoising_errors`) falls to
  ``1/(N-1)`` as sigma -> 0: of the 3N-3 mean-free noise directions, only the
  3 rotations cannot be told apart from the clean molecule;
- the noise level estimate ``sigma_hat = std(F/2)`` (:func:`std_per_molecule`)
  that the adaptive samplers use equals the std of the target ``x_0 - x_t``.

Everything that runs on the model's device avoids host synchronization (no
``.item()``, no data dependent shapes), so a callback can queue it behind the
training stream. Only :func:`noisy_positions` is host work: it is the
alignment of the noisy molecules, solved on the CPU like in the training
transform.
"""
from typing import Callable, Dict, Optional, Sequence

import torch
from schnetpack import properties
from schnetpack.nn import scatter_add

from gpff import keys
from gpff.utils import alignment

__all__ = [
    "center_per_molecule",
    "mean_free_noise",
    "noisy_positions",
    "denoise",
    "denoising_errors",
    "std_per_molecule",
]

Model = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


def center_per_molecule(x: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """Subtract each molecule's mean from its atoms' rows of ``x``.

    Args:
        x: (total_atoms, 3) values.
        idx_m: (total_atoms,) molecule of each atom.
        n_atoms: (n_molecules,) atoms per molecule.
    """
    counts = n_atoms.to(x.dtype).unsqueeze(-1)
    return x - (scatter_add(x, idx_m, n_atoms.shape[0]) / counts)[idx_m]


def mean_free_noise(
    x_0: torch.Tensor,
    idx_m: torch.Tensor,
    n_atoms: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Standard normal noise with zero mean per molecule, like the training noise.

    Drawn on the CPU so that a CPU ``generator`` can seed it on any device.
    """
    noise = torch.randn(x_0.shape, generator=generator, dtype=x_0.dtype)
    return center_per_molecule(noise.to(x_0.device), idx_m, n_atoms)


def noisy_positions(
    x_0: torch.Tensor,
    numbers: torch.Tensor,
    n_atoms: torch.Tensor,
    sigmas: Sequence[float],
    noise: torch.Tensor,
    align_permutation: bool = False,
    align_rotation: bool = False,
) -> torch.Tensor:
    """The noisy molecules ``x_0 + sigma * noise`` at every noise level, aligned
    to ``x_0`` like the training transform does (permutation first, then
    rotation). Host work: the permutation is a Hungarian assignment per molecule
    and noise level on the CPU.

    The same noise is used at every sigma, so the shape of the curve over sigma
    is not masked by sampling noise between noise levels.

    Args:
        x_0: (total_atoms, 3) clean positions, centered per molecule.
        numbers: (total_atoms,) atomic numbers.
        n_atoms: (n_molecules,) atoms per molecule.
        sigmas: noise levels.
        noise: (total_atoms, 3) noise, e.g. from :func:`mean_free_noise`.
        align_permutation: permute the noisy atoms to their nearest same-element
            atoms of ``x_0``.
        align_rotation: rotate the noisy molecule onto ``x_0``.

    Returns:
        (len(sigmas), total_atoms, 3) tensor.
    """
    sizes = n_atoms.tolist()
    positions = []
    for sigma in sigmas:
        x = x_0 + sigma * noise
        if align_permutation or align_rotation:
            x = torch.cat([
                _align(x_m, x_0_m, z_m, align_permutation, align_rotation)
                for x_m, x_0_m, z_m in zip(x.split(sizes), x_0.split(sizes), numbers.split(sizes))
            ])
        positions.append(x)
    return torch.stack(positions)


def _align(
    x: torch.Tensor, x_0: torch.Tensor, numbers: torch.Tensor, permutation: bool, rotation: bool
) -> torch.Tensor:
    if permutation:
        x, _ = alignment.align_permutation(x, x_0, numbers, numbers)
    if rotation:
        x = alignment.align_rotation(x, x_0)
    return x


def denoise(model: Model, batch: Dict[str, torch.Tensor], positions: torch.Tensor) -> torch.Tensor:
    """``D(x) = x + F(x)/2`` at ``positions``, with the rest of ``batch``.

    The batch's neighbor list stays valid at any positions: it is fully
    connected (``AllToAllNeighborList``).
    """
    # a fresh dict: schnetpack models write their outputs into the inputs
    inputs = {**batch, properties.R: positions}
    return positions + model(inputs)[keys.forces] / 2


@torch.no_grad()
def denoising_errors(
    model: Model,
    batch: Dict[str, torch.Tensor],
    x_0: torch.Tensor,
    sigmas: Sequence[float],
    positions: torch.Tensor,
) -> torch.Tensor:
    """Per-molecule denoising error at each noise level, in units of the noise.

    ``e[m, k] = |D(x_k) - x_0|^2 / ((3N_m - 3) sigma_k^2)``: 1 means no noise
    was removed, a perfect denoiser approaches ``1/(N_m-1)`` as sigma -> 0.
    Queues one forward pass per sigma and never waits for the device.

    Args:
        model: the force model.
        batch: the molecules (atom types, indices and neighbor list).
        x_0: (total_atoms, 3) clean positions, centered per molecule (as
            ``keys.x_0`` is).
        sigmas: noise levels.
        positions: (len(sigmas), total_atoms, 3) noisy molecules, from
            :func:`noisy_positions`.

    Returns:
        (n_molecules, len(sigmas)) tensor of errors.
    """
    idx_m = batch[properties.idx_m]
    n_atoms = batch[properties.n_atoms]
    dims = (3 * n_atoms - 3).to(x_0.dtype)

    errors = []
    for sigma, x in zip(sigmas, positions):
        squared = ((denoise(model, batch, x) - x_0) ** 2).sum(-1)
        errors.append(scatter_add(squared, idx_m, n_atoms.shape[0]) / (dims * sigma**2))
    return torch.stack(errors, dim=1)


def std_per_molecule(values: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """Unbiased std of the 3N values of each molecule, like the sampler's
    ``sigma_hat = std(F/2)``.

    Args:
        values: (total_atoms, 3) values, e.g. ``F/2`` or ``x_0 - x_t``.
        idx_m: (total_atoms,) molecule of each atom.
        n_atoms: (n_molecules,) atoms per molecule.

    Returns:
        (n_molecules,) tensor.
    """
    n_molecules = n_atoms.shape[0]
    n = (3 * n_atoms).to(values.dtype)
    mean = scatter_add(values, idx_m, n_molecules).sum(-1) / n
    centered = values - mean[idx_m].unsqueeze(-1)
    return (scatter_add(centered.square(), idx_m, n_molecules).sum(-1) / (n - 1)).sqrt()
