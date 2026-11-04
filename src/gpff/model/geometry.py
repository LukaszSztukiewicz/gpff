from typing import NamedTuple

import torch
import schnetpack.nn as snn


__all__ = ["MoleculeGeometry", "molecule_geometry", "radius_of_gyration"]

# Floor of the mean square radius in A^2, so that a single atom (or a molecule
# collapsed to a point) has a finite log R_g and a zero direction.
_EPSILON = 1e-6


class MoleculeGeometry(NamedTuple):
    """
    The size and shape of every molecule of a batch, and where its atoms lie in it.
    All entries are invariant to translations; the invariants and radial are
    invariant to rotations and reflections too, the direction is a vector.

    Attributes:
        log_rg: (n_molecules,) log of the radius of gyration R_g, the root mean
            square distance of the atoms from the centroid.
        shape: (n_molecules, 2) invariants of the gyration tensor
            G = mean_i c_i c_i^T of the centered positions c_i, scaled to [0, 1]
            (they are 1 for three equal principal variances, a ball, and 0 for
            a line): 3 I2 / tr(G)^2 and 27 det(G) / tr(G)^3, with I2 the sum of
            the principal 2 x 2 minors of G. tr(G) = R_g^2 is the third invariant
            and enters through log_rg.
        direction: (n_atoms, 3) c_i / R_g, the position of an atom relative to the
            centroid in units of R_g.
        radial: (n_atoms, 1) |c_i| / R_g.
    """

    log_rg: torch.Tensor
    shape: torch.Tensor
    direction: torch.Tensor
    radial: torch.Tensor


def molecule_geometry(positions: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor) -> MoleculeGeometry:
    """
    The MoleculeGeometry of a batch, in O(n_atoms).

    Args:
        positions: (total_atoms, 3) positions.
        idx_m: (total_atoms,) molecule of every atom.
        n_atoms: (n_molecules,) number of atoms of every molecule.
    """
    n_molecules = n_atoms.shape[0]
    count = n_atoms.to(positions.dtype)
    centroid = snn.scatter_add(positions, idx_m, dim_size=n_molecules) / count[:, None]
    centered = positions - centroid[idx_m]
    gyration = (
        snn.scatter_add(centered[:, :, None] * centered[:, None, :], idx_m, dim_size=n_molecules)
        / count[:, None, None]
    )
    # closed forms of the invariants of the symmetric [[a, d, e], [d, b, f], [e, f, c]]: smooth
    # for the singular tensors of molecules with 1 or 2 atoms, where torch's det is not
    a, b, c = gyration[:, 0, 0], gyration[:, 1, 1], gyration[:, 2, 2]
    d, e, f = gyration[:, 0, 1], gyration[:, 0, 2], gyration[:, 1, 2]
    trace = a + b + c
    second = a * b + a * c + b * c - d * d - e * e - f * f
    determinant = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
    scale = trace + _EPSILON
    shape = torch.stack([3 * second / scale**2, 27 * determinant / scale**3], dim=-1)
    rg = scale.sqrt()
    direction = centered / rg[idx_m][:, None]
    # the epsilon keeps the gradient of the norm finite at the centroid
    radial = torch.sqrt(direction.pow(2).sum(-1, keepdim=True) + 1e-12)
    return MoleculeGeometry(rg.log(), shape, direction, radial)


def radius_of_gyration(positions: torch.Tensor, idx_m: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """
    (n_molecules,) root mean square distance of the atoms of every molecule from its centroid.
    """
    n_molecules = n_atoms.shape[0]
    count = n_atoms.to(positions.dtype)
    centroid = snn.scatter_add(positions, idx_m, dim_size=n_molecules) / count[:, None]
    mean_square = (
        snn.scatter_add((positions - centroid[idx_m]).pow(2).sum(-1), idx_m, dim_size=n_molecules) / count
    )
    return (mean_square + _EPSILON).sqrt()
