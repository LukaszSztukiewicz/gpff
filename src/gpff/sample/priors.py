from abc import ABC, abstractmethod
import torch
from typing import Dict, Optional
from schnetpack import properties

from gpff import keys
from gpff.utils import (
    batch_apply_covariances,
    batch_match_principal_variances,
)
from gpff.utils.alignment import _pad_by_molecule

# ShapeGMM used to live here under the name CovarianceGMM, and cached models are
# pickled against that name. Keep it importable so torch.load can resolve them.
from gpff.sample.shape_predictor import ShapeGMM as CovarianceGMM  # noqa: F401

__all__ = [
    "Prior",
    "GaussianPrior",
    "ShapePrior",
    "ScaffoldPrior",
    "ScaffoldGaussianPrior",
]


class Prior(ABC):
    @abstractmethod
    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of priors from a batch of samples.

        Args:
            batch: A batch of samples from the prior.
            conditions: Optional conditioning shared with the sampler, e.g.
                target variances or scaffold masks. Which entries are required
                depends on the prior.
        """
        pass


class GaussianPrior(Prior):
    def __init__(self, sigma: float = 1.):
        self.sigma = sigma

    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch[properties.R] = torch.randn_like(batch[properties.R]) * self.sigma
        return batch


class ShapePrior(Prior):
    """
    Prior that draws positions with prescribed principal variances per molecule.

    Conditions:
        - keys.target_variances: (n_molecules, 3) principal variances,
          descending, ordered like idx_m.unique(). See
          :meth:`ShapeGMM.sample_from_batch`.
    """

    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate priors matching one target variance vector per molecule.
        """
        random_positions = torch.randn_like(batch[properties.R])
        batch[properties.R] = batch_match_principal_variances(
            random_positions,
            conditions[keys.target_variances],
            batch[properties.idx_m],
        )
        return batch


class ScaffoldPrior(Prior):
    """
    Prior for scaffold-based sampling where scaffold atoms stay at their original
    positions and the free atoms are drawn from a normal distribution with the
    given per-molecule covariance and center.

    The prior computes nothing itself: it consumes the same conditions the
    scaffold sampler is conditioned with.

    Conditions:
        - keys.scaffold_mask: (n_atoms,) boolean mask, True for scaffold atoms.
        - keys.free_covariance: (n_molecules, 3, 3) target covariance of the
          free atoms, ordered like idx_m.unique().
        - keys.free_center: (n_molecules, 3) target centroid of the free atoms,
          ordered like idx_m.unique().
    """

    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Replace the free atom positions with samples matching the conditioned
        covariance and center; scaffold atoms keep their original positions.
        """
        idx_m = batch[properties.idx_m]
        free_mask = ~conditions[keys.scaffold_mask]
        random_positions = torch.randn_like(batch[properties.R][free_mask])
        batch[properties.R][free_mask] = batch_apply_covariances(
            random_positions,
            idx_m[free_mask],
            conditions[keys.free_covariance],
            centers=conditions[keys.free_center],
            molecule_ids=idx_m.unique(),
        )
        return batch


class ScaffoldGaussianPrior(Prior):
    """
    Prior for scaffold-based sampling where the scaffold keeps its geometry and
    the free atoms are drawn from an isotropic normal distribution, i.e. the
    same prior :class:`GaussianPrior` uses but applied to the free atoms only.

    Each molecule is first translated so that its scaffold's center of geometry
    sits at the origin. The free atoms are drawn around the origin, so without
    this the frozen scaffold and the free atom cloud would live in different
    frames: dataset structures are not centered.

    Nothing prescribes where or how large the free atom cloud is; see
    :class:`ScaffoldPrior` for a covariance- and center-matched prior, which
    needs no centering because it places the free atoms in the scaffold's own
    frame.

    Conditions:
        - keys.scaffold_mask: (n_atoms,) boolean mask, True for scaffold atoms.
    """

    def __init__(self, sigma: float = 1.):
        self.sigma = sigma

    def sample_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Center each molecule on its scaffold, then replace the free atom
        positions with isotropic normal samples.
        """
        positions = batch[properties.R]
        idx_m = batch[properties.idx_m]
        scaffold_mask = conditions[keys.scaffold_mask]

        # mean scaffold position of each molecule, from the padded batch
        inverse, _, counts, scaffold_positions, _ = _pad_by_molecule(
            torch.where(scaffold_mask[:, None], positions, 0.0), idx_m
        )
        n_scaffold = torch.bincount(inverse[scaffold_mask], minlength=len(counts))
        positions -= (scaffold_positions.sum(dim=1) / n_scaffold[:, None])[inverse]

        free_mask = ~scaffold_mask
        positions[free_mask] = torch.randn_like(positions[free_mask]) * self.sigma
        batch[properties.R] = positions
        return batch
