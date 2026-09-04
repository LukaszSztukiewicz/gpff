"""Condition generators building the shared prior/sampler conditioning per batch.

A condition generator turns a data batch into the ``conditions`` dict that both
the prior and the sampler consume, e.g. target principal variances for
shape-prescribing runs or scaffold tensors for scaffold-constrained runs.
Priors and samplers stay pure consumers of these entries; which keys are
required is documented on the individual prior and sampler classes.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from schnetpack import properties

from gpff import keys
from gpff.sample.shape_predictor import ShapeGMM
from gpff.utils import compute_covariance_matrix

__all__ = [
    "ConditionGenerator",
    "ShapeConditionGenerator",
    "ScaffoldConditionGenerator",
    "PrecomputedScaffoldConditionGenerator",
]


class ConditionGenerator(ABC):
    @abstractmethod
    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Build the conditions dict for one batch.

        Args:
            batch: A schnetpack batch dict with the original (data) positions;
                call this before the prior overwrites them.

        Returns:
            Conditions shared by the prior and the sampler.
        """
        pass


class ShapeConditionGenerator(ConditionGenerator):
    """
    Supplies one target variance vector per molecule from a shape predictor.

    Emits:
        - keys.target_variances: (n_molecules, 3) principal variances,
          descending, ordered like idx_m.unique().

    Pairs with :class:`gpff.sample.priors.ShapePrior` and
    :class:`gpff.sample.direct_denoising.ShapeDirectDenoisingSampler`.
    """

    def __init__(
        self,
        shape_predictor: ShapeGMM,
        target_ratio: Optional[Sequence[float]] = None,
    ):
        """
        Args:
            shape_predictor: Fitted shape predictor.
            target_ratio: The (xx, yy, zz) principal variance ratio to prescribe
                for every molecule of the run, e.g. [0.9, 0.05, 0.05] for a rod.
                Null keeps the shape the mixture predicts; setting it keeps the
                predicted molecule size but forces the requested elongation.
        """
        self.shape_predictor = shape_predictor
        self.target_ratio = target_ratio

    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            keys.target_variances: self.shape_predictor.sample_from_batch(
                batch, target_ratio=self.target_ratio
            )
        }


class ScaffoldConditionGenerator(ConditionGenerator):
    """
    Builds the scaffold conditions from prescribed scaffold atom ids and the
    batch's original positions.

    Emits:
        - keys.scaffold_mask: (n_atoms,) boolean mask, True for scaffold atoms.
        - keys.free_covariance: (n_molecules, 3, 3) covariance of the free
          (non-scaffold) atoms, ordered like idx_m.unique().
        - keys.free_center: (n_molecules, 3) centroid of the free atoms,
          ordered like idx_m.unique().

    Pairs with :class:`gpff.sample.priors.ScaffoldPrior` and
    :class:`gpff.sample.direct_denoising.ScaffoldDirectDenoisingSampler`.
    Covariance and centroid describe the data positions, so the free atoms are
    resampled to occupy the region they originally did around the scaffold.
    """

    def __init__(self, scaffold_ids: Union[Sequence[int], Sequence[Sequence[int]]]):
        """
        Args:
            scaffold_ids: Atom indices belonging to the scaffold, 0-based
                within each molecule. Either one flat list applied to every
                molecule, or one list per molecule ordered like idx_m.unique()
                (which must cover every molecule of the batch).
        """
        scaffold_ids = list(scaffold_ids)
        if len(scaffold_ids) == 0:
            raise ValueError("scaffold_ids must not be empty.")
        self.per_molecule = isinstance(scaffold_ids[0], Sequence)
        if self.per_molecule:
            self.scaffold_ids = [torch.as_tensor(ids, dtype=torch.long) for ids in scaffold_ids]
        else:
            self.scaffold_ids = torch.as_tensor(scaffold_ids, dtype=torch.long)

    def _molecule_scaffold_ids(self, molecule_position: int, n_atoms: int) -> torch.Tensor:
        if self.per_molecule:
            if molecule_position >= len(self.scaffold_ids):
                raise ValueError(
                    f"scaffold_ids covers {len(self.scaffold_ids)} molecules, but the "
                    f"batch contains more."
                )
            local_ids = self.scaffold_ids[molecule_position]
        else:
            local_ids = self.scaffold_ids
        if local_ids.min() < 0 or local_ids.max() >= n_atoms:
            raise ValueError(
                f"Scaffold ids {local_ids.tolist()} out of range for a molecule "
                f"with {n_atoms} atoms."
            )
        if local_ids.numel() >= n_atoms:
            raise ValueError(
                f"All {n_atoms} atoms are scaffold atoms; the free covariance "
                f"and center would be undefined."
            )
        return local_ids

    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = batch[properties.R]
        idx_m = batch[properties.idx_m]

        scaffold_mask = torch.zeros(len(idx_m), dtype=torch.bool, device=idx_m.device)
        covariances = []
        centers = []
        for molecule_position, molecule_id in enumerate(idx_m.unique()):
            atom_indices = torch.where(idx_m == molecule_id)[0]
            local_ids = self._molecule_scaffold_ids(molecule_position, len(atom_indices))
            scaffold_mask[atom_indices[local_ids.to(atom_indices.device)]] = True

            free_positions = positions[atom_indices][~scaffold_mask[atom_indices]]
            covariances.append(compute_covariance_matrix(free_positions))
            centers.append(free_positions.mean(dim=0))

        return {
            keys.scaffold_mask: scaffold_mask,
            keys.free_covariance: torch.stack(covariances),
            keys.free_center: torch.stack(centers),
        }


class PrecomputedScaffoldConditionGenerator(ConditionGenerator):
    """
    Builds the scaffold conditions by looking each molecule's scaffold up from a
    precomputed file, keyed by dataset index.

    Emits the same entries as :class:`ScaffoldConditionGenerator`:

        - keys.scaffold_mask: (n_atoms,) boolean mask, True for scaffold atoms.
        - keys.free_covariance: (n_molecules, 3, 3) covariance of the free
          (non-scaffold) atoms, ordered like idx_m.unique().
        - keys.free_center: (n_molecules, 3) centroid of the free atoms,
          ordered like idx_m.unique().

    Molecules are identified by the dataset index the batch carries in
    ``properties.idx``, so batching and ordering do not matter and each molecule
    gets its own scaffold. Every sampled structure must be covered by the file;
    restrict sampling by passing a split file whose test split contains only
    those structures. Pairs with the scaffold priors and samplers; the free
    covariance and center are simply unused by the ones that do not constrain
    the free atoms' shape.
    """

    def __init__(self, scaffold_ids_path: Union[str, Path]):
        """
        Args:
            scaffold_ids_path: JSON file with a ``molecules`` list, each entry
                carrying ``dataset_idx`` and ``scaffold_ids`` (0-based within
                the molecule, in database atom order).
        """
        self.scaffold_ids_path = Path(scaffold_ids_path)
        payload = json.loads(self.scaffold_ids_path.read_text())
        try:
            molecules = payload["molecules"]
        except (TypeError, KeyError):
            raise ValueError(
                f"{self.scaffold_ids_path} must contain a 'molecules' list."
            ) from None
        self.scaffold_ids: Dict[int, torch.Tensor] = {
            int(entry["dataset_idx"]): torch.as_tensor(entry["scaffold_ids"], dtype=torch.long)
            for entry in molecules
        }
        if not self.scaffold_ids:
            raise ValueError(f"{self.scaffold_ids_path} contains no molecules.")

    @property
    def dataset_indices(self) -> List[int]:
        """
        Dataset indices this generator can condition, in file order.
        """
        return list(self.scaffold_ids)

    def _molecule_scaffold_ids(self, dataset_index: int, n_atoms: int) -> torch.Tensor:
        try:
            local_ids = self.scaffold_ids[dataset_index]
        except KeyError:
            raise ValueError(
                f"No scaffold for dataset index {dataset_index} in "
                f"{self.scaffold_ids_path}. Restrict sampling to the covered "
                f"molecules with test_indices."
            ) from None
        if local_ids.numel() == 0:
            raise ValueError(f"Dataset index {dataset_index} has an empty scaffold.")
        if local_ids.min() < 0 or local_ids.max() >= n_atoms:
            raise ValueError(
                f"Scaffold ids of dataset index {dataset_index} are out of range "
                f"for a molecule with {n_atoms} atoms."
            )
        if local_ids.unique().numel() != local_ids.numel():
            raise ValueError(f"Scaffold ids of dataset index {dataset_index} contain duplicates.")
        if local_ids.numel() >= n_atoms:
            raise ValueError(
                f"All {n_atoms} atoms of dataset index {dataset_index} are scaffold "
                f"atoms; the free covariance and center would be undefined."
            )
        return local_ids

    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if properties.idx not in batch:
            raise ValueError(
                f"Batch carries no {properties.idx}; "
                f"{type(self).__name__} needs dataset indices to look scaffolds up."
            )
        positions = batch[properties.R]
        idx_m = batch[properties.idx_m]
        dataset_indices = batch[properties.idx]

        scaffold_mask = torch.zeros(len(idx_m), dtype=torch.bool, device=idx_m.device)
        covariances = []
        centers = []
        for molecule_position, molecule_id in enumerate(idx_m.unique()):
            atom_indices = torch.where(idx_m == molecule_id)[0]
            local_ids = self._molecule_scaffold_ids(
                int(dataset_indices[molecule_position]), len(atom_indices)
            )
            scaffold_mask[atom_indices[local_ids.to(atom_indices.device)]] = True

            free_positions = positions[atom_indices][~scaffold_mask[atom_indices]]
            covariances.append(compute_covariance_matrix(free_positions))
            centers.append(free_positions.mean(dim=0))

        return {
            keys.scaffold_mask: scaffold_mask,
            keys.free_covariance: torch.stack(covariances),
            keys.free_center: torch.stack(centers),
        }
