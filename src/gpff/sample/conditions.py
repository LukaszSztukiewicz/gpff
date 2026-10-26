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
from typing import Any, Dict, List, Optional, Sequence, Union

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
    "DesignFreedomConditionGenerator",
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


class DesignFreedomConditionGenerator(ConditionGenerator):
    """
    Supplies the per-atom design-freedom field: one starting sigma per atom.

    Emits:
        - keys.sigma_field: (n_atoms, 1) starting sigma of each atom.

    The field says how far an atom may travel from the reference geometry:
    zero freezes it, the model's sigma_max leaves it de novo, and the values
    in between bound a redesign -- a few tenths of an Angstrom for a
    conformational nudge, about one for a bioisosteric swap, a few for an
    R-group. A field taking only zero and sigma_max is the scaffold mask of
    :class:`ScaffoldConditionGenerator`; what the continuous one adds is the
    range between them.

    Pairs with :class:`gpff.sample.priors.DesignFreedomPrior` and
    :class:`gpff.sample.direct_denoising.DesignFreedomDirectDenoisingSampler`.
    """

    def __init__(
        self,
        groups: Sequence[Dict[str, Any]],
        default_sigma: float = 0.0,
    ):
        """
        Args:
            groups: the freedom of each group of atoms, as entries holding a
                ``sigma`` and the ``ids`` it applies to, 0-based within every
                molecule, e.g.
                ``[{sigma: 0.0, ids: [0, 1, 2]}, {sigma: 2.0, ids: [3, 4]}]``.
                An atom may be listed in one group only. Ids past a molecule's
                last atom are skipped for that molecule, so one list serves a
                dataset of mixed sizes; an empty list gives a flat field.
            default_sigma: the freedom of every atom no group lists.
        """
        if default_sigma < 0:
            raise ValueError(f"default_sigma must be non-negative, got {default_sigma}.")
        self.default_sigma = default_sigma
        self.groups: List[tuple] = []
        claimed: Dict[int, float] = {}
        for group in groups:
            try:
                sigma = float(group["sigma"])
                ids = torch.as_tensor(list(group["ids"]), dtype=torch.long)
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Each group needs a 'sigma' and an 'ids' list, got {group}."
                ) from error
            if sigma < 0:
                raise ValueError(f"Group sigmas must be non-negative, got {sigma}.")
            if ids.numel() > 0 and ids.min() < 0:
                raise ValueError(f"Group ids must be non-negative, got {ids.tolist()}.")
            for atom_id in ids.tolist():
                if atom_id in claimed:
                    raise ValueError(
                        f"Atom {atom_id} is listed with sigma {claimed[atom_id]} and "
                        f"again with {sigma}; each atom belongs to one group."
                    )
                claimed[atom_id] = sigma
            self.groups.append((sigma, ids))

    def from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        idx_m = batch[properties.idx_m]
        sigma_field = torch.full(
            (len(idx_m), 1),
            self.default_sigma,
            dtype=batch[properties.R].dtype,
            device=idx_m.device,
        )
        for molecule_id in idx_m.unique():
            atom_indices = torch.where(idx_m == molecule_id)[0]
            n_atoms = len(atom_indices)
            for sigma, ids in self.groups:
                ids = ids[ids < n_atoms]
                if ids.numel() == 0:
                    continue
                sigma_field[atom_indices[ids.to(atom_indices.device)]] = sigma

        return {keys.sigma_field: sigma_field}
