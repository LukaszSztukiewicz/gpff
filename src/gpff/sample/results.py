# python
"""Conversion of raw sampler batch snapshots into per-molecule results."""
import torch
from typing import Any, Dict, List, Optional, Tuple
from ase import Atoms
from schnetpack import properties
from gpff import keys
from gpff.utils import batch_to_atoms_list


__all__ = [
    "to_serializable",
    "clone_batch",
    "batches_to_trajectories",
    "batches_to_relaxed",
]


def to_serializable(value: Any) -> Any:
    """
    Convert a value (tensor, dict, list, scalar) into plain Python types.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clone the full batch state to CPU for trajectory snapshots.
    """
    cloned_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned_batch[key] = value.detach().cpu().clone()
        else:
            cloned_batch[key] = value
    return cloned_batch


def batches_to_trajectories(
    batch_results: List[Dict[str, Any]],
) -> List[List[Tuple[Atoms, Dict[str, Any]]]]:
    """
    Convert raw batch snapshots to per-molecule trajectories.

    Each returned trajectory is a list of ``(Atoms, metadata)`` tuples. If a
    trajectory converges, repeated converged tail states are removed and only
    the first converged structure is kept.
    """
    if len(batch_results) == 0:
        return []

    stepwise_atoms: List[List[Tuple[Atoms, Dict[str, Any]]]] = []
    for batch in batch_results:
        step_atoms = batch_to_atoms_list(batch)
        idx_m = batch[properties.idx_m]
        molecule_ids = idx_m.unique()
        n_atoms = idx_m.shape[0]
        n_molecules = molecule_ids.shape[0]
        molecule_masks = [idx_m == molecule_id for molecule_id in molecule_ids]
        step_metadata: List[Dict[str, Any]] = [{} for _ in range(n_molecules)]

        for key, value in batch.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] == n_atoms:
                    for molecule_idx, mask in enumerate(molecule_masks):
                        step_metadata[molecule_idx][key] = to_serializable(value[mask])
                elif value.ndim > 0 and value.shape[0] == n_molecules:
                    for molecule_idx in range(n_molecules):
                        step_metadata[molecule_idx][key] = to_serializable(value[molecule_idx])
                elif value.ndim == 0:
                    serialized_value = to_serializable(value)
                    for molecule_idx in range(n_molecules):
                        step_metadata[molecule_idx][key] = serialized_value
                else:
                    serialized_value = to_serializable(value)
                    for molecule_idx in range(n_molecules):
                        step_metadata[molecule_idx][key] = serialized_value
            else:
                serialized_value = to_serializable(value)
                for molecule_idx in range(n_molecules):
                    step_metadata[molecule_idx][key] = serialized_value

        stepwise_atoms.append(
            [
                (atoms, metadata)
                for atoms, metadata in zip(step_atoms, step_metadata)
            ]
        )

    trajectories = [[] for _ in range(len(stepwise_atoms[0]))]
    for step_atoms in stepwise_atoms:
        for molecule_idx, atom_entry in enumerate(step_atoms):
            trajectories[molecule_idx].append(atom_entry)

    trimmed_trajectories: List[List[Tuple[Atoms, Dict[str, Any]]]] = []
    for trajectory in trajectories:
        first_converged_idx: Optional[int] = None
        for step_idx, (_, metadata) in enumerate(trajectory):
            if metadata[keys.converged]:
                first_converged_idx = step_idx
                break
        if first_converged_idx is None:
            trimmed_trajectories.append(trajectory)
        else:
            trimmed_trajectories.append(trajectory[: first_converged_idx + 1])

    return trimmed_trajectories


def batches_to_relaxed(
    batch_results: List[Dict[str, Any]],
) -> List[Tuple[Atoms, Dict[str, Any]]]:
    """
    Convert raw batch snapshots to relaxed per-molecule ``(Atoms, metadata)``.

    Only the final relaxed structure per molecule is returned.
    """
    trajectories = batches_to_trajectories(batch_results)
    return [trajectory[-1] for trajectory in trajectories]
