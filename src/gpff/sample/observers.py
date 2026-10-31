# python
"""Observers for the sampler denoising loop (logging, trajectory recording)."""
import math
import torch
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from gpff import keys
from gpff.sample.results import clone_batch
from schnetpack import properties


__all__ = [
    "SamplerObserver",
    "ProgressReporter",
    "TrajectoryDiagnosticsRecorder",
    "TrajectoryRecorder",
]


class SamplerObserver:
    """
    Hook interface for observing a sampler's denoising loop.

    Observers are attached to a sampler at construction time or passed per
    call to ``Sampler.denoise``. All hooks are optional no-ops.
    """

    def on_start(self, sampler: Any, batch: Dict[str, torch.Tensor]) -> None:
        """Called once before the denoising loop starts."""

    def on_conditions(self, conditions: Dict[str, torch.Tensor]) -> None:
        """Called once with the cast conditions used for this denoising run."""

    def on_log(self, **kwargs) -> None:
        """Called with sampler-specific values from inside denoise_step."""

    def on_step_end(self, batch: Dict[str, torch.Tensor], step: int) -> None:
        """Called after each denoising step (including postprocessing)."""

    def on_end(self, batch: Dict[str, torch.Tensor]) -> None:
        """Called once after the denoising loop finishes."""


class TrajectoryRecorder(SamplerObserver):
    """
    Record a CPU snapshot of the batch after every denoising step.

    Snapshots are available as ``recorder.snapshots`` after ``denoise``
    returns. The recorder resets on ``on_start`` so it can be reused across
    multiple ``denoise`` calls.
    """

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []

    def on_start(self, sampler: Any, batch: Dict[str, torch.Tensor]) -> None:
        self.snapshots = []

    def on_step_end(self, batch: Dict[str, torch.Tensor], step: int) -> None:
        self.snapshots.append(clone_batch(batch))


class TrajectoryDiagnosticsRecorder(SamplerObserver):
    """Record compact, per-molecule summaries throughout a denoising run.

    Unlike ``TrajectoryRecorder``, this observer does not retain coordinates.
    Its JSON-serializable records track shape, motion, finite coordinates,
    atom composition, sampler state, and sampler log values for each molecule
    after the prior and every denoising step.

    Args:
        sample_offset: Global sample index assigned to the first molecule in
            this batch.
        max_molecules: Record only the first N molecules in the batch.
        run_metadata: Small scalar identifiers copied into every record.
    """

    _state_keys = (keys.sigma_curr, keys.nfe, keys.converged)

    def __init__(
        self,
        sample_offset: int = 0,
        max_molecules: Optional[int] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.sample_offset = sample_offset
        self.max_molecules = max_molecules
        self.run_metadata = dict(run_metadata or {})
        self.records: List[Dict[str, Any]] = []
        self._molecule_ids: List[torch.Tensor] = []
        self._atom_indices: List[torch.Tensor] = []
        self._initial_centers: Dict[int, torch.Tensor] = {}
        self._previous_positions: Dict[int, torch.Tensor] = {}
        self._n_atoms = 0
        self._n_molecules = 0
        self._n_steps = 0
        self._trajectory_step = 0
        self._last_log: Dict[str, Any] = {}
        self._condition_records: Dict[int, Dict[str, Any]] = {}
        self._target_variances: Dict[int, torch.Tensor] = {}
        self._scaffold_masks: Dict[int, torch.Tensor] = {}
        self._prior_positions: Dict[int, torch.Tensor] = {}
        self._start_batch: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _json_float(value: Any) -> Optional[float]:
        value = float(value)
        return value if math.isfinite(value) else None

    @classmethod
    def _compact_value(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() > 8:
                values = value.float()
                finite = values[torch.isfinite(values)]
                if finite.numel() == 0:
                    return {"mean": None, "std": None, "min": None, "max": None}
                return {
                    "mean": cls._json_float(finite.mean()),
                    "std": cls._json_float(finite.std(unbiased=False)),
                    "min": cls._json_float(finite.min()),
                    "max": cls._json_float(finite.max()),
                }
            value = value.tolist()
        if isinstance(value, dict):
            return {str(key): cls._compact_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._compact_value(item) for item in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return cls._json_float(value) if isinstance(value, float) else value
        return str(value)

    def on_start(self, sampler: Any, batch: Dict[str, torch.Tensor]) -> None:
        self.records = []
        self._n_steps = sampler.n_steps
        molecule_index = batch[properties.idx_m]
        molecule_ids = torch.unique(molecule_index, sorted=True)
        self._n_atoms = molecule_index.shape[0]
        self._n_molecules = molecule_ids.shape[0]
        keep = self._n_molecules
        if self.max_molecules is not None:
            keep = min(keep, self.max_molecules)
        self._molecule_ids = [molecule_ids[index] for index in range(keep)]
        self._atom_indices = [
            torch.nonzero(molecule_index == molecule_id, as_tuple=False).flatten()
            for molecule_id in self._molecule_ids
        ]
        self._initial_centers = {}
        self._previous_positions = {}
        self._condition_records = {}
        self._target_variances = {}
        self._scaffold_masks = {}
        self._prior_positions = {}
        self._trajectory_step = 0
        self._last_log = {}
        self._start_batch = batch

    def on_conditions(self, conditions: Dict[str, torch.Tensor]) -> None:
        for molecule_position, atom_indices in enumerate(self._atom_indices):
            sample_idx = self.sample_offset + molecule_position
            condition_record: Dict[str, Any] = {}
            for name, value in conditions.items():
                if not isinstance(value, torch.Tensor):
                    condition_record[name] = self._compact_value(value)
                    continue
                if name == keys.scaffold_mask and value.shape[0] == self._n_atoms:
                    scaffold_mask = value[atom_indices].bool().reshape(-1)
                    self._scaffold_masks[sample_idx] = scaffold_mask
                    condition_record[name] = {
                        "constrained_atom_count": int(scaffold_mask.sum().item())
                    }
                elif value.ndim > 0 and value.shape[0] == self._n_molecules:
                    molecule_value = value[molecule_position]
                    condition_record[name] = self._compact_value(molecule_value)
                    if name == keys.target_variances:
                        self._target_variances[sample_idx] = molecule_value.detach().clone()
                elif value.ndim > 0 and value.shape[0] == self._n_atoms:
                    atom_values = value[atom_indices]
                    condition_record[name] = self._compact_value(atom_values)
                else:
                    condition_record[name] = self._compact_value(value)
            self._condition_records[sample_idx] = condition_record

        if self._start_batch is not None:
            self._record_snapshot(self._start_batch, step=self._n_steps + 1, phase="prior")
            self._start_batch = None

    def on_log(self, **kwargs) -> None:
        self._last_log = {
            str(key): self._compact_value(value)
            for key, value in kwargs.items()
        }

    def on_step_end(self, batch: Dict[str, torch.Tensor], step: int) -> None:
        self._trajectory_step += 1
        self._record_snapshot(batch, step=step, phase="denoise")
        self._last_log = {}

    def _state_value(
        self,
        batch: Dict[str, torch.Tensor],
        key: str,
        molecule_position: int,
        atom_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            return None
        if value.ndim == 0:
            return value
        if value.shape[0] == self._n_molecules:
            return value[molecule_position]
        if value.shape[0] == self._n_atoms:
            return value[atom_indices]
        return None

    def _record_snapshot(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        phase: str,
    ) -> None:
        positions = batch[properties.R].detach()
        atom_types = batch.get(properties.Z)

        for molecule_position, atom_indices in enumerate(self._atom_indices):
            sample_idx = self.sample_offset + molecule_position
            molecule_positions = positions[atom_indices]
            finite_atoms = torch.isfinite(molecule_positions).all(dim=-1)
            finite_positions = molecule_positions[finite_atoms]
            atom_count = atom_indices.numel()
            center = None
            radius_gyration = None
            principal_variances = None
            shape_ratios = None
            centroid_drift = None
            step_rms_displacement = None
            max_atom_displacement = None

            if finite_positions.shape[0] > 0:
                center = finite_positions.mean(dim=0)
                centered = finite_positions - center
                radius_gyration = self._json_float(
                    centered.square().sum(dim=-1).mean().sqrt()
                )
                initial_center = self._initial_centers.get(sample_idx)
                if initial_center is None and phase == "prior":
                    self._initial_centers[sample_idx] = center.detach().clone()
                    initial_center = center
                if initial_center is not None:
                    centroid_drift = self._json_float(torch.linalg.vector_norm(center - initial_center))

                if finite_positions.shape[0] > 1:
                    covariance = centered.T @ centered / finite_positions.shape[0]
                    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
                    principal_variances = [self._json_float(value) for value in eigenvalues]
                    if eigenvalues[0] > 0:
                        shape_ratios = [
                            self._json_float(eigenvalues[1] / eigenvalues[0]),
                            self._json_float(eigenvalues[2] / eigenvalues[0]),
                        ]
                    else:
                        shape_ratios = [None, None]

            previous_positions = self._previous_positions.get(sample_idx)
            if previous_positions is not None and previous_positions.shape == molecule_positions.shape:
                valid_displacements = finite_atoms & torch.isfinite(previous_positions).all(dim=-1)
                if valid_displacements.any():
                    displacement = torch.linalg.vector_norm(
                        molecule_positions[valid_displacements]
                        - previous_positions[valid_displacements],
                        dim=-1,
                    )
                    step_rms_displacement = self._json_float(displacement.square().mean().sqrt())
                    max_atom_displacement = self._json_float(displacement.max())
            self._previous_positions[sample_idx] = molecule_positions.detach().clone()
            if phase == "prior":
                self._prior_positions[sample_idx] = molecule_positions.detach().clone()

            target_shape_ratios = None
            condition_shape_error = None
            target_variances = self._target_variances.get(sample_idx)
            if target_variances is not None and target_variances.numel() >= 3:
                target_values = target_variances.detach().reshape(-1).float()
                if target_values[0] > 0:
                    target_shape_ratios = [
                        self._json_float(target_values[1] / target_values[0]),
                        self._json_float(target_values[2] / target_values[0]),
                    ]
                    if shape_ratios is not None and all(value is not None for value in shape_ratios):
                        condition_shape_error = self._json_float(
                            torch.tensor(shape_ratios, device=target_values.device).sub(
                                torch.stack((target_values[1] / target_values[0], target_values[2] / target_values[0]))
                            ).abs().mean()
                        )

            scaffold_displacement = None
            scaffold_mask = self._scaffold_masks.get(sample_idx)
            prior_positions = self._prior_positions.get(sample_idx)
            if scaffold_mask is not None and prior_positions is not None:
                scaffold_mask = scaffold_mask.to(device=molecule_positions.device)
                valid_scaffold = (
                    scaffold_mask
                    & torch.isfinite(molecule_positions).all(dim=-1)
                    & torch.isfinite(prior_positions).all(dim=-1)
                )
                if valid_scaffold.any():
                    scaffold_delta = molecule_positions[valid_scaffold] - prior_positions[valid_scaffold]
                    scaffold_displacement = self._json_float(
                        torch.linalg.vector_norm(scaffold_delta, dim=-1).square().mean().sqrt()
                    )

            composition = None
            composition_name = "atomic_number_counts"
            if isinstance(atom_types, torch.Tensor) and atom_types.shape[0] == self._n_atoms:
                molecule_types = atom_types[atom_indices]
                if molecule_types.ndim > 1:
                    if molecule_types.shape[-1] == 1:
                        molecule_types = molecule_types.squeeze(-1)
                    else:
                        molecule_types = molecule_types.argmax(dim=-1)
                        composition_name = "atom_type_index_counts"
                elements, counts = torch.unique(molecule_types, sorted=True, return_counts=True)
                composition = {
                    str(int(element)): int(count)
                    for element, count in zip(elements.detach().cpu(), counts.detach().cpu())
                }

            record: Dict[str, Any] = {
                **self.run_metadata,
                "sample_idx": sample_idx,
                "phase": phase,
                "trajectory_step": self._trajectory_step,
                "step": int(step),
                "n_atoms": int(atom_count),
                "finite_position_fraction": self._json_float(finite_atoms.float().mean()),
                "radius_gyration": radius_gyration,
                "principal_variances": principal_variances,
                "shape_ratios": shape_ratios,
                "target_shape_ratios": target_shape_ratios,
                "condition_shape_error": condition_shape_error,
                "scaffold_rms_displacement_from_prior": scaffold_displacement,
                "conditions": self._condition_records.get(sample_idx) or None,
                "centroid_drift": centroid_drift,
                "step_rms_displacement": step_rms_displacement,
                "max_atom_displacement": max_atom_displacement,
                "sampler_log": self._last_log or None,
            }
            if composition is not None:
                record[composition_name] = composition
            for key in self._state_keys:
                value = self._state_value(batch, key, molecule_position, atom_indices)
                if value is None:
                    continue
                if key == keys.converged:
                    record["frozen"] = bool(value.bool().any().item())
                else:
                    record[key] = self._json_float(value.float().mean())
            self.records.append(record)


class ProgressReporter(SamplerObserver):
    """
    Print sampler values and/or show a tqdm progress bar during denoising.

    Args:
        verbose: if True, print values passed to ``on_log``.
        show_progressbar: if True, show tqdm progress updates.
    """

    _step_keys = ["step", "timestep", "stop", "iteration"]

    def __init__(self, verbose: bool = False, show_progressbar: bool = False):
        self.verbose = verbose
        self.show_progressbar = show_progressbar
        self._n_steps: int = 0
        self._progress_bar: Optional[tqdm] = None
        self._last_progress_step: int = -1

    def on_start(self, sampler: Any, batch: Dict[str, torch.Tensor]) -> None:
        self._n_steps = sampler.n_steps
        self._close_progress_bar()

    def on_end(self, batch: Dict[str, torch.Tensor]) -> None:
        self._close_progress_bar()

    def _close_progress_bar(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        self._last_progress_step = -1

    def _fmt_value(self, key: str, value: Any) -> str:
        # Skip formatting step-related keys as they're used for progress tracking
        if key.lower() in self._step_keys:
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return f"{value.item():.2f}"
            return str(value)
        # Special rule for "forces" key: show 1 digit if value >= 10, 2 digits otherwise
        if key.lower() == keys.forces:
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                v = value.item()
            elif isinstance(value, (int, float)):
                v = value
            else:
                return str(value)
            if v >= 10:
                return f"{v:.1f}"
            else:
                return f"{v:.2f}"
        if isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            return f"{value.item():.2f}"
        elif isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)

    def on_log(self, **kwargs) -> None:
        # Extract step from kwargs (could be 'step', 'timestep', 'stop', etc.)
        step = None
        for key in self._step_keys:
            if key in kwargs:
                step = kwargs[key]
                break

        if self.verbose:
            print(", ".join([f"{key}={self._fmt_value(key, value)}" for key, value in kwargs.items()]))

        if not self.show_progressbar:
            return

        # Initialize progress bar on first call
        if self._progress_bar is None:
            self._progress_bar = tqdm(total=self._n_steps, desc="Denoising", unit="step", dynamic_ncols=True)
            self._last_progress_step = -1

        # Build postfix string for progress bar (excluding step-related keys)
        postfix_items = []
        for key, value in kwargs.items():
            if key.lower() not in self._step_keys:
                fmt_val = self._fmt_value(key, value)
                if fmt_val is not None:
                    postfix_items.append(f"{key}={fmt_val}")
        postfix_str = ", ".join(postfix_items)

        # Update progress bar only every 10 steps
        if step is not None:
            step_int = int(self._n_steps - step)
            # Update if it's a multiple of 10, or if it's the last step
            should_update = (step_int % 10 == 0) or (step_int == self._n_steps - 1)

            if should_update and step_int != self._last_progress_step:
                self._progress_bar.n = step_int + 1
                self._progress_bar.set_postfix_str(postfix_str)
                self._progress_bar.refresh()
                self._last_progress_step = step_int
            else:
                # Still update postfix even if not updating progress bar position
                self._progress_bar.set_postfix_str(postfix_str)
        else:
            # If no step provided, increment by 1 (but only every 10 calls)
            # This is a fallback for cases where step is not provided
            if self._progress_bar.n % 10 == 0:
                self._progress_bar.update(1)
                self._progress_bar.set_postfix_str(postfix_str)
                self._progress_bar.refresh()
