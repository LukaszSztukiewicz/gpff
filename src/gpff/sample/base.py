# python
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List, Sequence
from abc import ABC, abstractmethod
from gpff.utils import compute_neighbors
from gpff.utils.alignment import _pad_by_molecule
from gpff.sample.schedules import NoiseSchedule, EDMNoiseSchedule
from gpff.sample.observers import SamplerObserver
from gpff.sample.results import clone_batch
from schnetpack import properties
from gpff import keys


__all__ = ["Calculator", "Sampler", "DiffusionSampler"]


class Calculator:
    """
    Wraps the denoising model for the samplers.

    Owns device/dtype placement of the model and the batch, the neighbor list,
    and converts the model's raw prediction -- noise for noise models, forces
    for force models -- into the quantities a sampler requests.

    Args:
        model: the model to run inference with.
        noise_model: if True, the model predicts noise; otherwise forces.
        noise_key: the prediction key holding the noise.
        dtype: the data type to use for model and batch.
        device: the device to use for model and batch.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        noise_model: bool = False,
        noise_key: str = keys.noise,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cuda",
    ):
        self.noise_model = noise_model
        self.noise_key = noise_key
        if isinstance(dtype, str):
            if dtype.lower() == "float32":
                self.dtype = torch.float32
            elif dtype.lower() == "float64":
                self.dtype = torch.float64
            else:
                raise ValueError(
                    f"Unknown dtype string: {dtype}. Use 'float32' or 'float64'."
                )
        else:
            self.dtype = dtype
        self.device = device
        self.update_model(model)

    def update_model(self, model: Optional[nn.Module] = None):
        """
        Update the wrapped model, moving it to the calculator's device and dtype.
        """
        if model is not None:
            model = model.to(device=self.device, dtype=self.dtype)
        self.model = model

    def check_model(self):
        """
        Raise RuntimeError if no model is set.
        """
        if self.model is None:
            raise RuntimeError(
                "Calculator model is not set. Please provide a model before calling denoise."
            )

    def update_neighbors(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Ensure the batch holds a neighbor list, updating it in place.

        The list is fully connected, so it stays valid as the positions move
        and is built once for the whole denoising run.
        """
        if properties.idx_i not in batch:
            batch.update(
                compute_neighbors(batch, fully_connected=True, device=self.device)
            )
        return batch

    def cast_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cast all tensors in a batch dict to the calculator's device and dtype.
        """
        return {
            k: (
                v.to(dtype=self.dtype, device=self.device)
                if torch.is_floating_point(v)
                else v.to(device=self.device)
            )
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

    def _compute_std_per_molecule(
        self,
        idx_m: torch.Tensor,
        target_property: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the standard deviation of target_property for each molecule.

        Args:
            idx_m: Molecule index tensor of shape (n_atoms,) mapping each atom to its molecule.
            target_property: Property tensor of shape (n_atoms, 3).

        Returns:
            sigma_t: Standard deviation tensor of shape (n_atoms, 1) with
                     corresponding std values for each atom.
        """
        inverse, _, counts, padded, mask = _pad_by_molecule(target_property, idx_m)
        n = (counts * target_property.shape[-1]).to(target_property.dtype)
        mean = padded.sum(dim=(1, 2)) / n
        centered = (padded - mean[:, None, None]) * mask[..., None]
        # unbiased, like torch.std over the molecule's flattened values
        std = (centered.square().sum(dim=(1, 2)) / (n - 1)).sqrt()
        return std[inverse].unsqueeze(-1)

    def calculate(
        self,
        batch: Dict[str, torch.Tensor],
        forces: bool = False,
        noise: bool = False,
        score: bool = False,
        x0: bool = False,
        estimate_sigma: bool = False,
        sigma_t: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one model forward pass and return the requested quantities.

        Increments the batch NFE counter for unconverged molecules.

        Args:
            batch: Input dictionary containing molecular data.
            forces: if True, return the forces.
            noise: if True, return the noise.
            score: if True, return the score.
            x0: if True, return the predicted x0.
            estimate_sigma: if True, estimate sigma_t from the prediction and
                return it under keys.sigma_curr.
            sigma_t: The current sigma_t.
        Returns:
            Dictionary with the requested quantities.
        """
        if self.noise_model and any([forces, score, x0]) and sigma_t is None:
            raise ValueError("sigma_t must be provided when any of forces, score, x0 is True and noise_model is True.")
        if not self.noise_model and any([noise, score]) and sigma_t is None and not estimate_sigma:
            raise ValueError("sigma_t must be provided when any of noise, score is True and noise_model is False.")

        self.check_model()
        batch = self.update_neighbors(batch)
        predictions = self.model(batch)
        results = {}

        # noise model
        if self.noise_model:
            epsilon = predictions[self.noise_key].detach()
            if forces:
                results[keys.forces] = -2*sigma_t*epsilon
            if noise:
                results[self.noise_key] = epsilon
            if score:
                results[keys.score] = -epsilon / sigma_t
            if x0:
                results[keys.x_0] = batch[properties.R] - sigma_t * epsilon
        # force model
        else:
            predicted_forces = predictions[keys.forces].detach()
            # Compute std of forces for each molecule
            if estimate_sigma:
                sigma_t = self._compute_std_per_molecule(batch[properties.idx_m], predicted_forces/2)
                results[keys.sigma_curr] = sigma_t
            if forces:
                results[keys.forces] = predicted_forces
            if noise:
                results[keys.noise] = - predicted_forces / (2*sigma_t)
            if score:
                results[keys.score] = predicted_forces / (2*sigma_t**2)
            if x0:
                results[keys.x_0] = batch[properties.R] + predicted_forces / 2
                # clean types: predicted directly, or from the type force of
                # models trained before (runs loving-leech, up-panda)
                if keys.h_0 in predictions:
                    results[keys.h_0] = predictions[keys.h_0].detach()
                elif keys.type_forces in predictions:
                    results[keys.h_0] = batch[keys.h] + predictions[keys.type_forces].detach() / 2
            # clean-type distribution of a discrete (absorbing) type head
            if "type_probabilities" in predictions:
                results["type_probabilities"] = predictions["type_probabilities"].detach()

        # update number of function evaluations
        batch[keys.nfe] += ~batch[keys.converged]

        return results


class Sampler(ABC):
    """
    Base class for samplers.

    Args:
        calculator: the Calculator running model inference. It owns the model
            as well as device, dtype and neighbor list handling. Defaults to a
            Calculator without a model, which has to be set via update_model
            before denoising.
        n_steps: the number of steps to take.
        time_aware_model: if True, the model is time-aware and the time is passed as an input.
        observers: optional list of SamplerObserver instances attached to every
            denoise call, e.g. a ProgressReporter or TrajectoryRecorder.
        atom_types: atomic numbers of the entries of noisy one-hot types, if
            the batch has them (keys.h).
    """

    def __init__(
        self,
        calculator: Optional[Calculator] = None,
        n_steps: int = 50,
        time_aware_model: bool = False,
        observers: Optional[List[SamplerObserver]] = None,
        atom_types: Sequence[int] = (1, 6, 7, 8, 9),
    ):
        self.calculator = calculator if calculator is not None else Calculator()
        self.n_steps = n_steps
        self.time_aware_model = time_aware_model
        self.observers: List[SamplerObserver] = list(observers or [])
        self._active_observers: Optional[List[SamplerObserver]] = None
        self.atom_types = torch.tensor(atom_types)

    @property
    def device(self) -> torch.device | str:
        return self.calculator.device

    @property
    def dtype(self) -> torch.dtype:
        return self.calculator.dtype

    def update_model(self, model: Optional[nn.Module] = None):
        """
        Update the model for the sampler.
        """
        self.calculator.update_model(model)

    def _ones_like_batch_dim(self, idx_m: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            idx_m.shape[0],
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(-1)

    def update_convergence(
        self,
        batch: Dict[str, torch.Tensor],
        value: torch.Tensor,
        threshold: float | torch.Tensor,
        reduce: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Update convergence flags and freeze mask from a per-atom target value.

        The monitored value is expected to be a per-atom scalar tensor with
        shape ``(n_atoms,)`` or ``(n_atoms, 1)``. Values are reduced to a
        per-molecule statistic (default: mean) before thresholding.
        """
        idx_m = batch[properties.idx_m]
        converged = batch[keys.converged]
        n_molecules = converged.shape[0]
        prop_atom = value.squeeze(-1) if value.ndim > 1 else value

        prop_mol = torch.scatter_reduce(
            torch.zeros(n_molecules, device=value.device, dtype=value.dtype),
            0,
            idx_m,
            prop_atom,
            reduce=reduce,
            include_self=False,
        )
        threshold_tensor = torch.as_tensor(threshold, device=prop_mol.device, dtype=prop_mol.dtype)
        batch[keys.converged] = converged | (prop_mol <= threshold_tensor)
        return batch

    def update_batch(
        self,
        batch: Dict[str, torch.Tensor],
        updates: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Update batch entries; atoms of converged molecules keep their previous values.

        All update tensors are expected to be atom-wise (leading dimension
        ``len(idx_m)``).
        """
        converged_atoms = batch[keys.converged][batch[properties.idx_m]].unsqueeze(-1)

        for key, value in updates.items():
            if key not in batch:
                batch[key] = value.clone()
                continue

            old_value = batch[key].clone()

            mask = converged_atoms
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            batch[key] = torch.where(mask, old_value, value)

        return batch

    def update_types(self, batch: Dict[str, torch.Tensor], h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Set the noisy one-hot types; each atom's element is their largest entry.
        """
        batch = self.update_batch(batch, updates={keys.h: h})
        batch[properties.Z] = self.atom_types.to(h.device)[batch[keys.h].argmax(-1)]
        return batch

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Hook to seed sampler-specific batch state before the denoising loop.
        """
        return batch

    def check_convergence(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Mark converged molecules. Called by the denoising loop after every
        denoise_step.

        The base criteria are: everything is converged on the last step, and
        molecules whose positions contain NaNs are frozen so they stop
        accumulating NFE. Subclasses add their own criteria on top. Checks
        that must run mid-step (e.g. to freeze molecules before this step's
        position update) stay inside denoise_step instead.
        """
        if step == 1:
            batch[keys.converged] = torch.ones_like(batch[keys.converged])

        positions = batch[properties.R]
        idx_m = batch[properties.idx_m]
        nan_atoms = torch.isnan(positions).any(dim=-1)
        if not nan_atoms.any():
            return batch

        nan_molecules = torch.zeros(
            batch[properties.n_atoms].shape[0],
            device=positions.device,
            dtype=torch.int32,
        )
        nan_molecules.scatter_reduce_(
            0,
            idx_m,
            nan_atoms.to(torch.int32),
            reduce="amax",
            include_self=False,
        )
        batch[keys.converged] = batch[keys.converged] | nan_molecules.bool()
        return batch

    @torch.no_grad()
    def denoise(
        self,
        batch: Dict[str, torch.Tensor],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        observers: Optional[List[SamplerObserver]] = None,
    ) -> Dict[str, Any]:
        """
        Denoise a batch and return the final batch state (cloned to CPU).

        To capture intermediate steps, pass a ``TrajectoryRecorder`` observer
        and read its ``snapshots`` after this call returns.

        Args:
            batch: A schnetpack batch dict containing batched molecular data.
            conditions: Optional sampler conditioning, e.g. scaffold masks or
                target variances. Passed to denoise_step at every step; which
                entries are required depends on the sampler.
            observers: Optional additional observers for this call only, on top
                of the sampler's own ``self.observers``.
        """
        self.calculator.check_model()

        # cast conditions once; the loop below never modifies them
        conditions = self.calculator.cast_batch(conditions or {})

        # initialize convergence flag for each molecule
        batch[keys.converged] = torch.zeros(batch[properties.n_atoms].shape[0], device=self.device, dtype=torch.bool)
        batch[keys.nfe] = torch.zeros(batch[properties.n_atoms].shape[0], device=self.device, dtype=torch.int32)

        batch = self._init_state(batch)

        # assemble the observers active for this call
        active_observers = list(self.observers) + list(observers or [])

        self._active_observers = active_observers
        try:
            for observer in active_observers:
                observer.on_start(self, batch)

            # iterative denoising loop (count down from n_steps to 1)
            for step in range(self.n_steps, 0, -1):
                batch = self.calculator.cast_batch(batch)

                # perform denoising step
                batch = self.denoise_step(batch, step=step, conditions=conditions)
                batch = self.check_convergence(batch, step=step)

                for observer in active_observers:
                    observer.on_step_end(batch, step)

                # check if the molecules have converged
                if self._batch_is_converged(batch):
                    break

            for observer in active_observers:
                observer.on_end(batch)
        finally:
            self._active_observers = None

        return clone_batch(batch)

    def _batch_is_converged(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Check if all molecules in the batch are converged.
        """
        return batch[keys.converged].all()
    
    def _log(self, **kwargs) -> None:
        """
        Forward sampler values (called from denoise_step) to the active observers.
        """
        observers = (
            self._active_observers
            if self._active_observers is not None
            else self.observers
        )
        for observer in observers:
            observer.on_log(**kwargs)

    @abstractmethod
    def denoise_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        One step of the denoising process.
        """
        raise NotImplementedError

    def get_denoising_ratio(self, step: int) -> float:
        """
        Get the denoising ratio for the given step.
        Starting at approx. 0 and increasing to 1.
        Step counts down from n_steps to 1.
        """
        if step < 1 or step > self.n_steps:
            raise ValueError(f"Step {step} must be between 1 and n_steps {self.n_steps}")
        return (self.n_steps - step + 1) / (self.n_steps)

    def get_noise_ratio(self, step: int) -> float:
        """
        Get the noise ratio for the given step.
        Starting at approx. 1 and decreasing to 0.
        """
        return 1 - self.get_denoising_ratio(step)


class DiffusionSampler(Sampler):
    """
    Base class for diffusion-based samplers using EDM noise schedule.
    
    Provides common functionality for samplers that use a noise schedule
    object exposing sigma transitions.
    """

    def __init__(
        self,
        n_steps: int = 50,
        n_steps_target: Optional[int] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        **kwargs,
    ):
        super().__init__(n_steps=n_steps, **kwargs)
        self.n_steps_target = n_steps if n_steps_target is None else n_steps_target
        if noise_schedule is None:
            noise_schedule = EDMNoiseSchedule(
                n_steps=n_steps,
            )
        self.noise_schedule: NoiseSchedule = noise_schedule

    def _init_state(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Seed the sigma state and per-molecule step size for the schedule.
        """
        idx_m = batch[properties.idx_m]
        batch[keys.sigma_curr] = self._ones_like_batch_dim(idx_m) * self.noise_schedule.sigma_max
        batch[keys.sigma_last] = batch[keys.sigma_curr].clone()
        batch[keys.step_size] = self._ones_like_batch_dim(idx_m) * 1 / (self.n_steps_target - 1)
        return batch

    def adjust_step_size(
        self,
        batch: Dict[str, torch.Tensor],
        sigma_curr: torch.Tensor | float,
        step: Optional[int] = None,
    ) -> torch.Tensor | float:
        step_size = batch[keys.step_size]

        if keys.sigma_next not in batch:
            return step_size
        sigma_last = batch[keys.sigma_last]

        t_last = self.noise_schedule.get_t(sigma_last)
        t_observed = self.noise_schedule.get_t(sigma_curr)

        delta_t_observed = t_last - t_observed
        step_size = delta_t_observed.clamp(min=1/(self.n_steps - 1))
        return step_size

    def clamp_sigma_next(
        self,
        sigma_next: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Clamp sigma_next so it never exceeds the schedule value of the next step.
        """
        return sigma_next.clamp(max=self.noise_schedule.get_sigma(step - 1))
