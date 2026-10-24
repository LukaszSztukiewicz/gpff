# python
"""Observers for the sampler denoising loop (logging, trajectory recording)."""
import torch
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from gpff import keys
from gpff.sample.results import _MODEL_FEATURE_KEYS, clone_batch


__all__ = ["SamplerObserver", "ProgressReporter", "TrajectoryRecorder"]


class SamplerObserver:
    """
    Hook interface for observing a sampler's denoising loop.

    Observers are attached to a sampler at construction time or passed per
    call to ``Sampler.denoise``. All hooks are optional no-ops.
    """

    def on_start(self, sampler: Any, batch: Dict[str, torch.Tensor]) -> None:
        """Called once before the denoising loop starts."""

    def on_log(self, **kwargs) -> None:
        """Called with sampler-specific values from inside denoise_step."""

    def on_step_end(self, batch: Dict[str, torch.Tensor], step: int) -> None:
        """Called after each denoising step (including postprocessing)."""

    def on_end(self, batch: Dict[str, torch.Tensor]) -> None:
        """Called once after the denoising loop finishes."""


class TrajectoryRecorder(SamplerObserver):
    """
    Record a CPU snapshot of the batch after every denoising step, without
    the model's atom features (scalar and vector representations).

    Snapshots are available as ``recorder.snapshots`` after ``denoise``
    returns. The recorder resets on ``on_start`` so it can be reused across
    multiple ``denoise`` calls.
    """

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []

    def on_start(self, sampler: Any, batch: Dict[str, torch.Tensor]) -> None:
        self.snapshots = []

    def on_step_end(self, batch: Dict[str, torch.Tensor], step: int) -> None:
        # the model's atom features are the bulk of the batch and no result reads them
        self.snapshots.append(
            clone_batch({k: v for k, v in batch.items() if k not in _MODEL_FEATURE_KEYS})
        )


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
