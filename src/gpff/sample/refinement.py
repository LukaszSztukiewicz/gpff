"""Deterministic DD continuation and convergence traces of saved geometries."""
from __future__ import annotations

import numpy as np
import torch
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn

from gpff import keys
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.sample.observers import SamplerObserver
from gpff.utils.batching import atoms_to_dict, batch_to_atoms_list


def atoms_batch(structures, first_idx=0):
    """Collate ASE structures in their existing atom and molecule order."""
    return _atoms_collate_fn([
        {**atoms_to_dict(atoms), properties.idx: torch.tensor([first_idx + k])}
        for k, atoms in enumerate(structures)
    ])


class _RefinementSampler(DirectDenoisingSampler):
    def check_convergence(self, batch, step, **kwargs):
        # The generic sampler calls a molecule converged at its step limit.
        # Retain only actual force convergence (and nonfinite failures) here.
        return super().check_convergence(batch, step=0, **kwargs)


class MaxShift(SamplerObserver):
    """Largest atomic step per molecule; NaN when that molecule was already stopped."""

    def on_start(self, sampler, batch):
        self.previous = batch[properties.R].detach().cpu().clone()
        self.previous_nfe = batch[keys.nfe].detach().cpu().clone()
        self.shifts = []

    def on_step_end(self, batch, step):
        positions = batch[properties.R].detach().cpu()
        shift = torch.linalg.norm(positions - self.previous, dim=1)
        per_molecule = torch.zeros(len(batch[properties.n_atoms]), dtype=shift.dtype).scatter_reduce_(
            0, batch[properties.idx_m].cpu(), shift, reduce="amax", include_self=False
        )
        nfe = batch[keys.nfe].detach().cpu()
        per_molecule[nfe == self.previous_nfe] = float("nan")
        self.shifts.append(per_molecule.numpy())
        self.previous, self.previous_nfe = positions.clone(), nfe.clone()


def refine_structures(structures, calculator, f_max=0.05, max_steps=200, batch_size=128, trace=False):
    """Return per-structure atoms, dd_nfe, dd_converged, dd_failed and optional trace.

    The threshold is the maximum atomic pseudo-force norm. Each accepted DD
    step is half that force; the terminal convergence check also costs an NFE.
    Limit exhaustion is not force convergence. Inputs are left untouched.
    """
    if f_max <= 0 or not np.isfinite(f_max) or max_steps < 1 or batch_size < 1:
        raise ValueError("f_max must be finite and positive; max_steps and batch_size must be positive.")
    if any(not len(a) or not np.isfinite(a.positions).all() for a in structures):
        raise ValueError("Refinement needs nonempty structures with finite positions.")
    sampler = _RefinementSampler(
        calculator=calculator, stochastic_lambda=0.0, n_steps=max_steps, f_max=f_max
    )
    results = []
    shifts = np.full((max_steps, len(structures)), np.nan, dtype=np.float32) if trace else None
    for start in range(0, len(structures), batch_size):
        chunk = structures[start:start + batch_size]
        observer = MaxShift() if trace else None
        final = sampler.denoise(atoms_batch(chunk, start), observers=[observer] if trace else [])
        if trace:
            shifts[:len(observer.shifts), start:start + len(chunk)] = np.stack(observer.shifts)
        for k, atoms in enumerate(batch_to_atoms_list(final)):
            finite = bool(np.isfinite(atoms.positions).all())
            results.append({
                "atoms": atoms,
                "dd_nfe": int(final[keys.nfe][k]),
                "dd_converged": finite and bool(final[keys.converged][k]),
                "dd_failed": not finite,
            })
    return results, shifts
