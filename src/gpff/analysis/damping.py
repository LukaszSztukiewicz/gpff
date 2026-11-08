"""Fixed-point DD probes: oscillation, translation, rotation and internal change."""
from __future__ import annotations

import numpy as np
import torch
from schnetpack import properties

from gpff import keys
from gpff.analysis.relaxation_diagnostics import internal_displacement, kabsch_rmsd
from gpff.sample.refinement import atoms_batch


def probe_damping(structures, calculator, etas=(1.0, 0.5, 1.5), steps=60, batch_size=128):
    """Per eta, return arrays (steps x molecules) and final RMSDs (molecules).

    Iterates x <- x + eta * F/2 without convergence stopping. Cosines near -1
    indicate alternating steps, +1 steady drift. Internal steps remove the
    best infinitesimal rigid-body motion. Nonfinite predictions abort the
    probe rather than turning a failed trajectory into a convergence result.
    """
    if steps < 1 or batch_size < 1 or not len(etas) or any(not np.isfinite(e) or e <= 0 for e in etas):
        raise ValueError("Positive finite etas and positive steps/batch_size are required.")
    if any(not len(a) or not np.isfinite(a.positions).all() for a in structures):
        raise ValueError("The probe needs nonempty structures with finite positions.")
    outputs = []
    for eta in etas:
        arrays = {key: np.full((steps, len(structures)), np.nan) for key in
                  ("max_shift", "step_cosine", "sigma", "translation", "internal_max_shift")}
        rmsd, aligned = np.empty(len(structures)), np.empty(len(structures))
        for start in range(0, len(structures), batch_size):
            chunk = structures[start:start + batch_size]
            batch = calculator.cast_batch(atoms_batch(chunk, start))
            batch[keys.converged] = torch.zeros(len(chunk), device=calculator.device, dtype=torch.bool)
            batch[keys.nfe] = torch.zeros(len(chunk), device=calculator.device, dtype=torch.int32)
            ptr = np.r_[0, np.cumsum([len(a) for a in chunk])]
            previous = None
            with torch.no_grad():
                for step in range(steps):
                    prediction = calculator.calculate(batch, forces=True, estimate_sigma=True)
                    delta = eta * prediction[keys.forces] / 2
                    positions, shift = batch[properties.R].cpu().numpy(), delta.cpu().numpy()
                    if not np.isfinite(shift).all() or not np.isfinite(positions).all():
                        raise ValueError(f"Nonfinite DD trajectory at eta={eta}, step={step + 1}.")
                    sigma = prediction[keys.sigma_curr].cpu().numpy()
                    for k, (a, b) in enumerate(zip(ptr[:-1], ptr[1:])):
                        v, r = shift[a:b], positions[a:b]
                        j = start + k
                        arrays["max_shift"][step, j] = np.linalg.norm(v, axis=1).max()
                        arrays["sigma"][step, j] = sigma[a:b].mean()
                        arrays["translation"][step, j] = np.linalg.norm(v.mean(0))
                        arrays["internal_max_shift"][step, j] = np.linalg.norm(
                            internal_displacement(v, r), axis=1
                        ).max()
                        if previous is not None:
                            norm = np.linalg.norm(v) * np.linalg.norm(previous[a:b])
                            if norm > 0:
                                arrays["step_cosine"][step, j] = np.sum(v * previous[a:b]) / norm
                    previous = shift.copy()
                    batch[properties.R] = batch[properties.R] + delta
            final = batch[properties.R].cpu().numpy()
            if not np.isfinite(final).all():
                raise ValueError(f"Nonfinite final DD trajectory at eta={eta}.")
            for k, (a, b) in enumerate(zip(ptr[:-1], ptr[1:])):
                rmsd[start + k] = np.sqrt(np.square(final[a:b] - chunk[k].positions).sum(1).mean())
                aligned[start + k] = kabsch_rmsd(chunk[k].positions, final[a:b])
        outputs.append({"eta": eta, **arrays, "rmsd": rmsd, "aligned_rmsd": aligned})
    return outputs
