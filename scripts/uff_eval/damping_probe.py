"""Is DD's step-size floor an oscillation? Fixed-point iteration x <- x + eta * F/2 on saved structures.

    python damping_probe.py <model_path> <structures.db> [--limit 512] [--steps 60] [--etas 1 0.5 1.5]

Per eta and step: median and p90 over molecules of the largest atomic step, and of the cosine between
consecutive steps (flattened per molecule; -1 is a period-2 oscillation, +1 a steady drift); the
translation part of the step (|mean step|) and the largest atomic step after removing translation and
rotation. Also the median sigma estimate (std of F/2) and, at the end, the RMSD from the start, raw and
after optimal superposition (Kabsch), i.e. the internal change.
"""
import argparse

import numpy as np
import torch
from ase.db import connect
from schnetpack import properties
from schnetpack.utils import load_model

from gpff import keys
from gpff.sample import Calculator
from gpff.utils.alignment import _pad_by_molecule
from refine_dd import to_batch


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ rotation - b) ** 2).sum(1).mean()))


def internal_step(step, positions):
    """The step minus its best rigid-body part (translation + linearized rotation omega x r)."""
    r = positions - positions.mean(0)
    v = step - step.mean(0)
    # least squares for omega in v ~ omega x r = -[r]_x omega
    cross = np.stack([np.cross(np.eye(3)[k], r) for k in range(3)], axis=-1)  # (n, 3, 3): column k = e_k x r
    omega, *_ = np.linalg.lstsq(cross.reshape(-1, 3), v.reshape(-1), rcond=None)
    return v - cross @ omega


def per_molecule(values, idx_m, reduce):
    out = torch.zeros(int(idx_m.max()) + 1, device=values.device, dtype=values.dtype)
    return out.scatter_reduce(0, idx_m, values, reduce=reduce, include_self=False)


parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("structures_db")
parser.add_argument("--limit", type=int, default=512)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--etas", type=float, nargs="+", default=[1.0, 0.5, 1.5])
args = parser.parse_args()

torch.set_float32_matmul_precision("high")
calculator = Calculator(model=load_model(args.model_path, device="cpu"), device="cuda")
with connect(args.structures_db) as db:
    atoms_list = [row.toatoms() for row in db.select(limit=args.limit)]

for eta in args.etas:
    shifts, cosines, sigmas, translations, internals = [], [], [], [], []
    rmsd_end, kabsch_end = [], []
    for start in range(0, len(atoms_list), 128):
        batch = calculator.cast_batch(to_batch(atoms_list[start:start + 128], start))
        idx_m = batch[properties.idx_m]
        n_mol = batch[properties.n_atoms].shape[0]
        batch[keys.converged] = torch.zeros(n_mol, dtype=torch.bool, device=calculator.device)
        batch[keys.nfe] = torch.zeros(n_mol, dtype=torch.int32, device=calculator.device)
        x_start = batch[properties.R].clone()
        previous = None
        s_b, c_b, g_b, t_b, i_b = [], [], [], [], []
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=idx_m.device), torch.cumsum(batch[properties.n_atoms], 0)]).cpu().numpy()
        with torch.no_grad():
            for _ in range(args.steps):
                prediction = calculator.calculate(batch, forces=True, estimate_sigma=True)
                step = eta * prediction[keys.forces] / 2
                s_b.append(per_molecule(torch.linalg.norm(step, dim=1), idx_m, "amax").cpu())
                step_np, pos_np = step.cpu().numpy(), batch[properties.R].cpu().numpy()
                t_b.append(torch.tensor([np.linalg.norm(step_np[a:b].mean(0)) for a, b in zip(ptr[:-1], ptr[1:])]))
                i_b.append(torch.tensor([np.linalg.norm(internal_step(step_np[a:b], pos_np[a:b]), axis=1).max() for a, b in zip(ptr[:-1], ptr[1:])]))
                g_b.append(per_molecule(prediction[keys.sigma_curr].squeeze(-1), idx_m, "amax").cpu())
                if previous is not None:
                    dot = per_molecule((step * previous).sum(1), idx_m, "sum")
                    norms = per_molecule(step.square().sum(1), idx_m, "sum").sqrt() * per_molecule(previous.square().sum(1), idx_m, "sum").sqrt()
                    c_b.append((dot / norms).cpu())
                else:
                    c_b.append(torch.full((n_mol,), float("nan")))
                previous = step
                batch[properties.R] = batch[properties.R] + step
        shifts.append(torch.stack(s_b)); cosines.append(torch.stack(c_b)); sigmas.append(torch.stack(g_b))
        translations.append(torch.stack(t_b)); internals.append(torch.stack(i_b))
        start_np, end_np = x_start.cpu().numpy(), batch[properties.R].cpu().numpy()
        kabsch_end.append(torch.tensor([kabsch_rmsd(start_np[a:b], end_np[a:b]) for a, b in zip(ptr[:-1], ptr[1:])]))
        sq = per_molecule((batch[properties.R] - x_start).square().sum(1), idx_m, "sum") / batch[properties.n_atoms]
        rmsd_end.append(sq.sqrt().cpu())
    shifts, cosines, sigmas, translations, internals = (
        torch.cat(t, dim=1).numpy() for t in (shifts, cosines, sigmas, translations, internals)
    )
    rmsd_end, kabsch_end = torch.cat(rmsd_end).numpy(), torch.cat(kabsch_end).numpy()
    print(f"\n== eta {eta}: {shifts.shape[1]} molecules, RMSD from start after {args.steps} steps: "
          f"median {np.median(rmsd_end):.3f} A, p90 {np.percentile(rmsd_end, 90):.3f} A; "
          f"after superposition median {np.median(kabsch_end):.3f} A, p90 {np.percentile(kabsch_end, 90):.3f} A")
    print("  step   max-step p50   p90    cos(step, previous) p50   p10    sigma-hat p50   translation p50   internal max-step p50   p90")
    for k in [0, 1, 2, 4, 9, 19, 39, args.steps - 1]:
        if k < args.steps:
            c = cosines[k]
            cs = "      -        -" if np.isnan(c).all() else f"   {np.nanmedian(c):+.2f}    {np.nanpercentile(c, 10):+.2f}"
            print(f"  {k + 1:4d}     {np.median(shifts[k]):.4f}   {np.percentile(shifts[k], 90):.4f}   {cs}          {np.median(sigmas[k]):.4f}          {np.median(translations[k]):.4f}            {np.median(internals[k]):.4f}   {np.percentile(internals[k], 90):.4f}")
