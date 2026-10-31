"""Relax every structure of a database with a MACE model, many molecules at once on the GPU.

    python mace_relax.py <model> <structures.db> <out.db> [--fmax 0.02] [--max_steps 500]
                         [--batch_size 256] [--dtype float64] [--limit N] [--check N]

Batched copy of ASE's LBFGS (maxstep 0.2 A, H0 = 1/70 A^2/eV, no line search), one history per
molecule; a pair with y.s <= 0 is not stored, and a direction that is not downhill falls back to
steepest descent with a fresh history. A molecule stops once its largest atomic force is below
fmax (eV/A). The graph is MACE's own (AtomicData.from_config, built once per molecule) with the edges
recomputed on the GPU every step: all ordered intramolecular pairs closer than the model's r_max,
no periodicity. Each step evaluates only the molecules that have not stopped yet. Writes out.db row by row in input order, with data energy_before / energy_after
(kcal/mol), converged, steps and relaxed (False if the energy or forces became non-finite).
--check N compares energies and forces of the first N structures with MACECalculator and exits.
"""
import argparse
import os
import time

import numpy as np
import torch
from ase import units
from ase.db import connect
from mace import data as mace_data
from mace.calculators import MACECalculator
from mace.tools import torch_geometric, torch_tools

KCAL_PER_EV = 1 / (units.kcal / units.mol)


class BatchedMACE:
    """Energies and forces of a batch of molecules whose positions change."""

    def __init__(self, calculator, atoms_list, dtype, device):
        self.model = calculator.models[0]
        self.r_max = float(self.model.r_max)
        keyspec = mace_data.KeySpecification(info_keys=calculator.info_keys, arrays_keys=calculator.arrays_keys)
        with torch_tools.default_dtype(dtype):
            graphs = [
                mace_data.AtomicData.from_config(
                    mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calculator.head),
                    z_table=calculator.z_table, cutoff=self.r_max, heads=calculator.available_heads,
                )
                for atoms in atoms_list
            ]
        batch = torch_geometric.Batch.from_data_list(graphs).to(device)
        self.data = batch.to_dict()
        n_atoms = torch.tensor([len(a) for a in atoms_list], device=device)
        offsets = torch.cumsum(n_atoms, 0) - n_atoms
        # every ordered pair of distinct atoms within a molecule
        pairs = []
        for n, o in zip(n_atoms.tolist(), offsets.tolist()):
            i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
            keep = i != j
            pairs.append(torch.stack([i[keep], j[keep]]) + o)
        self.pairs = torch.cat(pairs, dim=1)
        self.batch = self.data["batch"]
        self.n_molecules = len(atoms_list)

    def __call__(self, positions, active=None):
        """
        Energies (n_molecules,) in eV and forces (n_atoms, 3) in eV/A at the given positions. With
        an active mask (n_molecules,) only those molecules are evaluated; the others get energy NaN
        and zero forces.
        """
        n_atoms = positions.shape[0]
        if active is None:
            active = torch.ones(self.n_molecules, dtype=torch.bool, device=positions.device)
        atom_mask = active[self.batch]
        graph_ids = torch.nonzero(active).squeeze(1)
        new_atom = torch.cumsum(atom_mask, 0) - 1
        new_graph = torch.full((self.n_molecules,), -1, dtype=torch.long, device=positions.device)
        new_graph[graph_ids] = torch.arange(len(graph_ids), device=positions.device)

        # pairs are intramolecular, so one active end means both
        pairs = new_atom[self.pairs[:, atom_mask[self.pairs[0]]]]
        sub_positions = positions[atom_mask]
        lengths = torch.linalg.norm(sub_positions[pairs[0]] - sub_positions[pairs[1]], dim=1)
        edge_index = pairs[:, lengths < self.r_max]
        data = {}
        for key, value in self.data.items():
            if key in ("edge_index", "shifts", "unit_shifts", "ptr", "positions"):
                continue
            if key in ("cell", "rcell"):
                data[key] = value.view(self.n_molecules, 3, 3)[graph_ids].reshape(-1, 3)
            elif value.shape[0] == n_atoms:
                data[key] = value[atom_mask]
            elif value.shape[0] == self.n_molecules:
                data[key] = value[graph_ids]
            else:
                data[key] = value
        data["batch"] = new_graph[self.batch[atom_mask]]
        counts = torch.bincount(data["batch"], minlength=len(graph_ids))
        data["ptr"] = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        data["positions"] = sub_positions.detach().clone()
        data["edge_index"] = edge_index
        data["shifts"] = torch.zeros(edge_index.shape[1], 3, dtype=positions.dtype, device=positions.device)
        data["unit_shifts"] = torch.zeros_like(data["shifts"])
        out = self.model(data, compute_force=True, training=False)

        energy = torch.full((self.n_molecules,), float("nan"), dtype=positions.dtype, device=positions.device)
        energy[graph_ids] = out["energy"].detach()
        forces = torch.zeros_like(positions)
        forces[atom_mask] = out["forces"].detach()
        return energy, forces


def scatter_dot(a, b, batch, n):
    return torch.zeros(n, dtype=a.dtype, device=a.device).index_add_(0, batch, (a * b).sum(-1))


def lbfgs(energy_forces, positions, batch, n_molecules, fmax, max_steps, memory=20, maxstep=0.2, alpha=70.0):
    """Per-molecule LBFGS on a flat (n_atoms, 3) position tensor; returns final state per molecule."""
    device, dtype = positions.device, positions.dtype
    n_atoms = positions.shape[0]
    S = torch.zeros(memory, n_atoms, 3, dtype=dtype, device=device)   # newest first
    Y = torch.zeros_like(S)
    rho = torch.zeros(memory, n_molecules, dtype=dtype, device=device)
    valid = torch.zeros(memory, n_molecules, dtype=torch.bool, device=device)
    active = torch.ones(n_molecules, dtype=torch.bool, device=device)
    converged = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    steps = torch.zeros(n_molecules, dtype=torch.long, device=device)
    energy, forces = energy_forces(positions)
    energy_start = energy.clone()
    r0 = f0 = None
    for _ in range(max_steps + 1):
        # a molecule with a non-finite energy or any non-finite force fails
        bad_atoms = ~torch.isfinite(forces).all(1)
        has_bad_atom = torch.zeros(n_molecules, dtype=torch.bool, device=device)
        has_bad_atom[batch[bad_atoms]] = True
        finite = torch.isfinite(energy) & ~has_bad_atom
        failed |= active & ~finite
        active &= finite
        max_force = torch.zeros(n_molecules, dtype=dtype, device=device).scatter_reduce(
            0, batch, torch.linalg.norm(torch.nan_to_num(forces), dim=1), reduce="amax", include_self=False)
        newly = active & (max_force < fmax)
        converged |= newly
        active &= ~newly
        if not active.any() or steps.max() >= max_steps:
            break
        atom_active = active[batch].unsqueeze(-1)

        if r0 is not None:
            s, y = positions - r0, f0 - forces
            ys = scatter_dot(y, s, batch, n_molecules)
            store = active & (ys > 1e-10)
            store_atoms = store[batch].unsqueeze(-1)
            S = torch.where(store_atoms, torch.cat([s.unsqueeze(0), S[:-1]]), S)
            Y = torch.where(store_atoms, torch.cat([y.unsqueeze(0), Y[:-1]]), Y)
            rho = torch.where(store, torch.cat([(1 / ys.clamp_min(1e-10)).unsqueeze(0), rho[:-1]]), rho)
            valid = torch.where(store, torch.cat([torch.ones_like(store).unsqueeze(0), valid[:-1]]), valid)

        # two-loop recursion; padding slots have valid = False
        q = -forces.clone()
        a = torch.zeros(memory, n_molecules, dtype=dtype, device=device)
        for i in range(memory):
            a[i] = torch.where(valid[i], rho[i] * scatter_dot(S[i], q, batch, n_molecules), 0)
            q = q - a[i][batch].unsqueeze(-1) * Y[i]
        z = q / alpha
        for i in reversed(range(memory)):
            b = torch.where(valid[i], rho[i] * scatter_dot(Y[i], z, batch, n_molecules), 0)
            z = z + S[i] * (a[i] - b)[batch].unsqueeze(-1)
        p = -z
        # not downhill: steepest descent and a fresh history
        uphill = active & (scatter_dot(p, forces, batch, n_molecules) <= 0)
        if uphill.any():
            p = torch.where(uphill[batch].unsqueeze(-1), forces / alpha, p)
            valid = valid & ~uphill
        step_len = torch.linalg.norm(p, dim=1)
        longest = torch.zeros(n_molecules, dtype=dtype, device=device).scatter_reduce(
            0, batch, step_len, reduce="amax", include_self=False)
        scale = torch.where(longest >= maxstep, maxstep / longest.clamp_min(1e-12), torch.ones_like(longest))
        dr = p * scale[batch].unsqueeze(-1)

        r0, f0 = positions.clone(), forces.clone()
        positions = torch.where(atom_active, positions + dr, positions)
        steps += active.long()
        energy_new, forces_new = energy_forces(positions, active)
        # frozen molecules keep their last energy and forces
        energy = torch.where(active, energy_new, energy)
        forces = torch.where(atom_active, forces_new, forces)
    return positions, energy_start, energy, converged, failed, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("structures_db")
    parser.add_argument("output_db")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    calculator = MACECalculator(model_paths=args.model, device=device, default_dtype=args.dtype)
    with connect(args.structures_db) as db:
        atoms_list = [row.toatoms() for row in db.select(limit=args.check or args.limit)]

    if args.check:
        model = BatchedMACE(calculator, atoms_list, args.dtype, device)
        positions = torch.tensor(np.concatenate([a.positions for a in atoms_list]), dtype=dtype, device=device)
        energy, forces = model(positions)
        reference_e, reference_f = [], []
        for atoms in atoms_list:
            atoms = atoms.copy()
            atoms.calc = calculator
            reference_e.append(atoms.get_potential_energy())
            reference_f.append(atoms.get_forces())
        de = np.abs(energy.cpu().numpy() - np.array(reference_e)).max()
        df = np.abs(forces.cpu().numpy() - np.concatenate(reference_f)).max()
        print(f"max |dE| {de:.3e} eV, max |dF| {df:.3e} eV/A over {len(atoms_list)} structures")
        return

    if os.path.exists(args.output_db):
        os.remove(args.output_db)
    start = time.time()
    totals = {"converged": 0, "failed": 0, "steps": 0}
    with connect(args.output_db) as out:
        for first in range(0, len(atoms_list), args.batch_size):
            chunk = atoms_list[first:first + args.batch_size]
            model = BatchedMACE(calculator, chunk, args.dtype, device)
            positions = torch.tensor(np.concatenate([a.positions for a in chunk]), dtype=dtype, device=device)
            positions, e0, e1, converged, failed, steps = lbfgs(
                model, positions, model.batch, model.n_molecules, args.fmax, args.max_steps)
            positions = positions.cpu().numpy()
            offset = 0
            for k, atoms in enumerate(chunk):
                relaxed = atoms.copy()
                relaxed.positions = positions[offset:offset + len(atoms)]
                offset += len(atoms)
                ok = not bool(failed[k])
                out.write(relaxed if ok else atoms, data={
                    "energy_before": float(e0[k]) * KCAL_PER_EV if ok else float("nan"),
                    "energy_after": float(e1[k]) * KCAL_PER_EV if ok else float("nan"),
                    "converged": bool(converged[k]), "steps": int(steps[k]), "relaxed": ok,
                })
            totals["converged"] += int(converged.sum()); totals["failed"] += int(failed.sum())
            totals["steps"] += int(steps.sum())
            n = first + len(chunk)
            print(f"{n}/{len(atoms_list)}  converged {totals['converged'] / n:.4f}  failed {totals['failed']}  "
                  f"mean steps {totals['steps'] / n:.1f}  {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
