"""Continue saved samples with deterministic direct denoising until the forces converge.

    python refine_dd.py <model_path> <samples.db> <out.db> [--f_max 0.01] [--max_steps 200]
                        [--limit N] [--trace trace.npy]

Runs DirectDenoisingSampler with stochastic_lambda=0 from the positions in samples.db instead of a
prior draw. A molecule stops at the first step whose largest atomic force is <= f_max (that step
would move no atom further than f_max / 2 A), or after max_steps model calls. Writes the final
structures to out.db with nfe = the sample's nfe + the refinement's, and dd_nfe = the refinement's.
--trace saves, per step, the largest atomic shift of every molecule (n_steps, n_molecules), for
picking f_max; 0 once a molecule has stopped.
"""
import argparse
import os

import numpy as np
import torch
from ase.db import connect
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.utils import load_model

from gpff import keys
from gpff.analysis import get_rdkit_validity
from gpff.sample import Calculator
from gpff.sample.direct_denoising import DirectDenoisingSampler
from gpff.sample.observers import SamplerObserver
from gpff.utils import batch_to_atoms_list


class MaxShift(SamplerObserver):
    """Per step, the largest atomic displacement of every molecule."""

    def on_start(self, sampler, batch):
        self.previous = batch[properties.R].clone()
        self.shifts = []

    def on_step_end(self, batch, step):
        positions = batch[properties.R]
        shift = torch.linalg.norm(positions - self.previous.to(positions), dim=1)
        n_molecules = batch[properties.n_atoms].shape[0]
        per_molecule = torch.zeros(n_molecules, device=shift.device, dtype=shift.dtype).scatter_reduce(
            0, batch[properties.idx_m], shift, reduce="amax", include_self=False
        )
        self.shifts.append(per_molecule.cpu().numpy())
        self.previous = positions.clone()


def to_batch(atoms_list, first_idx):
    """Collate ase.Atoms into a schnetpack batch like the QM9 test loader's."""
    return _atoms_collate_fn([
        {
            properties.idx: torch.tensor([first_idx + k]),
            properties.n_atoms: torch.tensor([len(atoms)]),
            properties.Z: torch.tensor(atoms.numbers, dtype=torch.long),
            properties.R: torch.tensor(atoms.positions, dtype=torch.float32),
            properties.cell: torch.zeros(1, 3, 3),
            properties.pbc: torch.zeros(3, dtype=torch.bool),
        }
        for k, atoms in enumerate(atoms_list)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("samples_db")
    parser.add_argument("output_db")
    parser.add_argument("--f_max", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    calculator = Calculator(model=load_model(args.model_path, device="cpu"), device="cuda")
    sampler = DirectDenoisingSampler(
        calculator=calculator,
        n_steps=args.max_steps,
        stochastic_lambda=0.0,
        f_max=args.f_max,
    )

    with connect(args.samples_db) as db:
        rows = [(row.toatoms(), row.data.get(keys.nfe, 0)) for row in db.select(limit=args.limit)]

    if os.path.exists(args.output_db):
        os.remove(args.output_db)
    traces, dd_nfe, valid = [], [], []
    with connect(args.output_db) as out:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            tracer = MaxShift()
            final = sampler.denoise(to_batch([atoms for atoms, _ in chunk], start), observers=[tracer])
            # pad to max_steps so batches that stopped early stack
            trace = np.zeros((args.max_steps, len(chunk)), dtype=np.float32)
            trace[: len(tracer.shifts)] = np.stack(tracer.shifts)
            traces.append(trace)
            for atoms, (_, nfe), n in zip(batch_to_atoms_list(final), chunk, final[keys.nfe].tolist()):
                out.write(atoms, data={keys.nfe: int(nfe) + n, "dd_nfe": n})
                dd_nfe.append(n)
                valid.append(get_rdkit_validity(atoms))
            print(f"{start + len(chunk)}/{len(rows)}  mean dd_nfe {np.mean(dd_nfe):.1f}  "
                  f"stopped by f_max {np.mean(np.array(dd_nfe) < args.max_steps):.3f}  "
                  f"rdkit valid {np.mean(valid):.4f}", flush=True)

    if args.trace:
        np.save(args.trace, np.concatenate(traces, axis=1))


if __name__ == "__main__":
    main()
