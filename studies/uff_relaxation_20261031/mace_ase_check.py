"""Compare mace_relax.py's batched LBFGS with ASE's LBFGS + MACECalculator on the first n structures.

    python mace_ase_check.py <model> <structures.db> <batched_out.db> [--n 16] [--fmax 0.02]
"""
import argparse

import numpy as np
from ase.db import connect
from ase.optimize import LBFGS
from mace.calculators import MACECalculator

from mace_relax import KCAL_PER_EV

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("structures_db")
parser.add_argument("batched_db")
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--fmax", type=float, default=0.02)
args = parser.parse_args()

calculator = MACECalculator(model_paths=args.model, device="cuda", default_dtype="float64")
batched = list(connect(args.batched_db).select(limit=args.n))
diffs, rmsds = [], []
for row, b in zip(connect(args.structures_db).select(limit=args.n), batched):
    atoms = row.toatoms()
    atoms.calc = calculator
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=args.fmax, steps=500)
    e = atoms.get_potential_energy() * KCAL_PER_EV
    diffs.append(e - b.data["energy_after"])
    rmsds.append(np.sqrt(((atoms.positions - b.toatoms().positions) ** 2).sum(1).mean()))
    print(f"{row.id:3d} ASE steps {opt.nsteps:3d}  batched steps {b.data['steps']:3d}  "
          f"E_ase - E_batched {diffs[-1]:+.4f} kcal/mol  RMSD {rmsds[-1]:.4f} A")
print(f"max |dE| {np.max(np.abs(diffs)):.4f} kcal/mol, max RMSD {np.max(rmsds):.4f} A")
