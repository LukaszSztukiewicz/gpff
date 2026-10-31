"""Why molecules lose RDKit validity under relaxation (UFF or xTB): SMILES and the bonds that moved most.

    python inspect_lost.py <out_dir> <name> [--n 10]
"""
import argparse
import os
import pickle
from io import StringIO

import numpy as np
from ase.db import connect
from ase.io import write
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from gpff.analysis import get_bond_lengths, mol_from_atoms

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("name")
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

records = pickle.load(open(os.path.join(args.out_dir, f"{args.name}.pkl"), "rb"))
relaxed = [row.toatoms() for row in connect(os.path.join(args.out_dir, f"{args.name}_relaxed.db")).select()]
lost = [i for i, r in enumerate(records) if r["valid_before"] and not r["valid_after"]]
print(f"{len(lost)} lost of {sum(r['valid_before'] for r in records)} valid")
for i in lost[: args.n]:
    r = records[i]
    atoms = relaxed[i]
    tmp = StringIO(); write(tmp, atoms, format="xyz")
    mol = Chem.MolFromXYZBlock(tmp.getvalue())
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
        n_bonds_after = mol.GetNumBonds()
        rdDetermineBonds.DetermineBondOrders(mol, charge=0, allowChargedFragments=False)
        why = f"radicals {sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())}, smiles {Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))}"
    except Exception as e:
        why = f"{type(e).__name__}: {str(e)[:70]}"
    before_bonds = sum(len(v) for v in r["bonds_before"].values())
    worst = max(((label, np.abs(r["bonds_after"][label] - r["bonds_before"][label]).max()) for label in r["bonds_before"]), key=lambda x: x[1])
    print(f"\n#{i} {r['smiles_before']}\n  bonds before {before_bonds}, connectivity after {n_bonds_after}; largest bond change {worst[0]} {worst[1]:.3f} A\n  after: {why}")
