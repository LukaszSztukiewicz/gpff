"""Copy the first n test molecules of a split into their own database, in test order.

    python export_reference.py <qm9_filtered.db> <split.npz> <out.db> [--n 10000]

gpff_generate draws sample i from test molecule i (the test loader keeps the split order and the
prior only replaces the positions), so row i of the output holds the DFT geometry with the
composition of sample i.
"""
import argparse
import os

import numpy as np
from ase.db import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datapath")
    parser.add_argument("split_path")
    parser.add_argument("output_db")
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    test_idx = np.load(args.split_path)["test_idx"][: args.n]
    if os.path.exists(args.output_db):
        os.remove(args.output_db)
    with connect(args.datapath) as source, connect(args.output_db) as target:
        for idx in test_idx:
            # ASE database ids start at 1, dataset indices at 0
            target.write(source.get(id=int(idx) + 1).toatoms(), data={"test_idx": int(idx)})
    print(f"wrote {len(test_idx)} test molecules to {args.output_db}")


if __name__ == "__main__":
    main()
